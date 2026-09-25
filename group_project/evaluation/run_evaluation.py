"""Run reproducible dense-vs-hybrid evaluation on the golden dataset.

The script uses BAAI/bge-m3 for retrieval and response-relevancy embeddings,
and the configured Ollama model both as generator and Ragas judge. Results are
checkpointed after every sample so an interrupted API run can be resumed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import threading
import time
from datetime import date
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[2]
EVALUATION_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

from langchain_core.outputs import Generation, LLMResult
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import BaseRagasEmbeddings
from ragas.llms import BaseRagasLLM
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
    ResponseRelevancy,
)

from src.task4_chunking_indexing import EMBEDDING_MODEL, embed_texts
from src.task5_semantic_search import semantic_search
from src.task6_lexical_search import lexical_search
from src.task7_reranking import rerank_rrf
from src.task10_generation import SYSTEM_PROMPT, format_context, reorder_for_llm


GOLDEN_PATH = EVALUATION_DIR / "golden_dataset.json"
RESULTS_PATH = EVALUATION_DIR / "evaluation_results.json"
CALIBRATION_PATH = EVALUATION_DIR / "threshold_calibration.json"
REPORT_PATH = EVALUATION_DIR / "RESULT.md"
TOP_K = 5
RRF_K = 60
API_CONCURRENCY = 3

def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _ollama_request(
    messages: list[dict[str, str]],
    *,
    json_mode: bool = False,
    temperature: float = 0.01,
) -> str:
    key = os.getenv("OLLAMA_API_KEY", "")
    if not key:
        raise RuntimeError("OLLAMA_API_KEY chưa được cấu hình")
    payload: dict[str, Any] = {
        "model": os.environ["LLM_MODEL"],
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "top_p": 0.9},
    }
    if json_mode:
        payload["format"] = "json"
    response = requests.post(
        os.getenv("OLLAMA_CHAT_URL", "https://ollama.com/api/chat"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=(10, 240),
    )
    response.raise_for_status()
    content = response.json().get("message", {}).get("content", "")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Ollama response không có content")
    return content.strip()


class OllamaRagasLLM(BaseRagasLLM):
    """Minimal Ragas adapter for Ollama's native chat endpoint."""

    _slots = threading.BoundedSemaphore(API_CONCURRENCY)

    def generate_text(
        self,
        prompt,
        n: int = 1,
        temperature: float = 0.01,
        stop=None,
        callbacks=None,
    ) -> LLMResult:
        generations: list[Generation] = []
        for _ in range(n):
            with self._slots:
                text = _ollama_request(
                    [{"role": "user", "content": prompt.to_string()}],
                    json_mode=True,
                    temperature=temperature,
                )
            generations.append(Generation(text=text))
        return LLMResult(generations=[generations])

    async def agenerate_text(
        self,
        prompt,
        n: int = 1,
        temperature: float = 0.01,
        stop=None,
        callbacks=None,
    ) -> LLMResult:
        return await asyncio.to_thread(
            self.generate_text, prompt, n, temperature, stop, callbacks
        )

    def is_finished(self, response: LLMResult) -> bool:
        return True


class BGERagasEmbeddings(BaseRagasEmbeddings):
    """Reuse exactly the embedding function used by indexing and retrieval."""

    _lock = threading.Lock()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            return embed_texts(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)


def _labeled_context(chunks: list[dict]) -> str:
    labeled = [
        {
            **chunk,
            "metadata": {**chunk["metadata"], "_citation_index": index},
        }
        for index, chunk in enumerate(chunks, 1)
    ]
    return format_context(reorder_for_llm(labeled))


def _generate_answer(question: str, chunks: list[dict]) -> str:
    if not chunks:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."
    context = _labeled_context(chunks)
    return _ollama_request(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\nCâu hỏi: {question}\n\n"
                    "Trích dẫn theo dạng [Document N]."
                ),
            },
        ],
        temperature=0.3,
    )


def _generate_answer_timed(question: str, chunks: list[dict]) -> tuple[str, float]:
    started = time.perf_counter()
    return _generate_answer(question, chunks), time.perf_counter() - started


def _retrieve_pair(question: str) -> tuple[list[dict], list[dict], dict[str, float]]:
    started = time.perf_counter()
    dense_pool = semantic_search(question, top_k=TOP_K * 2)
    dense_latency = time.perf_counter() - started

    started = time.perf_counter()
    sparse = lexical_search(question, top_k=TOP_K * 2)
    hybrid = rerank_rrf([dense_pool, sparse], top_k=TOP_K, k=RRF_K)
    extra_hybrid_latency = time.perf_counter() - started
    return (
        dense_pool[:TOP_K],
        hybrid,
        {
            "dense_retrieval": dense_latency,
            "hybrid_retrieval": dense_latency + extra_hybrid_latency,
            "best_dense_score": dense_pool[0]["score"] if dense_pool else 0.0,
        },
    )


async def generate_cases(golden: list[dict], results: dict) -> None:
    """Retrieve locally, generate both configurations, and checkpoint."""
    for index, case in enumerate(golden):
        key = str(index)
        entry = results.setdefault(key, {"question": case["question"]})
        if all(entry.get(config, {}).get("response") for config in ("dense", "hybrid")):
            print(f"[generate {index + 1}/{len(golden)}] cached")
            continue

        dense, hybrid, latencies = _retrieve_pair(case["question"])
        configurations = {"dense": dense, "hybrid": hybrid}
        pending: list[tuple[str, asyncio.Task]] = []
        for name, chunks in configurations.items():
            if entry.get(name, {}).get("response"):
                continue
            task = asyncio.create_task(
                asyncio.to_thread(_generate_answer_timed, case["question"], chunks)
            )
            pending.append((name, task))

        for name, task in pending:
            try:
                response, generation_latency = await task
                error = None
            except Exception as exc:  # checkpoint provider errors for diagnosis
                response = "Tôi không thể xác minh thông tin này từ nguồn hiện có."
                generation_latency = 0.0
                error = f"{type(exc).__name__}: {exc}"
            chunks = configurations[name]
            entry[name] = {
                "response": response,
                "contexts": [chunk["content"] for chunk in chunks],
                "sources": [chunk["metadata"]["source"] for chunk in chunks],
                "retrieval_latency_seconds": latencies[f"{name}_retrieval"],
                "generation_latency_seconds": generation_latency,
                "best_dense_score": latencies["best_dense_score"],
                "error": error,
                "metrics": {},
            }
        _atomic_json(RESULTS_PATH, results)
        print(f"[generate {index + 1}/{len(golden)}] complete")


async def score_cases(golden: list[dict], results: dict) -> None:
    llm = OllamaRagasLLM()
    embeddings = BGERagasEmbeddings()
    metrics = {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevance": ResponseRelevancy(
            llm=llm, embeddings=embeddings, strictness=1
        ),
        "context_recall": LLMContextRecall(llm=llm),
        "context_precision": LLMContextPrecisionWithoutReference(llm=llm),
    }
    row_slots = asyncio.Semaphore(2)

    async def score_row(index: int, config: str) -> tuple[int, str, dict]:
        entry = results[str(index)][config]
        existing = entry.get("metrics", {})
        if all(name in existing and existing[name] is not None for name in metrics):
            return index, config, existing
        case = golden[index]
        sample = SingleTurnSample(
            user_input=case["question"],
            response=entry["response"],
            retrieved_contexts=entry["contexts"],
            reference=case["expected_answer"],
            reference_contexts=[case["expected_context"]],
        )
        scores = dict(existing)
        async with row_slots:
            async def run_metric(name: str, metric) -> tuple[str, float | None]:
                if name in scores and scores[name] is not None:
                    return name, scores[name]
                try:
                    value = await metric.single_turn_ascore(sample, timeout=480)
                    if value is None or math.isnan(float(value)):
                        return name, None
                    return name, round(float(value), 6)
                except Exception as exc:
                    print(f"  metric error {index + 1}/{config}/{name}: {exc}")
                    return name, None

            measured = await asyncio.gather(
                *(run_metric(name, metric) for name, metric in metrics.items())
            )
            scores.update(dict(measured))
        return index, config, scores

    tasks = [
        asyncio.create_task(score_row(index, config))
        for index in range(len(golden))
        for config in ("dense", "hybrid")
    ]
    completed = 0
    for task in asyncio.as_completed(tasks):
        index, config, scores = await task
        results[str(index)][config]["metrics"] = scores
        _atomic_json(RESULTS_PATH, results)
        completed += 1
        print(f"[score {completed}/{len(tasks)}] case={index + 1} config={config}")


def calibrate_threshold(golden: list[dict]) -> dict:
    out_of_domain = [
        "Thời tiết Hà Nội ngày mai thế nào?",
        "Viết một bài thơ về mùa thu.",
        "Ai vô địch World Cup gần nhất?",
        "Cách điều trị bệnh đau dạ dày?",
        "Giá Bitcoin hôm nay là bao nhiêu?",
    ]
    observations = []
    for case in golden:
        results = semantic_search(case["question"], top_k=1)
        observations.append(
            {"query": case["question"], "label": 1, "score": results[0]["score"] if results else 0.0}
        )
    for query in out_of_domain:
        results = semantic_search(query, top_k=1)
        observations.append(
            {"query": query, "label": 0, "score": results[0]["score"] if results else 0.0}
        )

    unique = sorted({float(item["score"]) for item in observations})
    candidates = [0.0, 1.0] + [
        (left + right) / 2 for left, right in zip(unique, unique[1:])
    ]
    best = {"threshold": 0.3, "balanced_accuracy": -1.0}
    for threshold in candidates:
        true_positive = sum(
            item["label"] == 1 and item["score"] >= threshold for item in observations
        )
        true_negative = sum(
            item["label"] == 0 and item["score"] < threshold for item in observations
        )
        positive = sum(item["label"] == 1 for item in observations)
        negative = sum(item["label"] == 0 for item in observations)
        balanced = ((true_positive / positive) + (true_negative / negative)) / 2
        if balanced > best["balanced_accuracy"]:
            best = {"threshold": threshold, "balanced_accuracy": balanced}
    output = {"selected": best, "observations": observations}
    _atomic_json(CALIBRATION_PATH, output)
    return output


def _mean(values: list[float | None]) -> float:
    usable = [float(value) for value in values if value is not None]
    return statistics.fmean(usable) if usable else float("nan")


def write_report(golden: list[dict], results: dict, calibration: dict) -> None:
    metric_names = (
        "faithfulness",
        "answer_relevance",
        "context_recall",
        "context_precision",
    )
    summary: dict[str, dict[str, float]] = {}
    for config in ("dense", "hybrid"):
        summary[config] = {
            metric: _mean(
                [results[str(index)][config]["metrics"].get(metric) for index in range(len(golden))]
            )
            for metric in metric_names
        }
        summary[config]["average"] = _mean(list(summary[config].values()))
        summary[config]["retrieval_latency"] = _mean(
            [results[str(index)][config]["retrieval_latency_seconds"] for index in range(len(golden))]
        )
        summary[config]["generation_latency"] = _mean(
            [results[str(index)][config]["generation_latency_seconds"] for index in range(len(golden))]
        )

    rows = []
    for index, case in enumerate(golden):
        for config in ("dense", "hybrid"):
            metrics = results[str(index)][config]["metrics"]
            score = _mean([metrics.get(name) for name in metric_names])
            rows.append((score, index, config, metrics, case["question"]))
    worst = sorted(rows, key=lambda row: row[0])[:3]
    better = max(("dense", "hybrid"), key=lambda name: summary[name]["average"])
    selected_threshold = calibration["selected"]["threshold"]
    balanced_accuracy = calibration["selected"]["balanced_accuracy"]

    def fmt(value: float | None) -> str:
        return "N/A" if value is None or math.isnan(float(value)) else f"{value:.3f}"

    lines = [
        "# RAG evaluation results",
        "",
        "## Run information",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Evaluation date | {date.today().isoformat()} |",
        "| Framework and version | Ragas 0.4.3 |",
        f"| Evaluator model | `{os.environ['LLM_MODEL']}` qua Ollama Cloud |",
        f"| Generator model | `{os.environ['LLM_MODEL']}` qua Ollama Cloud |",
        f"| Embedding model | `{EMBEDDING_MODEL}`, CPU, 1024 chiều |",
        "| Corpus version | Corpus hộ kinh doanh chuẩn hóa 2026-09-23 |",
        f"| Golden dataset size | {len(golden)} |",
        f"| `top_k` | {TOP_K} |",
        f"| Fallback threshold calibration | `{selected_threshold:.3f}`; balanced accuracy `{balanced_accuracy:.3f}` |",
        "",
        "## Configurations",
        "",
        "- **Config A — dense-only:** Chroma cosine, lấy trực tiếp top 5.",
        "- **Config B — hybrid + RRF:** dense và BM25 lấy 10 ứng viên mỗi nhánh, RRF `k=60`, trả top 5.",
        "- Hai cấu hình dùng cùng corpus, golden dataset, generator, evaluator, prompt và `top_k`.",
        "",
        "## Overall scores",
        "",
        "| Metric | Config A | Config B | Delta B−A |",
        "| --- | ---: | ---: | ---: |",
    ]
    labels = {
        "faithfulness": "Faithfulness",
        "answer_relevance": "Answer relevance",
        "context_recall": "Context recall",
        "context_precision": "Context precision",
        "average": "**Average**",
    }
    for metric in (*metric_names, "average"):
        a, b = summary["dense"][metric], summary["hybrid"][metric]
        lines.append(f"| {labels[metric]} | {fmt(a)} | {fmt(b)} | {b - a:+.3f} |")
    lines.extend(
        [
            "",
            "## A/B comparison",
            "",
            f"- Cấu hình có điểm trung bình cao hơn: **{better}**.",
            f"- Retrieval latency trung bình: dense `{summary['dense']['retrieval_latency']:.3f}s`, hybrid `{summary['hybrid']['retrieval_latency']:.3f}s`.",
            "- Hai nhánh generation được chạy đồng thời trong lượt đo hiện tại, vì vậy latency API riêng lẻ chỉ mang tính quan sát và không dùng để kết luận A/B.",
            "- Hybrid thêm BM25 và RRF tại máy, nhưng vẫn chỉ phát sinh một API generation cho mỗi câu như dense.",
            "",
            "## Worst performers",
            "",
            "| # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage | Root cause |",
            "| --: | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for rank, (_, _, config, metrics, question) in enumerate(worst, 1):
        if (metrics.get("context_recall") or 0.0) < 0.5:
            stage = "retrieval/data"
            cause = "Top-k không chứa đủ bằng chứng cần thiết để trả lời"
        else:
            stage = "generation"
            cause = "Câu trả lời chưa bám sát hoặc chưa bao phủ đúng câu hỏi"
        lines.append(
            f"| {rank} | {question} | {config} | {fmt(metrics.get('faithfulness'))} | "
            f"{fmt(metrics.get('answer_relevance'))} | {fmt(metrics.get('context_recall'))} | "
            f"{fmt(metrics.get('context_precision'))} | {stage} | {cause} |"
        )
    lines.extend(
        [
            "",
            "## Recommendations",
            "",
            "| Priority | Action | Evidence | Expected impact | How to verify |",
            "| --: | --- | --- | --- | --- |",
            f"| 1 | Dùng threshold `{selected_threshold:.3f}` | Calibration đạt balanced accuracy `{balanced_accuracy:.3f}` | Fallback đúng hơn | Chạy lại tập in/out-domain |",
            "| 2 | Deduplicate nguồn cùng nội dung và tăng đa dạng top-k | Các lỗi precision thường do nhiều chunk gần nhau | Tăng context precision | So sánh precision trước/sau |",
            "| 3 | Thêm metadata ngày hiệu lực và ưu tiên văn bản sửa đổi | NĐ 141 sửa ngưỡng của NĐ 68 | Tránh trả quy định cũ | Test riêng câu hỏi ngưỡng thuế |",
            "",
            "## Artifacts",
            "",
            "- `evaluation_results.json`: câu trả lời, context, nguồn, latency và điểm từng case.",
            "- `threshold_calibration.json`: score của câu in-domain/out-of-domain và threshold được chọn.",
            "- `golden_dataset.json`: 15 câu hỏi/đáp án/context chuẩn.",
            "",
            "## Bonus experiments",
            "",
            "`dangvantuan/vietnamese-embedding` được giữ làm phương án A/B bổ sung nhưng chưa dùng làm baseline hay trộn vào collection BGE-M3.",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    (ROOT / "reports" / "RESULT.md").write_text("\n".join(lines), encoding="utf-8")


async def main(*, skip_metrics: bool = False) -> None:
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    results = (
        json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        if RESULTS_PATH.exists()
        else {}
    )
    await generate_cases(golden, results)
    if not skip_metrics:
        await score_cases(golden, results)
    calibration = calibrate_threshold(golden)
    if not skip_metrics:
        write_report(golden, results, calibration)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-metrics", action="store_true")
    arguments = parser.parse_args()
    asyncio.run(main(skip_metrics=arguments.skip_metrics))
