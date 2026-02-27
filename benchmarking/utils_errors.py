from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Iterable

_OOM_PATTERNS = (
    "out of memory",
    "cuda out of memory",
    "cublas",
    "cudnn",
    "cuda error",
    "memoryerror",
    "std::bad_alloc",
    "cannot allocate memory",
    "allocation failed",
    "alloc failed",
)


def is_oom_error(error_type: str | None, error_msg: str | None) -> bool:
    if not error_type and not error_msg:
        return False
    haystack = f"{error_type or ''} {error_msg or ''}".lower()
    return any(token in haystack for token in _OOM_PATTERNS)


def normalize_error_signature(error_type: str | None, error_msg: str | None) -> str:
    if error_type:
        base = f"{error_type}: {error_msg or ''}"
    else:
        base = error_msg or ""
    text = base.strip().lower()
    if not text:
        return "unknown"

    # Collapse whitespace and remove common noisy tokens.
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"0x[0-9a-f]+", "<hex>", text)

    # Normalize pgmpy-style unseen state errors.
    text = re.sub(r"state ['\"][^'\"]+['\"]", "state <state>", text)
    text = re.sub(
        r"(pgmpy output for) ['\"][^'\"]+['\"]",
        r"\1 <var>",
        text,
    )

    # Normalize generic variable/node references.
    text = re.sub(
        r"\b(node|variable|var) ['\"][^'\"]+['\"]",
        r"\1 <var>",
        text,
    )

    # Normalize common variable id patterns like n6_d_g.
    text = re.sub(r"\bn\d+[a-z0-9_]*\b", "<var>", text)

    # Replace long numeric sequences.
    text = re.sub(r"\d{3,}", "<num>", text)

    text = re.sub(r"\s+", " ", text).strip()
    return text or "unknown"


def classify_error(error_type: str | None, error_msg: str | None) -> dict:
    signature = normalize_error_signature(error_type, error_msg)
    return {
        "is_oom": bool(is_oom_error(error_type, error_msg)),
        "error_signature": signature,
    }


class ErrorSummary:
    def __init__(self, *, max_examples: int = 3) -> None:
        self.max_examples = int(max_examples)
        self.counts = {
            "error_type": Counter(),
            "error_signature": Counter(),
            "error_stage": Counter(),
            "is_oom": Counter(),
        }
        self.by_model: dict[str, dict[str, Counter | int]] = defaultdict(
            lambda: {
                "error_type": Counter(),
                "error_signature": Counter(),
                "error_stage": Counter(),
                "is_oom": Counter(),
                "total": 0,
            }
        )
        self.by_problem: dict[str, dict[str, Counter | int]] = defaultdict(
            lambda: {
                "error_type": Counter(),
                "error_signature": Counter(),
                "error_stage": Counter(),
                "is_oom": Counter(),
                "total": 0,
            }
        )
        self.examples: dict[str, list[str]] = defaultdict(list)

    def add(
        self,
        *,
        model: str | None,
        problem: str | None,
        error_type: str | None,
        error_signature: str | None,
        error_stage: str | None,
        error_msg: str | None,
        is_oom: bool | None,
    ) -> None:
        model_key = model or "unknown"
        problem_key = problem or "unknown"
        err_type = error_type or "UnknownError"
        err_sig = error_signature or "unknown"
        err_stage = error_stage or "unknown"
        oom_key = "oom" if is_oom else "other"

        self.counts["error_type"][err_type] += 1
        self.counts["error_signature"][err_sig] += 1
        self.counts["error_stage"][err_stage] += 1
        self.counts["is_oom"][oom_key] += 1

        model_bucket = self.by_model[model_key]
        model_bucket["total"] = int(model_bucket["total"]) + 1
        model_bucket["error_type"][err_type] += 1
        model_bucket["error_signature"][err_sig] += 1
        model_bucket["error_stage"][err_stage] += 1
        model_bucket["is_oom"][oom_key] += 1

        problem_bucket = self.by_problem[problem_key]
        problem_bucket["total"] = int(problem_bucket["total"]) + 1
        problem_bucket["error_type"][err_type] += 1
        problem_bucket["error_signature"][err_sig] += 1
        problem_bucket["error_stage"][err_stage] += 1
        problem_bucket["is_oom"][oom_key] += 1

        if error_msg:
            examples = self.examples[err_sig]
            if len(examples) < self.max_examples and error_msg not in examples:
                examples.append(str(error_msg))

    @staticmethod
    def _counter_payload(counter: Counter) -> dict:
        return dict(counter.most_common())

    def _bucket_payload(self, bucket: dict[str, Counter | int]) -> dict:
        return {
            "total": int(bucket.get("total", 0)),
            "error_type": self._counter_payload(bucket["error_type"]),
            "error_signature": self._counter_payload(bucket["error_signature"]),
            "error_stage": self._counter_payload(bucket["error_stage"]),
            "is_oom": self._counter_payload(bucket["is_oom"]),
        }

    def to_dict(self) -> dict:
        return {
            "counts": {
                "error_type": self._counter_payload(self.counts["error_type"]),
                "error_signature": self._counter_payload(
                    self.counts["error_signature"]
                ),
                "error_stage": self._counter_payload(self.counts["error_stage"]),
                "is_oom": self._counter_payload(self.counts["is_oom"]),
            },
            "by_model": {
                key: self._bucket_payload(bucket)
                for key, bucket in sorted(self.by_model.items())
            },
            "by_problem": {
                key: self._bucket_payload(bucket)
                for key, bucket in sorted(self.by_problem.items())
            },
            "examples": {key: list(values) for key, values in self.examples.items()},
        }


def _render_md_table(headers: Iterable[str], rows: Iterable[Iterable[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, divider, *body_lines])


def render_error_summary_md(summary: dict, *, top_k: int = 10) -> str:
    lines: list[str] = ["# Error Summary", ""]
    counts = summary.get("counts", {}) if isinstance(summary, dict) else {}

    def _add_count_section(title: str, key: str) -> None:
        data = counts.get(key, {}) if isinstance(counts, dict) else {}
        if not isinstance(data, dict) or not data:
            return
        items = list(data.items())[:top_k]
        lines.append(f"## {title}")
        lines.append("")
        table = _render_md_table(
            [key, "count"],
            [[str(label), str(count)] for label, count in items],
        )
        lines.append(table)
        lines.append("")

    _add_count_section("Counts by Error Type", "error_type")
    _add_count_section("Counts by Error Signature", "error_signature")
    _add_count_section("Counts by Error Stage", "error_stage")
    _add_count_section("OOM vs Other", "is_oom")

    def _add_bucket_section(title: str, bucket_key: str) -> None:
        bucket = summary.get(bucket_key, {}) if isinstance(summary, dict) else {}
        if not isinstance(bucket, dict) or not bucket:
            return
        lines.append(f"## {title}")
        lines.append("")
        rows = []
        for name, payload in list(bucket.items())[:top_k]:
            if not isinstance(payload, dict):
                continue
            total = payload.get("total", 0)
            top_types = payload.get("error_type", {})
            top_type = None
            if isinstance(top_types, dict) and top_types:
                top_type = next(iter(top_types.items()))
            if top_type:
                top_type_label = f"{top_type[0]} ({top_type[1]})"
            else:
                top_type_label = "-"
            rows.append([str(name), str(total), str(top_type_label)])
        if rows:
            lines.append(_render_md_table(["name", "total", "top_error_type"], rows))
            lines.append("")

    _add_bucket_section("Errors by Model", "by_model")
    _add_bucket_section("Errors by Problem", "by_problem")

    examples = summary.get("examples", {}) if isinstance(summary, dict) else {}
    if isinstance(examples, dict) and examples:
        lines.append("## Examples")
        lines.append("")
        for signature, msgs in list(examples.items())[:top_k]:
            lines.append(f"- {signature}")
            if isinstance(msgs, list):
                for msg in msgs:
                    lines.append(f"  - {msg}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"
