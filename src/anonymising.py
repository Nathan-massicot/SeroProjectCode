#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anonymiser (Transformers-only, regex-free)
=========================================

• Detects PERSON entities with a multilingual Hugging Face NER model
  (default: Davlan/bert-base-multilingual-cased-ner-hrl) and replaces
  each detected span with ***.
• Processes specific CSV files/columns (defaults match your project) and
  supports CLI overrides.
• Pure Transformers/PyTorch path (TensorFlow/Flax explicitly disabled).
• Robust on long texts via safe character chunking with overlap; merges
  spans and replaces from the end to preserve offsets.

Usage (defaults):
    python anonymising.py \
        --raw-dir "SeroProjectCode/SeroProjectCode/data/proccessed" \
        --out-dir "SeroProjectCode/SeroProjectCode/data/Anonymised"

Optional flags:
    --model MODEL_NAME_OR_PATH
    --careplan SERO-careplan.csv --careplan-col description
    --qresp SERO-QuestionnaireResponses.csv --qresp-col answer
    --support SERO-SupportCareplan.csv --support-col description
    --events SERO-events.csv --obs SERO-Observations.csv
    --dry-run

Environment (optional):
    export SERO_NER_MODEL="Davlan/bert-base-multilingual-cased-ner-hrl"
    export TRANSFORMERS_OFFLINE=1  # if the model is already cached

Dependencies (examples that worked on macOS ARM64 / Python 3.12):
    pip install "transformers==4.57.1" "torch==2.9.0" safetensors
"""

from __future__ import annotations

import os
import csv
import argparse
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Ensure Transformers won't try to import TF/Flax paths
# ---------------------------------------------------------------------------
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import pipeline  # type: ignore

# ---------------------------------------------------------------------------
# Model / NER pipeline
# ---------------------------------------------------------------------------
DEFAULT_MODEL = os.getenv(
    "SERO_NER_MODEL",
    "Davlan/bert-base-multilingual-cased-ner-hrl",
)

# We don't force device_map; HF will pick MPS on Apple Silicon automatically.
NER_PIPE = pipeline(
    task="token-classification",
    model=DEFAULT_MODEL,
    aggregation_strategy="simple",  # group B-PER/I-PER into one span
)

PERSON_TAGS = {"PER", "PERSON"}  # model-dependent tag names

# ---------------------------------------------------------------------------
# Chunking & span merging (robust on long texts)
# ---------------------------------------------------------------------------

def _iter_char_chunks(text: str, max_chars: int = 1400, overlap: int = 120):
    """Yield (start_index, chunk_text) pairs.
    Character-based chunking with overlap; try to cut on sentence/space
    boundaries to reduce entity splits across chunks.
    """
    n = len(text)
    if n <= max_chars:
        yield 0, text
        return

    start = 0
    while start < n:
        hard_end = min(n, start + max_chars)
        end = hard_end
        window = text[start:hard_end]
        for sep in (". ", "! ", "? ", "\n", " "):
            cut = window.rfind(sep)
            if cut != -1 and (start + cut) > start + 200:
                end = start + cut + len(sep)
                break
        yield start, text[start:end]
        if end >= n:
            break
        start = max(end - overlap, start + 1)


def _merge_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge overlapping/contiguous spans. Input needn't be sorted."""
    if not spans:
        return []
    spans = sorted(spans, key=lambda x: x[0])
    merged = [spans[0]]
    for s, e in spans[1:]:
        ls, le = merged[-1]
        if s <= le:  # overlap/contiguous
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged

# ---------------------------------------------------------------------------
# Core anonymisation
# ---------------------------------------------------------------------------

def detect_person_spans(text: str) -> List[Tuple[int, int]]:
    """Return global char spans for PERSON entities across chunked text."""
    if not text:
        return []
    spans: List[Tuple[int, int]] = []
    for base, chunk in _iter_char_chunks(text):
        ents = NER_PIPE(chunk)
        for ent in ents:
            group = (ent.get("entity_group") or ent.get("entity") or "").upper()
            if (group in PERSON_TAGS) or any(tag in group for tag in PERSON_TAGS):
                start = base + int(ent["start"])  # local → global
                end = base + int(ent["end"])      # local → global
                spans.append((start, end))
    return _merge_spans(spans)


def anonymize_text(text: str) -> str:
    """Replace every PERSON span by *** (no gender, no IDs)."""
    if not text:
        return text
    spans = detect_person_spans(text)
    if not spans:
        return text
    new_text = text
    for s, e in reversed(spans):
        new_text = new_text[:s] + "***" + new_text[e:]
    return new_text

# ---------------------------------------------------------------------------
# CSV processing
# ---------------------------------------------------------------------------

def _resolve_col_key(fieldnames: List[str] | None, target: Optional[str]) -> Optional[str]:
    """Case-insensitive header resolution; returns the real header or None."""
    if not target or not fieldnames:
        return None
    lower_map = {fn.lower(): fn for fn in fieldnames}
    return lower_map.get(target.lower())


def process_csv(input_path: str, output_path: str, col_to_anonymize: Optional[str], *, dry_run: bool = False) -> Tuple[int, int]:
    """Read CSV, anonymize only the target column, write CSV.
    Returns (rows_processed, rows_anonymised).
    """
    with open(input_path, newline='', encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames or []
        rows: List[dict] = []
        anonymised_rows = 0

        col_key = _resolve_col_key(fieldnames, col_to_anonymize)
        for row in reader:
            if col_key and row.get(col_key):
                original = row[col_key]
                new_val = anonymize_text(original)
                if new_val != original:
                    anonymised_rows += 1
                row[col_key] = new_val
            rows.append(row)

    if dry_run:
        print(f"[DRY] {input_path} → {output_path} (col={col_key or '—'}) rows={len(rows)} anonymised={anonymised_rows}")
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", newline='', encoding="utf-8") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[OK]  {input_path} → {output_path} (col={col_key or '—'}) rows={len(rows)} anonymised={anonymised_rows}")

    return len(rows), anonymised_rows

# ---------------------------------------------------------------------------
# CLI & defaults
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SERO text anonymiser (PERSON → ***)")
    p.add_argument("--raw-dir", default="data/proccessed")
    p.add_argument("--out-dir", default="data/Anonymised")
    p.add_argument("--model", default=DEFAULT_MODEL, help="HF model name or local path")
    p.add_argument("--dry-run", action="store_true", help="don’t write outputs, just log")

    # File names (override if needed)
    p.add_argument("--careplan", default="SERO-careplan.csv")
    p.add_argument("--careplan-col", default="description")
    p.add_argument("--qresp", default="SERO-QuestionnaireResponses.csv")
    p.add_argument("--qresp-col", default="answer")
    p.add_argument("--support", default="SERO-SupportCareplan.csv")
    p.add_argument("--support-col", default="description")
    p.add_argument("--events", default="SERO-events.csv")
    p.add_argument("--obs", default="SERO-Observations.csv")

    return p.parse_args()


def main():
    args = parse_args()

    # If a different model is passed via CLI, rebuild the pipeline once.
    global NER_PIPE
    if args.model and args.model != DEFAULT_MODEL:
        NER_PIPE = pipeline(
            task="token-classification",
            model=args.model,
            aggregation_strategy="simple",
        )

    base_in = args.raw_dir
    base_out = args.out_dir
    dry = args.dry_run

    files_config = [
        (args.careplan, args.careplan_col),
        (args.qresp, args.qresp_col),
        (args.support, args.support_col),
        (args.events, None),  # untouched (as requested)
        (args.obs, None),     # untouched
    ]

    total_rows = 0
    total_anonymised = 0

    for fname, col in files_config:
        in_path = os.path.join(base_in, fname)
        out_path = os.path.join(base_out, fname)
        if not os.path.exists(in_path):
            print(f"[WARN] {in_path} not found, skipping.")
            continue
        rows, ann = process_csv(in_path, out_path, col, dry_run=dry)
        total_rows += rows
        total_anonymised += ann

    print(f"[DONE] rows={total_rows} anonymised_rows={total_anonymised}")


if __name__ == "__main__":
    main()