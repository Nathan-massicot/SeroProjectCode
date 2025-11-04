#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
anonymising.py
- Detect PERSON entities with a multilingual Transformers NER model (mBERT/XLM-R).
- Replace every detected name with "***".
- Process specific CSV files/columns.

Env vars you can set (optional):
    export SERO_NER_MODEL="Davlan/bert-base-multilingual-cased-ner-hrl"
    # or:
    export SERO_NER_MODEL="Davlan/xlm-roberta-base-ner-hrl"
"""

import os
import csv
from typing import List, Optional

# Silence tokenizers’ parallelism and hard-disable TF/Flax so Transformers won’t try to import them
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

DEFAULT_MODEL = os.getenv(
    "SERO_NER_MODEL",
    "Davlan/bert-base-multilingual-cased-ner-hrl"
)

try:
    from transformers import pipeline
except Exception as e:
    raise RuntimeError(
        "Transformers is not installed or failed to import. "
        "Install dependencies first: pip install --upgrade transformers torch numpy"
    ) from e

# Create NER pipeline (PyTorch backend)
try:
    NER_PIPE = pipeline(
        task="token-classification",
        model=DEFAULT_MODEL,
        aggregation_strategy="simple"  # returns grouped spans with start/end
    )
except Exception as e:
    raise RuntimeError(
        f"Failed to load HF NER model '{DEFAULT_MODEL}'. "
        "Check your internet access or set SERO_NER_MODEL to another model."
    ) from e

PERSON_TAGS = {"PER", "PERSON"}  # model-dependent tag names


def anonymize_text(text: str) -> str:
    """Replace every PERSON entity with *** (no gender, no IDs)."""
    if not text:
        return text

    ents = NER_PIPE(text)  # [{'start':..,'end':..,'entity_group':..,'word':..}, ...]
    spans = []
    for ent in ents:
        group = (ent.get("entity_group") or ent.get("entity") or "").upper()
        if (group in PERSON_TAGS) or any(tag in group for tag in PERSON_TAGS):
            spans.append((int(ent["start"]), int(ent["end"])))

    if not spans:
        return text

    # Replace from the end to keep offsets valid
    spans.sort(key=lambda x: x[0], reverse=True)
    for start, end in spans:
        text = text[:start] + "***" + text[end:]
    return text


def _resolve_col_key(fieldnames: List[str], target: Optional[str]) -> Optional[str]:
    """Case-insensitive header resolution; returns the real header or None."""
    if not target or not fieldnames:
        return None
    lower_map = {fn.lower(): fn for fn in fieldnames}
    return lower_map.get(target.lower())


def process_csv(input_path: str, output_path: str, col_to_anonymize: Optional[str]):
    """Read CSV, anonymize only the target column, write CSV."""
    with open(input_path, newline='', encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames or []
        rows = []

        col_key = _resolve_col_key(fieldnames, col_to_anonymize)
        for row in reader:
            if col_key and row.get(col_key):
                row[col_key] = anonymize_text(row[col_key])
            rows.append(row)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline='', encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] {input_path} → {output_path} (col={col_key or '—'})")


def main():
    BASE_DIR = "data"
    RAW_DIR = os.path.join(BASE_DIR, "proccessed")   # keep your folder name as-is
    ANO_DIR = os.path.join(BASE_DIR, "Anonymised")

    files_config = [
        ("SERO-careplan.csv", "description"),
        ("SERO-QuestionnaireResponses.csv", "answer"),
        ("SERO-SupportCareplan.csv", "description"),
        ("SERO-events.csv", None),
        ("SERO-Observations.csv", None),
    ]

    for fname, col in files_config:
        input_path = os.path.join(RAW_DIR, fname)
        output_path = os.path.join(ANO_DIR, fname)
        if not os.path.exists(input_path):
            print(f"[WARN] {input_path} not found, skipping.")
            continue
        process_csv(input_path, output_path, col)


if __name__ == "__main__":
    main()