
from __future__ import annotations
import argparse
import json
import pandas as pd

from pathlib import Path
import csv
from typing import Optional
from typing import Dict, List


"""
SERO CSV Cleaner
----------------

Usage:
    python sero_csv_cleaner.py \
        --raw-dir "SeroProjectCode/SeroProjectCode/data/Raw" \
        --processed-dir "SeroProjectCode/SeroProjectCode/data/proccessed"

What it does:
- Loads every *.csv in --raw-dir
- Cleans each file:
  1) Trims whitespace-only cells -> NaN
  2) Drops fully-empty rows
  3) Drops duplicate rows
  4) Drops rows with ANY missing values
  5) For files whose name matches one of:
        ['careplan','obsvervation','observations','questionnaireresponses','supportcareplan']
     drops columns named exactly 'date' and 'time' (case-insensitive)
- Saves cleaned CSVs to --processed-dir (created if missing)
- Writes a cleaning summary CSV + JSON in --processed-dir

Notes:
- Reading is done with dtype=str to preserve original formatting.
- Missing values recognized: "", " ", "NA", "NaN", "null", "None" (case-insensitive).
"""


TARGET_BASENAMES = {
    "careplan",            # keep legacy misspelling to match existing files
    "observation",             # NEW: also drop date/time for 'observation'
    "questionnaireresponses",
    "supportcareplan",
}

DROP_COLS = {"date", "time"}  # exact names, case-insensitive

# -------- Delimiter detection & normalization helpers --------

CANDIDATE_DELIMITERS = [",", ";", "\t", "|"]

def detect_delimiter(path: Path, sample_bytes: int = 256 * 1024) -> str:
    """
    Detect the delimiter of a CSV file using csv.Sniffer on a sample.
    Falls back to counting candidate characters if sniffing fails.
    Returns a single-character delimiter (defaults to ',').
    """
    try:
        with path.open("rb") as fb:
            raw = fb.read(sample_bytes)
        sample = raw.decode("utf-8", errors="ignore")
    except Exception:
        sample = path.read_text(encoding="utf-8", errors="ignore")

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="".join(CANDIDATE_DELIMITERS))
        delim = dialect.delimiter
    except Exception:
        counts = {d: sample.count(d) for d in CANDIDATE_DELIMITERS}
        delim = max(counts, key=counts.get) if any(counts.values()) else ","

    if sample.count(delim) == 0:
        return ","
    return delim

def normalize_folder_to_comma(src_dir: Path, dst_dir: Optional[Path] = None, dry_run: bool = False) -> pd.DataFrame:
    """
    Ensure every *.csv under `src_dir` is comma-separated.
    - If `dst_dir` is None: rewrite in-place when needed.
    - If `dst_dir` is provided: write normalized copies there (preserving subfolders).
    - If `dry_run` is True: no writes; return a summary DataFrame of what would change.
    """
    src_dir = Path(src_dir)
    files = sorted([p for p in src_dir.rglob("*.csv") if p.is_file()])
    records = []

    for path in files:
        delim = detect_delimiter(path)
        df = pd.read_csv(
            path,
            sep=delim,
            dtype=str,
            keep_default_na=True,
            na_values=["", " ", "NA", "NaN", "null", "Null", "NULL", "None", "none"],
            encoding="utf-8",
            on_bad_lines="skip",
            engine="python",
            quoting=csv.QUOTE_MINIMAL,
        )

        if dst_dir is None:
            out_path = path
        else:
            out_path = Path(dst_dir) / path.relative_to(src_dir)
            out_path.parent.mkdir(parents=True, exist_ok=True)

        will_change = (delim != ",") or (dst_dir is not None and out_path != path)

        if not dry_run:
            if will_change:
                df.to_csv(out_path, index=False, encoding="utf-8", lineterminator="\n")
            else:
                out_path.parent.mkdir(parents=True, exist_ok=True)

        records.append({
            "filename": path.name,
            "input_path": str(path),
            "output_path": str(out_path),
            "original_delimiter": delim,
            "will_change": bool(will_change),
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
        })

    return pd.DataFrame.from_records(records)

def normalize_csvs_workflow(
    raw_dir: str | Path = "SeroProjectCode/SeroProjectCode/data/Raw",
    copy_to: str | Path = "SeroProjectCode/SeroProjectCode/data/normalized",
    do_preview: bool = True,
    do_copy_to_folder: bool = True
):
    """
    Convenience wrapper around `normalize_folder_to_comma`:
      1) Preview (no writes)
      2) Write normalized copies to `copy_to` (comma-separated)

    Returns a dict with DataFrames for the steps that ran:
      {
        "preview": <DataFrame>,
        "copy": <DataFrame>,
        "inplace": <DataFrame>
      }
    """
    raw_dir = Path(raw_dir)
    results = {}

    if do_preview:
        preview_df = normalize_folder_to_comma(raw_dir, dry_run=True)
        changed = preview_df[preview_df["will_change"]].shape[0]
        total = preview_df.shape[0]
        print(f"[Preview] {changed}/{total} file(s) would be rewritten to comma-separated.")
        results["preview"] = preview_df

    if do_copy_to_folder:
        dst_dir = Path(copy_to)
        copy_df = normalize_folder_to_comma(raw_dir, dst_dir=dst_dir)
        changed = copy_df[copy_df["will_change"]].shape[0]
        total = copy_df.shape[0]
        print(f"[Copy] Wrote normalized copies to: {dst_dir}  |  changed: {changed}/{total}")
        results["copy"] = copy_df

    return results


def should_drop_date_time(file_stem_lower: str) -> bool:
    # If the file name contains any of the target basenames, drop date/time
    return any(name in file_stem_lower for name in TARGET_BASENAMES) # Returns True if any target basename is found in the file stem 


def clean_dataframe(df: pd.DataFrame, drop_date_time: bool) -> Dict[str, int | List[str]]:
    """Clean a dataframe in-place per spec and return metrics."""
    metrics: Dict[str, int | List[str]] = {}

    # Normalize whitespace-only cells to NaN (works on string/object columns)
    df = df.replace(r'^\s*$', pd.NA, regex=True)

    # Count before any removal
    n_before = len(df)

    # Drop fully-empty rows
    df_noempty = df.dropna(how="all")
    empty_rows_removed = n_before - len(df_noempty)

    # Drop duplicate rows (exact duplicates across all columns)
    df_nodup = df_noempty.drop_duplicates()
    duplicates_removed = len(df_noempty) - len(df_nodup)

    # Drop rows with ANY missing values
    df_nomiss = df_nodup.dropna(how="any")
    missing_rows_removed = len(df_nodup) - len(df_nomiss)

    # Optionally drop 'date' and 'time' columns (exact names, case-insensitive)
    dropped_columns: List[str] = []
    if drop_date_time:
        cols_lower = {c.lower(): c for c in df_nomiss.columns}
        for target in DROP_COLS:
            if target in cols_lower:
                original_name = cols_lower[target]
                df_nomiss = df_nomiss.drop(columns=[original_name])
                dropped_columns.append(original_name)

    metrics.update(
        n_rows_before=n_before,
        n_empty_rows_removed=empty_rows_removed,
        n_duplicates_removed=duplicates_removed,
        n_missing_rows_removed=missing_rows_removed,
        n_rows_after=len(df_nomiss),
        columns_dropped=dropped_columns,
    )

    return df_nomiss, metrics


def process_file(file_path: Path, processed_dir: Path) -> Dict[str, object]:
    stem_lower = file_path.stem.lower()
    drop_dt = should_drop_date_time(stem_lower)

    # Detect delimiter and read accordingly
    delim = detect_delimiter(file_path)
    df = pd.read_csv(
        file_path,
        sep=delim,
        dtype=str,
        keep_default_na=True,
        na_values=["", " ", "NA", "NaN", "null", "Null", "NULL", "None", "none"],
        encoding="utf-8",
        on_bad_lines="skip",
        engine="python",
        quoting=csv.QUOTE_MINIMAL,
    )

    cleaned_df, metrics = clean_dataframe(df, drop_dt)

    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / file_path.name
    cleaned_df.to_csv(out_path, index=False, encoding="utf-8")

    record = {
        "filename": file_path.name,
        "input_path": str(file_path),
        "output_path": str(out_path),
        **metrics,
    }
    return record


def main():
    parser = argparse.ArgumentParser(description="Clean SERO CSVs per spec.")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="SeroProjectCode/SeroProjectCode/data/Raw",
        help="Directory containing raw CSV files.",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="SeroProjectCode/SeroProjectCode/data/proccessed",
        help="Directory to write cleaned CSV files.",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)

    if not raw_dir.exists():
        raise SystemExit(f"Raw directory not found: {raw_dir}")

    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSV files found in: {raw_dir}")

    summary_records: List[Dict[str, object]] = []
    for csv_path in csv_files:
        try:
            rec = process_file(csv_path, processed_dir)
            summary_records.append(rec)
            print(f"[OK] {csv_path.name}: {rec['n_rows_before']} -> {rec['n_rows_after']} rows")
        except Exception as e:
            print(f"[ERROR] Failed to process {csv_path.name}: {e}")

    # Build summary DataFrame
    if summary_records:
        summary_df = pd.DataFrame(summary_records)[
            [
                "filename",
                "n_rows_before",
                "n_empty_rows_removed",
                "n_duplicates_removed",
                "n_missing_rows_removed",
                "n_rows_after",
                "columns_dropped",
                "input_path",
                "output_path",
            ]
        ]
        summary_csv = processed_dir / "cleaning_summary.csv"
        summary_json = processed_dir / "cleaning_summary.json"
        summary_df.to_csv(summary_csv, index=False, encoding="utf-8")
        summary_df.to_json(summary_json, orient="records", force_ascii=False, indent=2)
        print(f"\nSummary written to:\n  - {summary_csv}\n  - {summary_json}")
    else:
        print("No files processed; no summary written.")


if __name__ == "__main__":
    main()
