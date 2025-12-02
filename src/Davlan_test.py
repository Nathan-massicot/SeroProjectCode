"test model from the NER_test.csv file."

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import math
from anonymising import anonymize_text  # importe ta fonction

CSV_PATH = "/Users/nathanmassicot/Library/Mobile Documents/com~apple~CloudDocs/Archives/BFH/BFH S1/SERO project/SeroProjectCode/data/NER-eval.csv/NER_test.csv"
TEXT_COL = "description"
LABEL_COL = "has_person"


def _shorten(text: str, max_len: int = 160) -> str:
    """Raccourcit le texte pour l'affichage dans le terminal."""
    text = text.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def evaluate_has_person(csv_path: str, text_col: str, label_col: str):
    """
    Compare:
      - y_true = has_person (0/1, annoté à la main)
      - y_pred = 1 if anonymize_text(description) had modify the text, sinon 0

    print TP, FP, FN, TN + accuracy, precision, recall, F1 in the terminal,
    and list the TP / FP lines from the CSV.
    """
    tp = fp = fn = tn = 0

    # pour inspecter les lignes
    tp_rows = []
    fp_rows = []

    with open(csv_path, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Sanity check sur les colonnes
        fieldnames = reader.fieldnames or []
        if text_col not in fieldnames:
            raise ValueError(f"Column {text_col!r} not found in CSV (got {fieldnames})")
        if label_col not in fieldnames:
            raise ValueError(f"Column {label_col!r} not found in CSV (got {fieldnames})")

        for row in reader:
            text = row.get(text_col, "") or ""

            # Vérité terrain (0 ou 1)
            raw_label = str(row.get(label_col, "")).strip().lower()
            if raw_label in ("1", "true", "yes", "y"):
                y_true = 1
            else:
                y_true = 0

            # Prédiction du modèle : est-ce qu'il a anonymisé quelque chose ?
            anonymized = anonymize_text(text)
            y_pred = 1 if anonymized != text else 0

            # Mise à jour des compteurs + stockage des lignes intéressantes
            if y_true == 1 and y_pred == 1:
                tp += 1
                tp_rows.append(
                    {
                        "id": row.get("id"),
                        "y_true": y_true,
                        "y_pred": y_pred,
                        "description": text,
                    }
                )
            elif y_true == 0 and y_pred == 1:
                fp += 1
                fp_rows.append(
                    {
                        "id": row.get("id"),
                        "y_true": y_true,
                        "y_pred": y_pred,
                        "description": text,
                    }
                )
            elif y_true == 1 and y_pred == 0:
                fn += 1
            else:  # y_true == 0 and y_pred == 0
                tn += 1

    total = tp + fp + fn + tn

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if not math.isnan(precision) and not math.isnan(recall) and (precision + recall) > 0
        else float("nan")
    )
    accuracy = (tp + tn) / total if total > 0 else float("nan")

    print("=== NER evaluation on has_person ===")
    print(f"CSV path   : {csv_path}")
    print(f"Text col   : {text_col}")
    print(f"Label col  : {label_col}")
    print()
    print(f"TP (true positive) : {tp}")
    print(f"FP (false positive): {fp}")
    print(f"FN (false negative): {fn}")
    print(f"TN (true negative) : {tn}")
    print(f"Total              : {total}")
    print()
    print(f"Accuracy : {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1       : {f1:.3f}")

    # --- Affichage des lignes TP / FP ---
    print("\n--- True Positives (label=1, pred=1) ---")
    for r in tp_rows:
        print(
            f"[id={r['id']}] y_true={r['y_true']} y_pred={r['y_pred']} "
            f"description={_shorten(r['description'])}"
        )

    print("\n--- False Positives (label=0, pred=1) ---")
    for r in fp_rows:
        print(
            f"[id={r['id']}] y_true={r['y_true']} y_pred={r['y_pred']} "
            f"description={_shorten(r['description'])}"
        )


if __name__ == "__main__":
    evaluate_has_person(CSV_PATH, TEXT_COL, LABEL_COL)