# Streamlit/sero_streamlit_app.py

import streamlit as st
import pandas as pd
import io
import csv
import zipfile
import os
import sys

# S'assurer que le dossier racine du projet est dans le PYTHONPATH
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.cleaning import clean_dataframe, should_drop_date_time
from src.anonymising import anonymize_text

# ---------- CONFIG GÉNÉRALE ----------
st.set_page_config(
    page_title="SERO – Load data",
    layout="wide"
)

# ---------- CONSTANTES & HELPERS CSV / CLEANING / ANONYMISATION ----------

NA_VALUES = ["", " ", "NA", "NaN", "null", "Null", "NULL", "None", "none"]


def create_zip_from_tables(tables_dict: dict, suffix: str) -> bytes:
    """
    Crée un ZIP en mémoire contenant un CSV par table.
    tables_dict : {logical_name: DataFrame}
    suffix : ajouté au nom du fichier (ex: 'clean' ou 'anonymised')
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for logical_name, df in tables_dict.items():
            filename = f"{logical_name}_{suffix}.csv"
            csv_str = df.to_csv(index=False)
            zf.writestr(filename, csv_str)
    buf.seek(0)
    return buf.getvalue()


def detect_delimiter_from_bytes(raw_bytes: bytes) -> str:
    """Détecte un délimiteur probable à partir d'un échantillon de bytes."""
    try:
        sample = raw_bytes[:256_000].decode("utf-8", errors="ignore")
    except Exception:
        return ","

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        delim = dialect.delimiter
    except Exception:
        counts = {d: sample.count(d) for d in [",", ";", "\t", "|"]}
        delim = max(counts, key=counts.get) if any(counts.values()) else ","

    if sample.count(delim) == 0:
        return ","
    return delim


def read_and_clean_uploaded_csv(uploaded_file, logical_name: str):
    """
    Lit un fichier CSV uploadé (Streamlit UploadedFile), détecte le délimiteur,
    charge en DataFrame, applique le cleaning (clean_dataframe) en mémoire,
    et retourne (df_clean, metrics).
    """
    # Lire tous les bytes une fois
    raw_bytes = uploaded_file.read()
    delim = detect_delimiter_from_bytes(raw_bytes)

    buffer = io.BytesIO(raw_bytes)
    df_raw = pd.read_csv(
        buffer,
        sep=delim,
        dtype=str,
        keep_default_na=True,
        na_values=NA_VALUES,
        encoding="utf-8",
        on_bad_lines="skip",
        engine="python",
        quoting=csv.QUOTE_MINIMAL,
    )

    # Décider si on drope les colonnes date/time en fonction du nom logique
    stem_lower = logical_name.lower()
    drop_dt = should_drop_date_time(stem_lower)

    df_clean, metrics = clean_dataframe(df_raw, drop_dt)
    return df_clean, metrics


def anonymize_column_with_progress(
    df: pd.DataFrame,
    col: str,
    progress_bar,
    progress_state: dict,
) -> pd.DataFrame:
    """
    Anonymise une colonne texte ligne par ligne avec mise à jour de la barre
    de progression. progress_state doit contenir :
        { "rows_done": int, "total_rows": int }
    """
    if col not in df.columns:
        return df

    n = len(df)
    if n == 0:
        return df

    # Pour éviter de spammer Streamlit, on limite le nombre d'updates
    step = max(1, n // 50)  # max ~50 updates par colonne

    for i, idx in enumerate(df.index):
        text = df.at[idx, col]
        if isinstance(text, str) and text.strip():
            df.at[idx, col] = anonymize_text(text)

        progress_state["rows_done"] += 1

        # Mise à jour de la barre seulement tous les "step"
        if i % step == 0:
            frac_rows = progress_state["rows_done"] / max(1, progress_state["total_rows"])
            # Hybrid progress : 0–0.3 = lecture/cleaning, 0.3–1.0 = anonymisation
            frac = 0.3 + 0.7 * frac_rows
            progress_bar.progress(min(1.0, frac))

    return df


def process_uploaded_files_with_cleaning_and_anonymisation(uploaded_files_dict):
    """
    Orchestration complète :
      1) Lecture + cleaning en mémoire (0.0 -> 0.3)
      2) Anonymisation des colonnes sensibles (0.3 -> 1.0)
    uploaded_files_dict : mapping
        {
            "events": UploadedFile,
            "careplan": UploadedFile,
            "supportcareplan": UploadedFile,
            "observations": UploadedFile,
            "qr": UploadedFile,
        }
    Retourne un tuple (cleaned_before_anon, anonymised) où chaque élément est un dict :
        {
            "events": df,
            "careplan": df,
            "supportcareplan": df,
            "observations": df,
            "qr": df,
        }
    """
    progress_bar = st.progress(0)
    status = st.empty()

    # ----- 1) Lecture + cleaning (0.0 -> 0.3) -----
    status.markdown("🔄 Cleaning CSV files...")
    cleaned = {}

    logical_order = ["events", "careplan", "supportcareplan", "observations", "qr"]
    total_files = len(logical_order)

    for i, key in enumerate(logical_order, start=1):
        f = uploaded_files_dict[key]

        # IMPORTANT : remettre le pointeur au début au cas où
        try:
            f.seek(0)
        except Exception:
            pass

        df_clean, metrics = read_and_clean_uploaded_csv(f, logical_name=f.name)
        cleaned[key] = df_clean

        frac = 0.3 * (i / total_files)
        progress_bar.progress(frac)
        status.markdown(f"✅ Cleaned **{f.name}** ({i}/{total_files})")

    # Garder une copie des données nettoyées AVANT anonymisation
    cleaned_before_anon = {k: v.copy(deep=True) for k, v in cleaned.items()}

    # ----- 2) Anonymisation (0.3 -> 1.0) -----
    status.markdown("🛡️ Anonymising sensitive text (PERSON entities)...")

    careplan_df = cleaned["careplan"]
    support_df = cleaned["supportcareplan"]
    qr_df = cleaned["qr"]

    total_rows = len(careplan_df) + len(support_df) + len(qr_df)
    if total_rows <= 0:
        # Pas de texte à anonymiser -> on passe directement à 100%
        progress_bar.progress(1.0)
        status.markdown("ℹ️ No rows to anonymise. Building SERO dataset...")
        cleaned["careplan"] = careplan_df
        cleaned["supportcareplan"] = support_df
        cleaned["qr"] = qr_df
    else:
        progress_state = {"rows_done": 0, "total_rows": total_rows}

        # careplan : description
        if "description" in careplan_df.columns:
            careplan_df = anonymize_column_with_progress(
                careplan_df,
                "description",
                progress_bar,
                progress_state,
            )

        # supportcareplan : description
        if "description" in support_df.columns:
            support_df = anonymize_column_with_progress(
                support_df,
                "description",
                progress_bar,
                progress_state,
            )

        # questionnaireResponses : answer
        if "answer" in qr_df.columns:
            qr_df = anonymize_column_with_progress(
                qr_df,
                "answer",
                progress_bar,
                progress_state,
            )

        cleaned["careplan"] = careplan_df
        cleaned["supportcareplan"] = support_df
        cleaned["qr"] = qr_df

        # Fin de la phase d'anonymisation -> barre à 100%
        progress_bar.progress(1.0)
        status.markdown("✅ Anonymisation complete. Building SERO dataset...")

    anonymised = {
        "events": cleaned["events"],
        "careplan": cleaned["careplan"],
        "supportcareplan": cleaned["supportcareplan"],
        "observations": cleaned["observations"],
        "qr": cleaned["qr"],
    }
    return cleaned_before_anon, anonymised


# ---------- HELPERS TEMPS / PÉRIODES ----------

def to_utc(series):
    """Convertit une série datetime en timezone UTC (naïve -> UTC)."""
    dt = pd.to_datetime(series, errors="coerce")
    if getattr(dt.dt, "tz", None) is None:
        dt = dt.dt.tz_localize("UTC")
    else:
        dt = dt.dt.tz_convert("UTC")
    return dt


def day_period_from_hour(h: int) -> str:
    """
    Coupe la journée en 3 blocs simples:
    - Morning : 05h–12h
    - Afternoon : 12h–18h
    - Evening/Night : 18h–05h
    """
    if 5 <= h < 12:
        return "Morning"
    elif 12 <= h < 18:
        return "Afternoon"
    else:
        return "Evening/Night"


def add_time_features(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Ajoute timestamp UTC, date, weekday, hour, day_period à partir d'une colonne datetime."""
    df = df.copy()
    df["timestamp"] = to_utc(df[col])
    df["date"] = df["timestamp"].dt.date
    df["weekday"] = df["timestamp"].dt.day_name()
    df["hour"] = df["timestamp"].dt.hour
    df["day_period"] = df["hour"].apply(day_period_from_hour)
    return df


def is_suicidal_answer(text) -> bool:
    """Heuristique simple: flag si la réponse contient 'suizid' ou 'suicide' (DE/EN)."""
    if not isinstance(text, str):
        return False
    low = text.lower()
    return ("suizid" in low) or ("suicide" in low)


# ---------- CONSTRUCTION DU DATASET SERO À PARTIR DES DF ----------

def build_sero_dataset(
    events: pd.DataFrame,
    careplan: pd.DataFrame,
    supportcareplan: pd.DataFrame,
    observations: pd.DataFrame,
    qr: pd.DataFrame,
):
    # Ajout features temporelles
    events = add_time_features(events, "event_time")
    careplan = add_time_features(careplan, "dateTime")
    supportcareplan = add_time_features(supportcareplan, "dateTime")
    observations = add_time_features(observations, "dateTime")
    qr = add_time_features(qr, "dateTime")

    # Harmoniser les identifiants "utilisateur"
    careplan = careplan.rename(columns={"subject": "iduser"})
    supportcareplan = supportcareplan.rename(columns={"subject": "iduser"})
    observations = observations.rename(columns={"subject": "iduser"})
    qr = qr.rename(columns={"subject": "iduser"})

    # Flag des réponses avec idées suicidaires
    if "answer" in qr.columns:
        qr["is_suicidal"] = qr["answer"].apply(is_suicidal_answer)
    else:
        qr["is_suicidal"] = False

    # Construction d'une table "interaction" globale
    e = events.copy()
    e["source"] = "event"
    e["kind"] = e["event_category"].astype(str) + "/" + e["event_name"].astype(str)
    e_int = e[["timestamp", "date", "weekday", "hour", "day_period", "iduser", "source", "kind"]]

    cp = careplan.copy()
    cp["source"] = "careplan"
    cp["kind"] = cp["topic"]
    cp_int = cp[["timestamp", "date", "weekday", "hour", "day_period", "iduser", "source", "kind"]]

    scp = supportcareplan.copy()
    scp["source"] = "supportCareplan"
    scp["kind"] = scp["topic"]
    scp_int = scp[["timestamp", "date", "weekday", "hour", "day_period", "iduser", "source", "kind"]]

    obs = observations.copy()
    obs["source"] = "observation"
    obs["kind"] = "observation"
    obs_int = obs[["timestamp", "date", "weekday", "hour", "day_period", "iduser", "source", "kind"]]

    qr2 = qr.copy()
    qr2["source"] = "questionnaire"
    qr2["kind"] = "questionnaire"
    qr_int = qr2[["timestamp", "date", "weekday", "hour", "day_period", "iduser", "source", "kind"]]

    all_int = pd.concat([e_int, cp_int, scp_int, obs_int, qr_int], ignore_index=True)

    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    return {
        "events": events,
        "careplan": careplan,
        "supportcareplan": supportcareplan,
        "observations": observations,
        "qr": qr,
        "all_interactions": all_int,
        "weekday_order": weekday_order,
    }


# ---------- SESSION STATE ----------

if "sero_data" not in st.session_state:
    st.session_state.sero_data = None

if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False


# ---------- UI : PAGE LOAD DATA (MULTI-UPLOAD) ----------

st.title("SERO – Load anonymised data")

st.markdown(
    "Please select all **5 CSV files** <u>at once</u>:",
    unsafe_allow_html=True,
)

# 👉 SIDEBAR : UNIQUEMENT SI LES DONNÉES SONT CHARGÉES
if st.session_state.data_loaded and st.session_state.sero_data is not None:
    with st.sidebar:
        st.success("Data loaded ✅")
        if st.button("Go to Analyse overview", key="go_analyse_sidebar"):
            st.switch_page("pages/Analyse_Overview.py")

    data = st.session_state.sero_data
    st.success("Data already loaded ✅")
    st.write(f"Total interactions in dataset: **{len(data['all_interactions'])}**")
    st.write("You can open the **Analyse** page from the sidebar button or navigation.")

st.subheader("Upload anonymised CSV files")

expected_files = {
    "SERO-events.csv": "events",
    "SERO-careplan.csv": "careplan",
    "SERO-SupportCareplan.csv": "supportcareplan",
    "SERO-Observations.csv": "observations",
    "SERO-QuestionnaireResponses.csv": "qr",
}

uploaded_files = st.file_uploader(
    "Select the 5 CSV files",
    type="csv",
    accept_multiple_files=True,
)

if uploaded_files:
    st.write("Uploaded files:")
    for f in uploaded_files:
        st.write(f"- `{f.name}`")


def validate_files(uploaded_files):
    """Vérifie qu'on a exactement les 5 fichiers attendus."""
    if not uploaded_files:
        return False, "No file uploaded.", None

    by_name = {f.name: f for f in uploaded_files}

    missing = [fn for fn in expected_files.keys() if fn not in by_name]
    extra = [fn for fn in by_name.keys() if fn not in expected_files.keys()]

    if missing:
        return False, f"Missing required files: {', '.join(missing)}", None
    if extra:
        return False, f"Unexpected files uploaded: {', '.join(extra)}", None

    return True, "", by_name


if st.button("Process uploaded files"):
    ok, msg, by_name = validate_files(uploaded_files)

    if not ok:
        st.error(msg)
    else:
        try:
            # Construire un dict pour l'orchestrateur (pipeline temps réel)
            uploaded_dict = {
                "events": by_name["SERO-events.csv"],
                "careplan": by_name["SERO-careplan.csv"],
                "supportcareplan": by_name["SERO-SupportCareplan.csv"],
                "observations": by_name["SERO-Observations.csv"],
                "qr": by_name["SERO-QuestionnaireResponses.csv"],
            }

            cleaned_tables, anonymised_tables = process_uploaded_files_with_cleaning_and_anonymisation(uploaded_dict)

            events_df = anonymised_tables["events"]
            careplan_df = anonymised_tables["careplan"]
            supportcareplan_df = anonymised_tables["supportcareplan"]
            observations_df = anonymised_tables["observations"]
            qr_df = anonymised_tables["qr"]

            # Construction du dataset SERO (inchangé, mais avec données nettoyées + anonymisées)
            data = build_sero_dataset(
                events_df,
                careplan_df,
                supportcareplan_df,
                observations_df,
                qr_df,
            )

            st.session_state.sero_data = data
            st.session_state.data_loaded = True
            st.session_state.cleaned_tables = cleaned_tables
            st.session_state.anonymised_tables = anonymised_tables

            # Création des ZIP téléchargeables
            clean_zip = create_zip_from_tables(cleaned_tables, "clean")
            anon_zip = create_zip_from_tables(anonymised_tables, "anonymised")

            st.success("Data cleaned, anonymised and processed successfully ✅")
            st.download_button(
                "⬇️ Download cleaned CSVs (before anonymisation)",
                data=clean_zip,
                file_name="sero_cleaned.zip",
                mime="application/zip",
            )
            st.download_button(
                "⬇️ Download anonymised CSVs",
                data=anon_zip,
                file_name="sero_anonymised.zip",
                mime="application/zip",
            )
            st.info("You can now go to the Analyse overview page using the sidebar or the button below.")
            if st.button("Go to Analyse overview", key="go_analyse_after_load"):
                st.switch_page("pages/Analyse_Overview.py")

        except Exception as e:
            st.error(f"Error while cleaning/anonymising files: {e}")