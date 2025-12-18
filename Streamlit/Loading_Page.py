# Streamlit/sero_streamlit_app.py

import streamlit as st
import pandas as pd
import io
import csv
import zipfile
import os
import sys

# Ensure the project root folder is in PYTHONPATH so imports from src/ work.
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.cleaning import clean_dataframe, should_drop_date_time
from src.anonymising import anonymize_text

 # ---------------------------------------------------------------------
# GENERAL CONFIGURATION
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="SERO – Load data",
    layout="wide"
)

 # ---------------------------------------------------------------------
# CONSTANTS & HELPERS FOR CSV / CLEANING / ANONYMISATION
# ---------------------------------------------------------------------

NA_VALUES = ["", " ", "NA", "NaN", "null", "Null", "NULL", "None", "none"]


def create_zip_from_tables(tables_dict: dict, suffix: str) -> bytes:
    """
    Create an in-memory ZIP archive containing one CSV per table.
    tables_dict: mapping {logical_name: DataFrame}
    suffix: appended to each file name (e.g. 'clean' or 'anonymised')
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
    """Detect a probable delimiter from a small byte sample of the file."""
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
    Read an uploaded CSV file (Streamlit UploadedFile), detect the delimiter,
    load it as a DataFrame, apply cleaning (clean_dataframe) in memory,
    and return (df_clean, metrics).
    """
    # Read all bytes once
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

    # Decide whether to drop date/time columns based on the logical file name
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
    Anonymise a text column row by row while updating the progress bar.
    progress_state must contain:
        { "rows_done": int, "total_rows": int }
    """
    if col not in df.columns:
        return df

    n = len(df)
    if n == 0:
        return df

    # To avoid spamming Streamlit, we limit the number of UI updates
    step = max(1, n // 50)  # max ~50 updates per column

    for i, idx in enumerate(df.index):
        text = df.at[idx, col]
        if isinstance(text, str) and text.strip():
            df.at[idx, col] = anonymize_text(text)

        progress_state["rows_done"] += 1

        # Update the progress bar only every "step" rows
        if i % step == 0:
            frac_rows = progress_state["rows_done"] / max(1, progress_state["total_rows"])
            # Hybrid progress: 0–0.3 = reading/cleaning, 0.3–1.0 = anonymisation
            frac = 0.3 + 0.7 * frac_rows
            progress_bar.progress(min(1.0, frac))

    return df


def process_uploaded_files_with_cleaning_and_anonymisation(uploaded_files_dict):
    """
    Full pipeline for uploaded CSV files:
      1) In-memory reading + cleaning (0.0 -> 0.3 of the progress bar)
      2) Anonymisation of sensitive text columns (0.3 -> 1.0)
    uploaded_files_dict: mapping
        {
            "events": UploadedFile,
            "careplan": UploadedFile,
            "supportcareplan": UploadedFile,
            "observations": UploadedFile,
            "qr": UploadedFile,
        }
    Returns a tuple (cleaned_before_anon, anonymised) where each element is a dict:
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

    # ----- 1) Reading + cleaning (0.0 -> 0.3) -----
    status.markdown("🔄 Cleaning CSV files...")
    cleaned = {}

    logical_order = ["events", "careplan", "supportcareplan", "observations", "qr"]
    total_files = len(logical_order)

    for i, key in enumerate(logical_order, start=1):
        f = uploaded_files_dict[key]

        # IMPORTANT: reset the file pointer to the beginning, just in case
        try:
            f.seek(0)
        except Exception:
            pass

        df_clean, metrics = read_and_clean_uploaded_csv(f, logical_name=f.name)
        cleaned[key] = df_clean

        frac = 0.3 * (i / total_files)
        progress_bar.progress(frac)
        status.markdown(f"✅ Cleaned **{f.name}** ({i}/{total_files})")

    # Keep a copy of the cleaned data BEFORE anonymisation
    cleaned_before_anon = {k: v.copy(deep=True) for k, v in cleaned.items()}

    # ----- 2) Anonymisation (0.3 -> 1.0) -----
    status.markdown("🛡️ Anonymising sensitive text (PERSON entities)...")

    careplan_df = cleaned["careplan"]
    support_df = cleaned["supportcareplan"]
    qr_df = cleaned["qr"]

    total_rows = len(careplan_df) + len(support_df) + len(qr_df)
    if total_rows <= 0:
        # No text to anonymise -> go directly to 100%
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

        # End of anonymisation phase -> set progress bar to 100%
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


# ---------------------------------------------------------------------
# TIME / PERIOD HELPERS
# ---------------------------------------------------------------------

def to_utc(series):
    """Convert a datetime-like Series to UTC timezone (naive -> UTC)."""
    dt = pd.to_datetime(series, errors="coerce")
    if getattr(dt.dt, "tz", None) is None:
        dt = dt.dt.tz_localize("UTC")
    else:
        dt = dt.dt.tz_convert("UTC")
    return dt


def day_period_from_hour(h: int) -> str:
    """
    Split the day into 3 simple time blocks:
    - Morning: 05:00–12:00
    - Afternoon: 12:00–18:00
    - Evening/Night: 18:00–05:00
    """
    if 5 <= h < 12:
        return "Morning"
    elif 12 <= h < 18:
        return "Afternoon"
    else:
        return "Evening/Night"


def add_time_features(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Add UTC timestamp, date, weekday, hour, and day_period based on a datetime column."""
    df = df.copy()
    df["timestamp"] = to_utc(df[col])
    df["date"] = df["timestamp"].dt.date
    df["weekday"] = df["timestamp"].dt.day_name()
    df["hour"] = df["timestamp"].dt.hour
    df["day_period"] = df["hour"].apply(day_period_from_hour)
    return df


def is_suicidal_answer(text) -> bool:
    """Simple heuristic: flag answer if it contains 'suizid' or 'suicide' (DE/EN)."""
    if not isinstance(text, str):
        return False
    low = text.lower()
    return ("suizid" in low) or ("suicide" in low)


# ---------------------------------------------------------------------
# BUILD SERO DATASET FROM DATAFRAMES
# ---------------------------------------------------------------------

def build_sero_dataset(
    events: pd.DataFrame,
    careplan: pd.DataFrame,
    supportcareplan: pd.DataFrame,
    observations: pd.DataFrame,
    qr: pd.DataFrame,
):
    # Add time-based features to each table
    events = add_time_features(events, "event_time")
    careplan = add_time_features(careplan, "dateTime")
    supportcareplan = add_time_features(supportcareplan, "dateTime")
    observations = add_time_features(observations, "dateTime")
    qr = add_time_features(qr, "dateTime")

    # Harmonise user identifiers across tables
    careplan = careplan.rename(columns={"subject": "iduser"})
    supportcareplan = supportcareplan.rename(columns={"subject": "iduser"})
    observations = observations.rename(columns={"subject": "iduser"})
    qr = qr.rename(columns={"subject": "iduser"})

    # Flag questionnaire answers that contain suicidal ideation
    if "answer" in qr.columns:
        qr["is_suicidal"] = qr["answer"].apply(is_suicidal_answer)
    else:
        qr["is_suicidal"] = False

    # Build a global "interaction" table combining all sources
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


# ---------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------
# Store the processed SERO dataset and a flag indicating whether data is loaded.

if "sero_data" not in st.session_state:
    st.session_state.sero_data = None

if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False


# ---------------------------------------------------------------------
# UI: LOAD DATA PAGE (MULTI-UPLOAD)
# ---------------------------------------------------------------------

st.title("SERO – Load anonymised data")

st.markdown(
    "Please select all **5 CSV files** <u>at once</u>:",
    unsafe_allow_html=True,
)

# 👉 SIDEBAR: ONLY SHOWN WHEN DATA IS ALREADY LOADED
if st.session_state.data_loaded and st.session_state.sero_data is not None:
    with st.sidebar:
        st.success("Data loaded ✅")
        if st.button("Go to Analyse overview", key="go_analyse_sidebar"):
            st.switch_page("pages/1 Usage Overview.py")

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

# If ZIP files were already created in a previous run, always show download buttons
# so the user can download both clean and anonymised CSVs even after a rerun.
if "clean_zip" in st.session_state and "anon_zip" in st.session_state:
    st.subheader("Download processed CSV files")
    st.download_button(
        "⬇️ Download cleaned CSVs (before anonymisation)",
        data=st.session_state.clean_zip,
        file_name="sero_cleaned.zip",
        mime="application/zip",
    )
    st.download_button(
        "⬇️ Download anonymised CSVs",
        data=st.session_state.anon_zip,
        file_name="sero_anonymised.zip",
        mime="application/zip",
    )

if uploaded_files:
    st.write("Uploaded files:")
    for f in uploaded_files:
        st.write(f"- `{f.name}`")


def validate_files(uploaded_files):
    """Check that we have exactly the 5 expected CSV files."""
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
            # Build a dict for the orchestrator (real-time pipeline)
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

            # Build the SERO dataset (same structure, but with cleaned + anonymised data)
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

            # Create ZIP archives for download (cleaned and anonymised tables)
            clean_zip = create_zip_from_tables(cleaned_tables, "clean")
            anon_zip = create_zip_from_tables(anonymised_tables, "anonymised")

            # Store ZIPs in session_state so download buttons remain available
            # even after Streamlit reruns (e.g. after one download).
            st.session_state.clean_zip = clean_zip
            st.session_state.anon_zip = anon_zip

            st.success("Data cleaned, anonymised and processed successfully ✅")
            st.info("You can now download the processed CSVs and go to the Analyse overview page using the sidebar or the button below.")
            if st.button("Go to Analyse overview", key="go_analyse_after_load"):
                st.switch_page("pages/Page1_UsageOverview.py")

        except Exception as e:
            st.error(f"Error while cleaning/anonymising files: {e}")