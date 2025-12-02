# Streamlit/sero_streamlit_app.py

import streamlit as st
import pandas as pd

# ---------- CONFIG GÉNÉRALE ----------
st.set_page_config(
    page_title="SERO – Load data",
    layout="wide"
)


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

st.write(
    "Merci de sélectionner en une fois les **5 fichiers CSV anonymisés** :  \n"
    "- `SERO-events.csv`\n"
    "- `SERO-careplan.csv`\n"
    "- `SERO-SupportCareplan.csv`\n"
    "- `SERO-Observations.csv`\n"
    "- `SERO-QuestionnaireResponses.csv`"
)

if st.session_state.data_loaded and st.session_state.sero_data is not None:
    data = st.session_state.sero_data
    st.success("Data already loaded ✅")
    st.write(f"Total interactions in dataset: **{len(data['all_interactions'])}**")
    st.write("You can open the **Analyse** page from the sidebar.")

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

# Petit résumé des fichiers uploadés
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
            with st.spinner("Reading and processing uploaded CSV files..."):
                events_df = pd.read_csv(by_name["SERO-events.csv"])
                careplan_df = pd.read_csv(by_name["SERO-careplan.csv"])
                supportcareplan_df = pd.read_csv(by_name["SERO-SupportCareplan.csv"])
                observations_df = pd.read_csv(by_name["SERO-Observations.csv"])
                qr_df = pd.read_csv(by_name["SERO-QuestionnaireResponses.csv"])

                data = build_sero_dataset(
                    events_df,
                    careplan_df,
                    supportcareplan_df,
                    observations_df,
                    qr_df,
                )

            st.session_state.sero_data = data
            st.session_state.data_loaded = True

            # 🔁 Redirection directe vers la page Analyse
            # 👉 ADAPTE le chemin ci-dessous selon le nom réel de ta page d'analyse
            st.switch_page("pages/Analyse_Overview.py")

        except Exception as e:
            st.error(f"Error while processing files: {e}")