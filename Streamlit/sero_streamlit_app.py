# sero_streamlit_app.py

import streamlit as st
import pandas as pd
from pathlib import Path
import altair as alt
import numpy as np
# ---------- CONFIG GÉNÉRALE ----------
st.set_page_config(
    page_title="SERO – Usage & Crisis Patterns",
    layout="wide"
)

st.title("SERO – Usage patterns & crisis-related activity (anonymised data)")

DATA_DIR = Path("../data/Anonymised")  # adapte si tes CSV sont dans un sous-dossier


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
    Coupe la journée en 3 blocs simples (comme tu voulais):
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


# ---------- CHARGEMENT DES DONNÉES ----------

@st.cache_data
def load_data(data_dir: Path):
    # events
    events_path = data_dir / "SERO-events.csv"
    careplan_path = data_dir / "SERO-careplan.csv"
    supportcareplan_path = data_dir / "SERO-SupportCareplan.csv"
    obs_path = data_dir / "SERO-Observations.csv"
    qr_path = data_dir / "SERO-QuestionnaireResponses.csv"

    events = pd.read_csv(events_path)
    careplan = pd.read_csv(careplan_path)
    supportcareplan = pd.read_csv(supportcareplan_path)
    observations = pd.read_csv(obs_path)
    qr = pd.read_csv(qr_path)

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
    qr["is_suicidal"] = qr["answer"].apply(is_suicidal_answer)

    # Construction d'une table "interaction" globale
    # - event: catégorie/nom d'événement
    e = events.copy()
    e["source"] = "event"
    e["kind"] = e["event_category"].astype(str) + "/" + e["event_name"].astype(str)
    e_int = e[["timestamp", "date", "weekday", "hour", "day_period", "iduser", "source", "kind"]]

    # - careplan: safety plan
    cp = careplan.copy()
    cp["source"] = "careplan"
    cp["kind"] = cp["topic"]
    cp_int = cp[["timestamp", "date", "weekday", "hour", "day_period", "iduser", "source", "kind"]]

    # - support careplan
    scp = supportcareplan.copy()
    scp["source"] = "supportCareplan"
    scp["kind"] = scp["topic"]
    scp_int = scp[["timestamp", "date", "weekday", "hour", "day_period", "iduser", "source", "kind"]]

    # - observations sur le board (position x/y)
    obs = observations.copy()
    obs["source"] = "observation"
    obs["kind"] = "observation"
    obs_int = obs[["timestamp", "date", "weekday", "hour", "day_period", "iduser", "source", "kind"]]

    # - questionnaires
    qr2 = qr.copy()
    qr2["source"] = "questionnaire"
    qr2["kind"] = "questionnaire"
    qr_int = qr2[["timestamp", "date", "weekday", "hour", "day_period", "iduser", "source", "kind"]]

    all_int = pd.concat([e_int, cp_int, scp_int, obs_int, qr_int], ignore_index=True)

    # ordre standard pour les jours
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


data = load_data(DATA_DIR)
events = data["events"]
careplan = data["careplan"]
supportcareplan = data["supportcareplan"]
observations = data["observations"]
qr = data["qr"]
all_interactions = data["all_interactions"]
weekday_order = data["weekday_order"]


# ---------- SIDEBAR : FILTRES GLOBAUX ----------

with st.sidebar:
    st.header("Filters")

    # Date range global
    min_date = all_interactions["date"].min()
    max_date = all_interactions["date"].max()
    date_range = st.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(date_range, tuple):
        start_date, end_date = date_range
    else:
        # cas où streamlit renvoie une seule date
        start_date, end_date = min_date, max_date

    # Filtre sur type de source
    available_sources = sorted(all_interactions["source"].unique())
    selected_sources = st.multiselect(
        "Sources to include",
        options=available_sources,
        default=available_sources,
    )

    # Filtre sur partie de journée (Morning/Afternoon/Evening/Night)
    available_periods = ["Morning", "Afternoon", "Evening/Night"]
    selected_periods = st.multiselect(
        "Day periods",
        options=available_periods,
        default=available_periods,
    )

    # Optionnel : filtre par sujet
    all_subjects = sorted(all_interactions["iduser"].dropna().unique())
    subject_filter = st.selectbox(
        "Filter by anonymised user (optional)",
        options=["(all)"] + all_subjects,
        index=0,
    )


# Appliquer les filtres globaux
mask = (
    (all_interactions["date"] >= start_date)
    & (all_interactions["date"] <= end_date)
    & (all_interactions["source"].isin(selected_sources))
    & (all_interactions["day_period"].isin(selected_periods))
)

if subject_filter != "(all)":
    mask &= all_interactions["iduser"].eq(subject_filter)

filtered_int = all_interactions[mask]

st.markdown(
    f"**Filtered interactions:** {len(filtered_int)} "
    f"(from {start_date} to {end_date})"
)


# ---------- 1. GLOBAL – QUAND LES UTILISATEURS UTILISENT L’APP ? ----------

st.header("1. Global usage – When are users mostly using the app?")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Unique users (filtered)", filtered_int["iduser"].nunique())
with col2:
    st.metric("Total interactions", len(filtered_int))
with col3:
    st.metric("Sources included", ", ".join(selected_sources) or "none")

# a) Distribution sur 24h (heure de la journée)
st.subheader("Usage over 24 hours")

hour_counts = (
    filtered_int.groupby("hour", observed=True)
    .size()
    .rename("count")
    .reset_index()
    .set_index("hour")
)

st.bar_chart(hour_counts)

# b) Pattern par jour de semaine
st.subheader("Weekly pattern – all interactions")

weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

weekday_counts = (
    filtered_int["weekday"]
    .value_counts()
    .reindex(weekday_order)
    .fillna(0)
    .rename("count")
    .to_frame()
)

weekday_df = weekday_counts.reset_index().rename(columns={"index": "weekday"})

chart_weekday = (
    alt.Chart(weekday_df)
    .mark_bar()
    .encode(
        x=alt.X("weekday:N", sort=weekday_order, title="Weekday"),
        y=alt.Y("count:Q", title="Number of interactions")
    )
)

st.altair_chart(chart_weekday, use_container_width=True)

# c) Répartition Morning / Afternoon / Evening/Night
st.subheader("Distribution by day period (Morning / Afternoon / Evening/Night)")

period_order = ["Morning", "Afternoon", "Evening/Night"]

period_counts = (
    filtered_int["day_period"]
    .value_counts()
    .reindex(period_order)
    .fillna(0)
    .rename("count")
    .to_frame()
)

period_df = period_counts.reset_index().rename(columns={"index": "day_period"})

chart_period = (
    alt.Chart(period_df)
    .mark_bar()
    .encode(
        x=alt.X("day_period:N", sort=period_order, title="Day period"),
        y=alt.Y("count:Q", title="Number of interactions")
    )
)

st.altair_chart(chart_period, use_container_width=True)

# ---------- 2. CRISIS / SUICIDAL IDEATION – WEEKLY PATTERN ----------

st.header("2. Weekly pattern of suicidal ideation entries (Questionnaires)")

qr_filtered = qr[
    (qr["date"] >= start_date)
    & (qr["date"] <= end_date)
]

if subject_filter != "(all)":
    qr_filtered = qr_filtered[qr_filtered["iduser"].eq(subject_filter)]

qr_suicidal = qr_filtered[qr_filtered["is_suicidal"]]

st.write(
    f"- Total questionnaire responses in range: **{len(qr_filtered)}**  \n"
    f"- Responses flagged as containing suicidal ideation: **{len(qr_suicidal)}**"
)

if len(qr_suicidal) > 0:
    # ----- Weekly pattern (Monday -> Sunday) -----
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    suicidal_weekday = (
        qr_suicidal["weekday"]
        .value_counts()
        .reindex(weekday_order)
        .fillna(0)
        .rename("count")
        .to_frame()
    )

    suicidal_weekday_df = (
        suicidal_weekday
        .reindex(weekday_order)
        .reset_index()
        .rename(columns={"index": "weekday"})
    )

    st.subheader("Weekly pattern – suicidal ideation answers")

    chart_suicidal_weekday = (
        alt.Chart(suicidal_weekday_df)
        .mark_bar()
        .encode(
            x=alt.X("weekday:N", sort=weekday_order, title="Weekday"),
            y=alt.Y("count:Q", title="Number of suicidal answers")
        )
    )

    st.altair_chart(chart_suicidal_weekday, use_container_width=True)

    # ----- Day-period pattern (Morning / Afternoon / Evening/Night) -----
    period_order = ["Morning", "Afternoon", "Evening/Night"]

    suicidal_period = (
        qr_suicidal["day_period"]
        .value_counts()
        .reindex(period_order)
        .fillna(0)
        .rename("count")
        .to_frame()
    )

    suicidal_period_df = (
        suicidal_period
        .reindex(period_order)
        .reset_index()
        .rename(columns={"index": "day_period"})
    )

    st.subheader("Day-period pattern – suicidal ideation answers")

    chart_suicidal_period = (
        alt.Chart(suicidal_period_df)
        .mark_bar()
        .encode(
            x=alt.X("day_period:N", sort=period_order, title="Day period"),
            y=alt.Y("count:Q", title="Number of suicidal answers")
        )
    )

    st.altair_chart(chart_suicidal_period, use_container_width=True)

else:
    st.info("No questionnaire answer flagged as suicidal in the current filter.")


# ---------- 3. EVENTS DURING THE DAY – TYPE & DAY PERIOD ----------

st.header("3. Amount and type of events during the day")

# Filtrer events de la même manière que filtered_int
events_mask = (
    (events["date"] >= start_date)
    & (events["date"] <= end_date)
    & (events["day_period"].isin(selected_periods))
)
if subject_filter != "(all)":
    events_mask &= events["iduser"].eq(subject_filter)

events_filtered = events[events_mask].copy()

st.write(f"Number of events in range: **{len(events_filtered)}**")

# Définir des types d'événements un peu plus lisibles
def label_event(row):
    if row["event_name"] == "optionCallEmergencyContact":
        return "CallEmergencyContact"
    elif row["event_category"] == "assessment":
        return "Assessment"
    elif row["event_category"] == "treasureChest":
        return "Board / TreasureChest"
    else:
        return f"{row['event_category']}/{row['event_name']}"

events_filtered["event_type"] = events_filtered.apply(label_event, axis=1)

# Tableau : 3 périodes de la journée x type d'événement
event_counts = (
    events_filtered
    .groupby(["day_period", "event_type"], observed=True)
    .size()
    .reset_index(name="n")
)

pivot_events = (
    event_counts
    .pivot(index="day_period", columns="event_type", values="n")
    .fillna(0)
    .reindex(index=["Morning", "Afternoon", "Evening/Night"])
)

st.subheader("Events per day period and type")
st.dataframe(pivot_events)

# Altair chart instead of st.bar_chart, with correct order
period_order = ["Morning", "Afternoon", "Evening/Night"]

pivot_long = (
    pivot_events
    .reindex(index=period_order)   # enforce order of day_period
    .reset_index()
    .melt(
        id_vars="day_period",
        var_name="event_type",
        value_name="count"
    )
)

chart_events = (
    alt.Chart(pivot_long)
    .mark_bar()
    .encode(
        x=alt.X("day_period:N", sort=period_order, title="Day period"),
        y=alt.Y("count:Q", title="Number of events"),
        color=alt.Color("event_type:N", title="Event type"),
        tooltip=["day_period", "event_type", "count"]
    )
)

st.altair_chart(chart_events, use_container_width=True)
# ---------- 4. SMOOTH GLOBAL TREND SINCE BEGINNING ----------

st.header("4. Smooth global usage trend since the beginning")

# Rebuild a mask that ignores the date slider,
# but keeps other sidebar filters (sources, day periods, user)
mask_full = (
    all_interactions["source"].isin(selected_sources)
    & all_interactions["day_period"].isin(selected_periods)
)

if subject_filter != "(all)":
    mask_full &= all_interactions["iduser"].eq(subject_filter)

full_int = all_interactions[mask_full].copy()

# Daily usage count over full history
daily_counts = (
    full_int.groupby("date")
    .size()
    .rename("count")
    .reset_index()
)

# Ensure proper datetime + chronological order
daily_counts["date"] = pd.to_datetime(daily_counts["date"])
daily_counts = daily_counts.sort_values("date").set_index("date")

# Rolling mean over the previous 15 days (including current day)
daily_counts["mean_15d"] = (
    daily_counts["count"]
    .rolling(window=15, min_periods=1)
    .mean()
)

# Keep only one point every 15 days (mean of the 15 days prior)
# Start from day 15 to really represent a full 15-day window
smooth_points = daily_counts.iloc[14::15].reset_index()

# If dataset is shorter than 15 days, fallback to at least one point
if smooth_points.empty:
    smooth_points = daily_counts.tail(1).reset_index()

st.write(
    f"Showing one point every 15 days, each being the mean usage of the previous 15 days. "
    f"Total points: **{len(smooth_points)}**"
)

chart_smooth = (
    alt.Chart(smooth_points)
    .mark_line(point=True)
    .encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("mean_15d:Q", title="Mean interactions over previous 15 days"),
        tooltip=[
            alt.Tooltip("date:T", title="Date"),
            alt.Tooltip("mean_15d:Q", title="15-day mean", format=".2f"),
        ],
    )
    .properties(height=350)
)

st.altair_chart(chart_smooth, use_container_width=True)