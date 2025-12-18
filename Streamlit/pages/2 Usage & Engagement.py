import streamlit as st
import altair as alt
import pandas as pd

st.title("SERO – Usage & engagement")



# ---------------------------------------------------------------------
#  CHECK THAT DATA IS LOADED IN SESSION STATE 
# ---------------------------------------------------------------------
if "data_loaded" not in st.session_state or not st.session_state.data_loaded:
    st.error("Data not loaded yet. Please go to the **Load data** page first.")
    st.stop()

data = st.session_state.sero_data
events = data["events"]
careplan = data["careplan"]
supportcareplan = data["supportcareplan"]
observations = data["observations"]
qr = data["qr"]
all_interactions = data["all_interactions"]


# ---------------------------------------------------------------------
#  SIDEBAR FILTERS (DATE + USER)
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Filters")

    # Global date range based on events (main driver of usage)
    min_date = events["date"].min()
    max_date = events["date"].max()

    date_range = st.date_input(
        "Date range (for analyses 2 & 3)",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(date_range, tuple):
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    # Optional filter by anonymised user (based on all interactions)
    all_users = sorted(all_interactions["iduser"].dropna().unique())
    user_filter = st.selectbox(
        "Filter by anonymised user (optional)",
        options=["(all)"] + all_users,
        index=0,
    )

st.markdown(
    "This page focuses on three questions:\n"
    "1. **Retention** – Who stays in the app, and for how long?  \n"
    "2. **Sessions with assessment** – Does an assessment trigger usage of the safety plan?  \n"
    "3. **Context of high-risk moments** – What do users do in the 2 hours around suicidal answers?"
)

# ---------------------------------------------------------------------
# 1. RETENTION – WHO STAYS HOW LONG?
# ---------------------------------------------------------------------

st.header("1. User retention – how long do people stay active in SERO?")

# For retention we look at the full time span per user, based only on events.
# We apply only the optional user filter, not the date range (otherwise
# retention would be truncated by the selected window).

if user_filter == "(all)":
    events_ret = events.copy()
else:
    events_ret = events[events["iduser"].eq(user_filter)].copy()

if events_ret.empty:
    st.info("No events available for the selected user / filter to compute retention.")
else:
    # Use the normalised timestamp (UTC) created during dataset build
    if "timestamp" not in events_ret.columns:
        st.error("Column `timestamp` missing in events – cannot compute retention.")
    else:
        t_series = events_ret.groupby("iduser")["timestamp"].agg(["min", "max"])
        t_series = t_series.rename(columns={"min": "first", "max": "last"})

        # Use total seconds to get a more precise duration (including fractions of days)
        delta = t_series["last"] - t_series["first"]
        # Retention expressed in months (approximate) and weeks
        t_series["months"] = delta.dt.total_seconds() / (60 * 60 * 24 * 30)
        t_series["weeks"] = delta.dt.total_seconds() / (60 * 60 * 24 * 7)

# ---------------------------------------------------------------------
# 1.a Early retention – distribution by week (first 10 weeks)
# ---------------------------------------------------------------------
        # Buckets for retention in weeks – 1-week steps from 0–1 up to 9–10,
        # plus a catch-all for >=10 weeks
        week_bins = list(range(0, 11)) + [120]  # 0–1,1–2,...,9–10,>=10
        week_labels = [f"{i}–{i+1}" for i in range(0, 10)] + [">=10"]
        t_series["duration_bucket_weeks"] = pd.cut(
            t_series["weeks"], bins=week_bins, labels=week_labels, right=False
        )

        retention_week_counts = (
            t_series["duration_bucket_weeks"]
            .value_counts()
            .reindex(week_labels)
            .fillna(0)
            .rename("n_users")
            .reset_index()
            .rename(columns={"index": "duration_bucket_weeks"})
        )

        n_users_total = t_series.shape[0]
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Number of users (for retention)", n_users_total)
        with col_b:
            st.metric(
                "Median retention (months)",
                f"{t_series['months'].median():.1f}" if n_users_total > 0 else "–",
            )

        st.subheader("Early retention – duration in weeks (first 10 weeks)")
        chart_retention_weeks = (
            alt.Chart(retention_week_counts)
            .mark_bar()
            .encode(
                x=alt.X(
                    "duration_bucket_weeks:N",
                    title="Duration of use (weeks)",
                    sort=week_labels,
                ),
                y=alt.Y("n_users:Q", title="Number of users"),
                tooltip=["duration_bucket_weeks", "n_users"],
            )
        )
        st.altair_chart(chart_retention_weeks, width="stretch")

        

# ---------------------------------------------------------------------
# 1.b Overall retention – duration in months (full span)
# ---------------------------------------------------------------------
        # Buckets for retention in months – 1-month steps to better inspect distribution
        # e.g. [0–1[, [1–2[, ... [23–24[, [24–120[
        month_bins = list(range(0, 25)) + [120]
        month_labels = [f"{i}–{i+1}" for i in range(0, 24)] + [">=24"]
        t_series["duration_bucket"] = pd.cut(
            t_series["months"], bins=month_bins, labels=month_labels, right=False
        )

        retention_counts = (
            t_series["duration_bucket"]
            .value_counts()
            .reindex(month_labels)
            .fillna(0)
            .rename("n_users")
            .reset_index()
            .rename(columns={"index": "duration_bucket"})
        )

        st.subheader("Retention by duration bucket (full dataset span, events only)")
        chart_retention = (
            alt.Chart(retention_counts)
            .mark_bar()
            .encode(
                x=alt.X(
                    "duration_bucket:N",
                    title="Duration of use (months)",
                    sort=month_labels,
                ),
                y=alt.Y("n_users:Q", title="Number of users"),
                tooltip=["duration_bucket", "n_users"],
            )
        )
        st.altair_chart(chart_retention, width="stretch")

        st.caption(
            "Retention is computed over the **full available time span** of the dataset, "
            "based on **event activity only** (technical usage: sessions and in-app actions). "
            "The date range filter does **not** affect this plot (only the optional user filter does)."
        )

# ---------------------------------------------------------------------
# 2. SESSIONS WITH ASSESSMENT – DOES IT TRIGGER THE SAFETY PLAN?
# ---------------------------------------------------------------------

st.header("2. Sessions with assessment – does an assessment trigger the safety plan?")

# Here we do apply the date range + optional user filter on events.
events_mask = (events["date"] >= start_date) & (events["date"] <= end_date)
if user_filter != "(all)":
    events_mask &= events["iduser"].eq(user_filter)

events_sess = events[events_mask].copy()

if events_sess.empty:
    st.info("No events in the selected date range / user filter.")
else:
    # Define flags at event level for assessment / safety / emergency
    events_sess["event_category_str"] = events_sess["event_category"].astype(str)
    events_sess["event_name_str"] = events_sess["event_name"].astype(str)

    # Assessment events (based on event_category)
    events_sess["is_assessment"] = events_sess["event_category_str"].eq("assessment")

    # Emergency calls (adapt to your actual labels if needed)
    emergency_names = {"optionCallEmergencyContact", "Call Emergency", "callEmergency"}
    events_sess["is_emergency"] = events_sess["event_name_str"].isin(emergency_names)

    # Safety plan usage: treasureChest (boards) or emergency call
    events_sess["is_safety"] = (
        events_sess["event_category_str"].eq("treasureChest")
        | events_sess["is_emergency"]
    )

    # Session-level aggregation (based on idvisit)
    if "idvisit" not in events_sess.columns:
        st.error(
            "Column `idvisit` not found in events – cannot compute session-based statistics."
        )
    else:
        session_stats = (
            events_sess.groupby("idvisit")
            .agg(
                n_events=("event_name", "size"),
                has_assessment=("is_assessment", "any"),
                has_safety=("is_safety", "any"),
                has_emergency=("is_emergency", "any"),
            )
            .reset_index()
        )

        # Only keep sessions where at least one assessment took place
        sess_assessment = session_stats[session_stats["has_assessment"]]

        n_assessment_sessions = len(sess_assessment)
        n_with_safety = sess_assessment["has_safety"].sum()
        n_with_emergency = sess_assessment["has_emergency"].sum()

        if n_assessment_sessions == 0:
            st.info(
                "No sessions with assessment found in the selected date range / user filter."
            )
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Sessions with assessment", n_assessment_sessions)
            with col2:
                st.metric(
                    "…with any safety plan use",
                    f"{n_with_safety} "
                    f"({n_with_safety / n_assessment_sessions * 100:.1f} %)",
                )
            with col3:
                st.metric(
                    "…with emergency call",
                    f"{n_with_emergency} "
                    f"({n_with_emergency / n_assessment_sessions * 100:.1f} %)",
                )

            funnel_df = pd.DataFrame(
                {
                    "step": [
                        "Sessions with assessment",
                        "…with safety plan use",
                        "…with emergency call",
                    ],
                    "count": [
                        n_assessment_sessions,
                        n_with_safety,
                        n_with_emergency,
                    ],
                }
            )

            st.subheader("From assessment to safety plan: session-level funnel")
            chart_funnel = (
                alt.Chart(funnel_df)
                .mark_bar()
                .encode(
                    x=alt.X("step:N", sort=None, title=""),
                    y=alt.Y("count:Q", title="Number of sessions"),
                    tooltip=["step", "count"],
                )
            )
            st.altair_chart(chart_funnel, width="stretch")

            st.caption(
                "Each bar corresponds to sessions (idvisit) in the selected date range. "
                "A session is considered to **contain an assessment** if at least one event "
                "has `event_category = 'assessment'`. Safety plan use includes any "
                "treasure chest interaction or emergency call."
            )

# ---------------------------------------------------------------------
# 3. CONTEXT OF HIGH-RISK MOMENTS – INTERACTIONS ±2H AROUND SUICIDAL ANSWERS
# ---------------------------------------------------------------------

st.header("3. What happens in the 2 hours around suicidal answers?")

# Filter questionnaire responses by date and user
qr_filtered = qr[(qr["date"] >= start_date) & (qr["date"] <= end_date)].copy()
if user_filter != "(all)":
    qr_filtered = qr_filtered[qr_filtered["iduser"].eq(user_filter)]

if "is_suicidal" not in qr_filtered.columns:
    st.error(
        "Column `is_suicidal` not found in questionnaire responses. "
        "Please make sure the dataset is built with the current pipeline."
    )
else:
    qr_suicidal = qr_filtered[qr_filtered["is_suicidal"]]

    st.write(
        f"- Total questionnaire answers in range: **{len(qr_filtered)}**  \n"
        f"- Suicidal answers (flagged by heuristic): **{len(qr_suicidal)}**"
    )

    if qr_suicidal.empty:
        st.info("No questionnaire answers flagged as suicidal in the current filter.")
    else:
        # Prepare interaction subset for context analysis (same user filter if applied)
        int_context = all_interactions.copy()
        if user_filter != "(all)":
            int_context = int_context[int_context["iduser"].eq(user_filter)]

        if "timestamp" not in int_context.columns or "timestamp" not in qr_suicidal.columns:
            st.error(
                "Timestamp column not found in interactions or questionnaire data – "
                "cannot compute ±2h windows."
            )
        else:
            window_interactions = []

            for _, row in qr_suicidal.iterrows():
                user_id = row["iduser"]
                t = row["timestamp"]

                mask_int = (
                    (int_context["iduser"] == user_id)
                    & (int_context["timestamp"] >= t - pd.Timedelta(hours=2))
                    & (int_context["timestamp"] <= t + pd.Timedelta(hours=2))
                )
                subset = int_context.loc[mask_int, ["source", "kind"]]

                # We exclude the questionnaire interaction itself to focus on what happens
                # around the suicidal answer (plans, boards, other events, etc.)
                subset = subset[subset["source"] != "questionnaire"]

                if not subset.empty:
                    window_interactions.append(subset)

            if not window_interactions:
                st.info(
                    "No interactions (other than the questionnaire itself) found in the ±2h "
                    "windows around suicidal answers for the current filters."
                )
            else:
                window_int_df = pd.concat(window_interactions, ignore_index=True)

                # Build a readable label: source – kind
                window_int_df["label"] = (
                    window_int_df["source"].astype(str) + " – " + window_int_df["kind"].astype(str)
                )

                top_interactions = (
                    window_int_df["label"]
                    .value_counts()
                    .head(10)
                    .reset_index()
                )
                top_interactions.columns = ["label", "count"]

                st.subheader(
                    "Top 10 interactions in the ±2h windows around suicidal answers"
                )

                chart_context = (
                    alt.Chart(top_interactions)
                    .mark_bar()
                    .encode(
                        x=alt.X("count:Q", title="Occurrences in ±2h windows"),
                        y=alt.Y(
                            "label:N",
                            sort="-x",
                            title="Interaction (source – kind)",
                        ),
                        tooltip=["label", "count"],
                    )
                )

                st.altair_chart(chart_context, width="stretch")

                st.caption(
                    "For each suicidal questionnaire answer, we look at all interactions "
                    "(events, care plans, supporter plans, observations) by the same user "
                    "within a ±2 hour window, excluding the questionnaire itself. The chart "
                    "summarises which interaction types are most frequent around those "
                    "high-risk moments."
                )