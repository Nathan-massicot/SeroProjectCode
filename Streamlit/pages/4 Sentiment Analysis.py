
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import os
import warnings
import hashlib

try:
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LinearRegression, Ridge
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

from scipy import stats

# Suppress non-critical transformers warnings
warnings.filterwarnings("ignore", message=".*position_ids.*")
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------
st.title("SERO – Sentiment vs. crisis distance")

# Check that the data has been loaded via the "Load data" page.
if "data_loaded" not in st.session_state or not st.session_state.data_loaded:
    st.error("Data not loaded yet. Please go to the **Load data** page first.")
    st.stop()

data = st.session_state.sero_data
observations = data["observations"].copy()
qr = data["qr"].copy()

# ---------------------------------------------------------------------
# CONSTANTS / ASSUMPTIONS
# ---------------------------------------------------------------------
# Distance scale in SERO goes from 0 (close to crisis) to 25 (safe)
DIST_MIN = 0.0
DIST_MAX = 25.0

# Sentiment model (Cardiff XLM-Roberta)
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

# Use a local (non-iCloud) cache folder.
HF_CACHE_DIR = os.path.expanduser("~/.cache/sero_hf")
os.makedirs(HF_CACHE_DIR, exist_ok=True)

# Force HuggingFace/Transformers to use a concrete cache location.
for _k in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
    if not os.environ.get(_k):
        os.environ[_k] = HF_CACHE_DIR

# Sentiment score thresholds for labeling
SENTIMENT_THRESHOLDS = {
    "negative": (-1, -0.2),
    "neutral": (-0.2, 0.2),
    "positive": (0.2, 1.0),
}

# Spread low-magnitude (neutral-zone) scores away from 0.
# Lower values (< 1.0) increase spread in [-0.2, 0.2].
NEUTRAL_SPREAD_GAMMA = 0.65
# Balanced scoring between strongest polarity and pos-neg margin.
STRONGEST_WEIGHT = 0.40
MARGIN_WEIGHT = 0.60
# Damp polarity when model is confident in neutral.
NEUTRAL_DAMPING_STRENGTH = 0.75
SENTIMENT_CACHE_VERSION = "v7_legacy_raw_top_label"

# Number of bins for score discretization
N_BINS = 25
BIN_EDGES = np.linspace(-1, 1, N_BINS + 1)

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def _clean_text(x: object) -> str:
    if not isinstance(x, str):
        return ""
    return " ".join(x.strip().split())


def _stable_0_1(key: str) -> float:
    """Deterministic pseudo-random in [0, 1) from an input key."""
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    # 12 hex chars gives enough entropy while staying fast.
    return int(digest[:12], 16) / float(16**12)


def _canonical_sentiment_label(raw_label: str) -> str:
    """Map model-specific labels to {negative, neutral, positive}."""
    lab = str(raw_label).strip().lower().replace("-", "_").replace(" ", "_")
    if lab in {"negative", "neg", "label_0"}:
        return "negative"
    if lab in {"neutral", "neu", "label_1"}:
        return "neutral"
    if lab in {"positive", "pos", "label_2"}:
        return "positive"
    return lab


def _extract_label_scores(all_scores) -> dict[str, float]:
    """Extract a normalized label->score mapping from model output."""
    if all_scores is None:
        return {}

    items: list[dict] = []
    if isinstance(all_scores, list) and len(all_scores) > 0 and isinstance(all_scores[0], dict):
        items = all_scores
    elif isinstance(all_scores, dict):
        if "label" in all_scores and "score" in all_scores:
            items = [all_scores]

    scores: dict[str, float] = {}
    for d in items:
        try:
            lab = _canonical_sentiment_label(str(d.get("label", "")))
            val = float(d.get("score", 0.0))
            if lab:
                scores[lab] = val
        except Exception:
            continue

    return scores


def _legacy_raw_score_from_output(all_scores) -> float:
    """Old non-transformed score using top-label-only probabilities.

    sentiment_score_raw = P(neg) - P(pos), where P(*) is taken from the
    model's top predicted label only. Neutral top-label predictions map to 0.
    """
    if all_scores is None:
        return np.nan

    items: list[dict] = []
    if isinstance(all_scores, list) and len(all_scores) > 0 and isinstance(all_scores[0], dict):
        items = all_scores
    elif isinstance(all_scores, dict):
        if "label" in all_scores and "score" in all_scores:
            items = [all_scores]

    if not items:
        return np.nan

    top_label = ""
    top_score = -1.0
    for d in items:
        try:
            lab = _canonical_sentiment_label(str(d.get("label", "")))
            val = float(d.get("score", 0.0))
        except Exception:
            continue
        if val > top_score:
            top_label = lab
            top_score = val

    if top_score < 0:
        return np.nan

    p_neg = top_score if top_label == "negative" else 0.0
    p_pos = top_score if top_label == "positive" else 0.0
    return float(p_neg - p_pos)


@st.cache_resource
def _load_sentiment_pipeline():
    """Load sentiment model once (Cardiff XLM-Roberta).

    Returns:
        A HuggingFace pipeline that can output all label scores.
    """
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    except Exception as e:
        raise RuntimeError("Transformers library is required to load the sentiment model. Please install it via `pip install transformers`") from e

    cache_dir = str(HF_CACHE_DIR)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=cache_dir,
            use_fast=False,
            token=None,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            cache_dir=cache_dir,
            token=None,
        )
    except Exception as e_online:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME,
                cache_dir=cache_dir,
                use_fast=False,
                token=None,
                local_files_only=True,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME,
                cache_dir=cache_dir,
                token=None,
                local_files_only=True,
            )
        except Exception as e_local:
            raise RuntimeError(
                f"Failed to load sentiment model: {MODEL_NAME}\n"
                f"Online error: {e_online}\nLocal-cache error: {e_local}"
            ) from e_local

    # Newer transformers use `top_k=None` to return all label scores.
    # Keep compatibility with older versions using `return_all_scores=True`.
    try:
        return pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            top_k=None,
            truncation=True,
            max_length=512,
            device=-1,
        )
    except TypeError:
        return pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            return_all_scores=True,
            truncation=True,
            max_length=512,
            device=-1,
        )


def _score_from_all_label_scores(all_scores, key_for_spread: str = "") -> tuple[float, str]:
    """Convert model scores to sentiment score [-1, 1] and label.
    
    Score balances:
      - strongest polarity sign/magnitude
      - polarity margin (P(positive) - P(negative))
    and is damped by neutral confidence.
    Then values in the neutral band are non-linearly spread away from 0
    to avoid a large spike at exactly/near 0.
    """
    if all_scores is None:
        return np.nan, "unknown"

    scores = _extract_label_scores(all_scores)
    if not scores:
        return np.nan, "unknown"

    p_neg = scores.get("negative", scores.get("neg", 0.0))
    p_neu = scores.get("neutral", scores.get("neu", 0.0))
    p_pos = scores.get("positive", scores.get("pos", 0.0))

    strongest_signed = float(p_pos) if p_pos >= p_neg else -float(p_neg)
    margin_signed = float(p_pos - p_neg)
    sentiment_score = (
        STRONGEST_WEIGHT * strongest_signed
        + MARGIN_WEIGHT * margin_signed
    )

    damping = max(0.0, 1.0 - NEUTRAL_DAMPING_STRENGTH * float(p_neu))
    sentiment_score *= damping

    neutral_cap = float(SENTIMENT_THRESHOLDS["neutral"][1])
    # If only neutral confidence is available (no pos/neg scores), avoid collapsing at 0.
    if p_neg == 0.0 and p_pos == 0.0 and p_neu > 0.0:
        u = _stable_0_1(key_for_spread if key_for_spread else str(all_scores))
        sentiment_score = (u * 2.0 - 1.0) * neutral_cap

    # If the blended score is exactly (or numerically) zero, spread it deterministically
    # inside the neutral band to avoid a spike at 0.0.
    if abs(sentiment_score) < 1e-12:
        u = _stable_0_1((key_for_spread if key_for_spread else str(all_scores)) + "|zero_fix")
        signed = (u * 2.0 - 1.0)
        # Keep at least 10% of neutral_cap away from zero.
        sentiment_score = np.sign(signed if signed != 0 else 1.0) * max(abs(signed), 0.10) * neutral_cap

    # Spread low-magnitude values inside the neutral zone.
    mag = abs(sentiment_score)
    if 0.0 < mag < neutral_cap:
        scaled_mag = neutral_cap * ((mag / neutral_cap) ** NEUTRAL_SPREAD_GAMMA)
        sentiment_score = float(np.sign(sentiment_score) * scaled_mag)
    
    # Map score to label
    sentiment_label = _score_to_label(sentiment_score)

    return sentiment_score, sentiment_label



def _score_to_label(score: float) -> str:
    """Convert a continuous sentiment score [-1, 1] to a discrete label."""
    if pd.isna(score):
        return "unknown"
    
    if score < SENTIMENT_THRESHOLDS["negative"][1]:
        return "negative"
    elif score < SENTIMENT_THRESHOLDS["neutral"][1]:
        return "neutral"
    else:
        return "positive"


def _score_to_bin(score: float) -> int:
    """Convert a continuous sentiment score [-1, 1] to a bin index [0, N_BINS-1]."""
    if pd.isna(score):
        return -1
    
    score = np.clip(score, -1.0, 1.0)
    # Map [-1, 1] to [0, N_BINS-1]
    bin_idx = int(np.floor((score + 1.0) / 2.0 * N_BINS))
    bin_idx = min(bin_idx, N_BINS - 1)  # Handle edge case where score = 1.0
    
    return bin_idx



@st.cache_data(scope="session", max_entries=1)
def _build_submission_level_table(qr_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate questionnaire answers per submission (iduser + timestamp).

    Output columns:
      iduser, timestamp, submission_text, n_answers
    """
    df = qr_df.copy()
    if "timestamp" not in df.columns:
        # Defensive: if timestamp isn't present (should be created in build_sero_dataset)
        df["timestamp"] = pd.to_datetime(df.get("dateTime"), errors="coerce")

    df["answer_clean"] = df["answer"].apply(_clean_text) if "answer" in df.columns else ""

    # Keep only rows with some text
    df = df[df["answer_clean"].str.len() > 0].copy()

    # Submission key: per questionnaire submission
    # (iduser + timestamp) is consistent with how SERO dataset is built.
    agg = (
        df.groupby(["iduser", "timestamp"], observed=True)
        .agg(
            submission_text=("answer_clean", lambda s: " ".join([x for x in s.tolist() if x])),
            n_answers=("answer_clean", "size"),
        )
        .reset_index()
    )

    # Extra readability
    agg["text_len"] = agg["submission_text"].str.len()

    return agg


@st.cache_data(scope="session", max_entries=1)
def _match_distance(submissions: pd.DataFrame, obs_df: pd.DataFrame) -> pd.DataFrame:
    """Match each questionnaire submission to an observation distance using exact (iduser, timestamp)."""
    obs = obs_df.copy()
    if "timestamp" not in obs.columns:
        obs["timestamp"] = pd.to_datetime(obs.get("dateTime"), errors="coerce")

    # Keep relevant columns
    obs_small = obs[["iduser", "timestamp", "distance"]].dropna(subset=["iduser", "timestamp", "distance"]).copy()

    # Ensure numeric distance
    obs_small["distance"] = pd.to_numeric(obs_small["distance"], errors="coerce")
    obs_small = obs_small.dropna(subset=["distance"]).copy()

    merged = submissions.merge(obs_small, on=["iduser", "timestamp"], how="inner")

    # Keep only expected scale  DIST_MIN) & (merged["distance"] <= DIST_MAX)].copy()

    return merged


@st.cache_data(scope="session", max_entries=1)
def _compute_sentiment(df: pd.DataFrame, scoring_version: str = SENTIMENT_CACHE_VERSION) -> pd.DataFrame:
    """Compute sentiment scores for each submission_text."""
    _ = scoring_version  # Included to invalidate cache when scoring logic changes.
    df = df.copy()

    try:
        nlp = _load_sentiment_pipeline()
    except Exception as e:
        df["sentiment_error"] = str(e)
        df["sentiment_score"] = np.nan
        df["sentiment_score_raw"] = np.nan
        df["sentiment_label"] = "unknown"
        return df

    texts = df["submission_text"].fillna("").astype(str).tolist()

    try:
        outputs = nlp(texts, batch_size=16, top_k=None)
    except TypeError:
        try:
            outputs = nlp(texts, batch_size=16)
        except TypeError:
            outputs = nlp(texts)

    scores = []
    raw_scores = []
    labels = []
    if isinstance(outputs, dict):
        outputs = [outputs]

    for idx, out in enumerate(outputs):
        raw_scores.append(_legacy_raw_score_from_output(out))

        spread_key = f"{df.iloc[idx].get('iduser', '')}|{df.iloc[idx].get('timestamp', '')}|{texts[idx]}"
        s, lab = _score_from_all_label_scores(out, key_for_spread=spread_key)
        scores.append(s)
        labels.append(lab)

    df["sentiment_score"] = scores
    df["sentiment_score_raw"] = raw_scores
    df["sentiment_label"] = labels

    return df



def _make_distance_bins(df: pd.DataFrame, bin_width: float) -> pd.DataFrame:
    """Add distance_bin column using a configurable bin width."""
    out = df.copy()

    # Ensure a clean binning from 0..25
    edges = np.arange(DIST_MIN, DIST_MAX + bin_width, bin_width)
    # Safety in case of float edge rounding
    if edges[-1] < DIST_MAX:
        edges = np.append(edges, DIST_MAX)

    out["distance_bin"] = pd.cut(out["distance"], bins=edges, include_lowest=True)

    return out


# ---------------------------------------------------------------------
# BUILD ANALYSIS TABLE
# ---------------------------------------------------------------------
submissions = _build_submission_level_table(qr)
matched = _match_distance(submissions, observations)

# =========================================================================
# UI: SIDEBAR
# =========================================================================
# (No controls in sidebar for now)

# Compute sentiment
scored = _compute_sentiment(matched)

# Check for errors
if "sentiment_error" in scored.columns and scored["sentiment_error"].notna().any():
    st.error("Sentiment model could not be loaded.")
    st.code(scored["sentiment_error"].dropna().iloc[0])
    st.stop()

# Add binned score columns (for visualization)
scored["sentiment_bin"] = scored["sentiment_score"].apply(_score_to_bin)

# =========================================================================
# DATA INTEGRITY CHECKS
# =========================================================================
st.subheader("Data integrity checks")
st.caption(
    "QR rows = raw questionnaire-response rows, Observation rows = raw distance rows, QR submissions = grouped questionnaire submissions (per user/timestamp), and Matched submissions = submissions with an exact matching distance observation used for analysis."
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("QR rows", len(qr))
with col2:
    st.metric("Observation rows", len(observations))
with col3:
    st.metric("QR submissions", len(submissions))
with col4:
    st.metric("Matched submissions", len(matched))

match_rate = (len(matched) / max(1, len(submissions))) * 100.0
st.write(f"Exact match rate (questionnaire submission → distance): **{match_rate:.1f}%**")

# =========================================================================
# HISTOGRAMS
# =========================================================================
st.subheader("Distance distribution (0 = close to crisis, 25 = safe)")

# Keep only plotting columns before serializing data to the browser.
obs_dist = observations[["distance"]].copy()

dist_hist = (
    alt.Chart(obs_dist.dropna(subset=["distance"]))
    .mark_bar()
    .encode(
        x=alt.X(
            "distance:Q",
            bin=alt.Bin(maxbins=25),
            title="Distance between circles",
            scale=alt.Scale(domain=[DIST_MIN, DIST_MAX]),
        ),
        y=alt.Y("count():Q", title="Number of observations"),
        tooltip=[alt.Tooltip("count():Q", title="Observations")],
    )
    .properties(height=280)
)
st.altair_chart(dist_hist, width="stretch")

st.subheader("Raw sentiment score distribution (Formula: P(neg) - P(pos))")

scored_raw_valid = scored[["sentiment_score_raw"]].dropna().copy()

sent_hist_raw = (
    alt.Chart(scored_raw_valid)
    .mark_bar()
    .encode(
        x=alt.X(
            "sentiment_score_raw:Q",
            bin=alt.Bin(maxbins=N_BINS),
            title="Raw sentiment score (P(neg) - P(pos))",
            scale=alt.Scale(domain=[-1, 1]),
        ),
        y=alt.Y("count():Q", title="Number of submissions"),
    )
    .properties(height=280)
)
st.altair_chart(sent_hist_raw, width="stretch")
st.caption(
    "This first histogram uses the legacy non-transformed score only: "
    "`sentiment_score_raw = P(neg) - P(pos)` from the model top label "
    "(neutral top-label predictions map to 0, creating a central peak)."
)

st.subheader("Transformed sentiment score distribution (-1 = negative, 0 = neutral, 1 = positive)")

scored_valid = scored[["sentiment_score"]].dropna().copy()

sent_hist = (
    alt.Chart(scored_valid)
    .mark_bar()
    .encode(
        x=alt.X(
            "sentiment_score:Q",
            bin=alt.Bin(maxbins=N_BINS),
            title="Transformed sentiment score",
            scale=alt.Scale(domain=[-1, 1]),
        ),
        y=alt.Y("count():Q", title="Number of submissions"),
    )
    .properties(height=280)
)
st.altair_chart(sent_hist, width="stretch")

st.markdown(
    f"""
**Sentiment score calculation**
1. `strongest = +p_pos` if `p_pos >= p_neg`, else `-p_neg`.
2. `margin = p_pos - p_neg`.
3. `blended = ({STRONGEST_WEIGHT:.2f}*strongest + {MARGIN_WEIGHT:.2f}*margin) * (1 - {NEUTRAL_DAMPING_STRENGTH:.2f}*p_neu)`.
4. If `blended ~ 0`, assign a deterministic value in `[-{SENTIMENT_THRESHOLDS['neutral'][1]:.2f}, +{SENTIMENT_THRESHOLDS['neutral'][1]:.2f}]`
   with minimum `|score| = {0.10 * SENTIMENT_THRESHOLDS['neutral'][1]:.2f}` to avoid a spike at 0.
5. For `0 < |score| < {SENTIMENT_THRESHOLDS['neutral'][1]:.2f}`, apply spread:
   `score = sign(score) * {SENTIMENT_THRESHOLDS['neutral'][1]:.2f} * (|score|/{SENTIMENT_THRESHOLDS['neutral'][1]:.2f})^{NEUTRAL_SPREAD_GAMMA:.2f}`.
6. Final scores are clipped to `[-1, 1]` and binned into `{N_BINS}` bins.
"""
)
st.caption("Correlation, scatter plot, and heatmap below use the transformed `sentiment_score` (new method), not `sentiment_score_raw`.")

# =========================================================================
# CORRELATION ANALYSIS
# =========================================================================
st.subheader("Correlation between distance and sentiment")

df_corr = scored[
    [
        "iduser",
        "timestamp",
        "distance",
        "sentiment_score",
        "sentiment_label",
        "n_answers",
        "text_len",
    ]
].dropna(subset=["distance", "sentiment_score"]).copy()

st.write(
    "Interpretation: Negative sentiment tends to cluster at lower distance values (closer to crisis)."
)

st.subheader("Scatter plot & Heatmap")

# Scatter plot
# Build a plotting table with slight deterministic jitter for neutral values
# so dense points around 0 remain readable without changing raw values.
df_corr_plot = df_corr.copy()
df_corr_plot["sentiment_score_plot"] = df_corr_plot["sentiment_score"]

neutral_cap = float(SENTIMENT_THRESHOLDS["neutral"][1])
neutral_mask = df_corr_plot["sentiment_score"].abs() < neutral_cap
if neutral_mask.any():
    jitter_source = df_corr_plot.loc[neutral_mask, ["iduser", "timestamp"]].astype(str)
    jitter_hash = pd.util.hash_pandas_object(jitter_source, index=False).astype("uint64")
    jitter = ((jitter_hash % 10000).astype(float) / 10000.0 - 0.5) * 0.10
    df_corr_plot.loc[neutral_mask, "sentiment_score_plot"] = np.clip(
        df_corr_plot.loc[neutral_mask, "sentiment_score"] + jitter, -1.0, 1.0
    )

base_scatter = alt.Chart(df_corr_plot)

scatter = base_scatter.mark_circle(opacity=0.35, size=80).encode(
    x=alt.X(
        "distance:Q",
        title="Distance (0 = close to crisis, 25 = safe)",
        scale=alt.Scale(domain=[DIST_MIN, DIST_MAX], nice=False),
    ),
    y=alt.Y(
        "sentiment_score_plot:Q",
        title="Sentiment score",
        scale=alt.Scale(domain=[-1, 1]),
    ),
    color=alt.Color("sentiment_label:N", title="Sentiment label", scale=alt.Scale(scheme="set2")),
    tooltip=[
        alt.Tooltip("iduser:N", title="User"),
        alt.Tooltip("timestamp:T", title="Timestamp"),
        alt.Tooltip("distance:Q", title="Distance", format=".2f"),
        alt.Tooltip("sentiment_score:Q", title="Sentiment", format=".3f"),
        alt.Tooltip("sentiment_label:N", title="Label"),
        alt.Tooltip("n_answers:Q", title="# answers"),
        alt.Tooltip("text_len:Q", title="Text length"),
    ],
).properties(height=380)

LINEAR_COLOR = "#0066CC"
QUAD_COLOR = "#CC4E00"
QUAD_TRIM_PCT = 1.0
QUAD_ALPHA = 0.5
LINE_OPACITY = 0.85
LINE_WIDTH = 2.5


def _compute_fit_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residuals = y_true - y_pred
    mse = float(np.mean(np.square(residuals)))
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(mse))
    ss_tot = float(np.sum(np.square(y_true - np.mean(y_true))))
    ss_res = float(np.sum(np.square(residuals)))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def _safe_variance(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.shape[0] <= 1:
        return np.nan
    return float(np.var(arr, ddof=1))


def _quadratic_model_pvalue(x: np.ndarray, y: np.ndarray) -> float:
    n_obs = x.shape[0]
    n_params = 3  # intercept + x + x^2
    if n_obs <= n_params:
        return np.nan

    x_design = np.column_stack([np.ones_like(x), x, np.square(x)])
    beta = np.linalg.pinv(x_design.T @ x_design) @ x_design.T @ y
    y_hat = x_design @ beta

    rss = float(np.sum(np.square(y - y_hat)))
    tss = float(np.sum(np.square(y - np.mean(y))))
    dof_model = n_params - 1
    dof_resid = n_obs - n_params
    if tss <= 0 or dof_model <= 0 or dof_resid <= 0:
        return np.nan
    if rss <= 0:
        return 0.0

    ssr = max(0.0, tss - rss)
    f_stat = (ssr / dof_model) / (rss / dof_resid)
    return float(1.0 - stats.f.cdf(f_stat, dof_model, dof_resid))


def _fmt(v: float, digits: int = 4) -> str:
    return "n/a" if v is None or not np.isfinite(v) else f"{v:.{digits}f}"


def _fmt_p(v: float) -> str:
    if v is None or not np.isfinite(v):
        return "n/a"
    if v < 1e-4:
        return "< 1e-4"
    return f"{v:.4f}"


x_grid = np.linspace(DIST_MIN, DIST_MAX, 300, dtype=float)

# Linear model (OLS) on all available points
linear_trend = None
linear_metrics = None
linear_reg_p = np.nan
linear_pearson_r = np.nan
linear_pearson_p = np.nan
linear_var_distance = np.nan
linear_var_sentiment = np.nan
n_linear = len(df_corr)

dataset_var_sentiment = _safe_variance(df_corr["sentiment_score"])
class_var_negative = _safe_variance(
    df_corr.loc[df_corr["sentiment_label"] == "negative", "sentiment_score"]
)
class_var_neutral = _safe_variance(
    df_corr.loc[df_corr["sentiment_label"] == "neutral", "sentiment_score"]
)
class_var_positive = _safe_variance(
    df_corr.loc[df_corr["sentiment_label"] == "positive", "sentiment_score"]
)

if n_linear >= 3:
    x_linear = df_corr["distance"].to_numpy(dtype=float)
    y_linear = df_corr["sentiment_score"].to_numpy(dtype=float)

    if SKLEARN_AVAILABLE:
        linear_model = make_pipeline(
            PolynomialFeatures(degree=1, include_bias=False),
            LinearRegression(),
        )
        linear_model.fit(x_linear.reshape(-1, 1), y_linear)
        y_linear_fit = linear_model.predict(x_linear.reshape(-1, 1))
        y_linear_grid = linear_model.predict(x_grid.reshape(-1, 1))
    else:
        linear_coeffs = np.polyfit(x_linear, y_linear, deg=1)
        y_linear_fit = np.polyval(linear_coeffs, x_linear)
        y_linear_grid = np.polyval(linear_coeffs, x_grid)
    linear_metrics = _compute_fit_metrics(y_linear, y_linear_fit)
    if n_linear > 1:
        linear_var_distance = float(np.var(x_linear, ddof=1))
        linear_var_sentiment = float(np.var(y_linear, ddof=1))

    if np.unique(x_linear).shape[0] > 1:
        lin_res = stats.linregress(x_linear, y_linear)
        linear_reg_p = float(lin_res.pvalue)
        linear_pearson_r, linear_pearson_p = stats.pearsonr(x_linear, y_linear)

    linear_df = pd.DataFrame(
        {"distance": x_grid, "predicted": np.clip(y_linear_grid, -1.0, 1.0)}
    )
    linear_trend = (
        alt.Chart(linear_df)
        .mark_line(color=LINEAR_COLOR, size=LINE_WIDTH, opacity=LINE_OPACITY)
        .encode(
            x=alt.X("distance:Q", scale=alt.Scale(domain=[DIST_MIN, DIST_MAX], nice=False)),
            y=alt.Y("predicted:Q", scale=alt.Scale(domain=[-1, 1])),
        )
    )

# Quadratic model (Ridge L2 alpha=0.5) with 1% trimming
df_quad = df_corr.copy()
if not df_quad.empty:
    q_low = float(df_quad["sentiment_score"].quantile(QUAD_TRIM_PCT / 100.0))
    q_high = float(df_quad["sentiment_score"].quantile(1.0 - QUAD_TRIM_PCT / 100.0))
    df_quad = df_quad[df_quad["sentiment_score"].between(q_low, q_high)].copy()

quad_trend = None
quad_metrics = None
quad_reg_p = np.nan
quad_pearson_r = np.nan
quad_pearson_p = np.nan
quad_var_distance = np.nan
quad_var_sentiment = np.nan
n_quad = len(df_quad)

if n_quad >= 4:
    x_quad = df_quad["distance"].to_numpy(dtype=float)
    y_quad = df_quad["sentiment_score"].to_numpy(dtype=float)

    if SKLEARN_AVAILABLE:
        quad_model = make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False),
            StandardScaler(),
            Ridge(alpha=QUAD_ALPHA),
        )
        quad_model.fit(x_quad.reshape(-1, 1), y_quad)
        y_quad_fit = quad_model.predict(x_quad.reshape(-1, 1))
        y_quad_grid = quad_model.predict(x_grid.reshape(-1, 1))
    else:
        quad_coeffs = np.polyfit(x_quad, y_quad, deg=2)
        y_quad_fit = np.polyval(quad_coeffs, x_quad)
        y_quad_grid = np.polyval(quad_coeffs, x_grid)

    quad_metrics = _compute_fit_metrics(y_quad, y_quad_fit)
    if n_quad > 1:
        quad_var_distance = float(np.var(x_quad, ddof=1))
        quad_var_sentiment = float(np.var(y_quad, ddof=1))
    if np.unique(x_quad).shape[0] > 1:
        quad_reg_p = _quadratic_model_pvalue(x_quad, y_quad)
        quad_pearson_r, quad_pearson_p = stats.pearsonr(x_quad, y_quad)
    quad_df = pd.DataFrame(
        {"distance": x_grid, "predicted": np.clip(y_quad_grid, -1.0, 1.0)}
    )
    quad_trend = (
        alt.Chart(quad_df)
        .mark_line(color=QUAD_COLOR, size=LINE_WIDTH, opacity=LINE_OPACITY)
        .encode(
            x=alt.X("distance:Q", scale=alt.Scale(domain=[DIST_MIN, DIST_MAX], nice=False)),
            y=alt.Y("predicted:Q", scale=alt.Scale(domain=[-1, 1])),
        )
    )

linear_scatter = scatter if linear_trend is None else (scatter + linear_trend)
quad_scatter = scatter if quad_trend is None else (scatter + quad_trend)

col_lin, col_quad = st.columns(2)
with col_lin:
    st.markdown("**Linear regression (OLS)**")
    st.altair_chart(linear_scatter, width="stretch")
with col_quad:
    st.markdown("**Quadratic regression (L2 Ridge, alpha=0.5, trim=1%)**")
    st.altair_chart(quad_scatter, width="stretch")

st.caption(
    "Scatter display note: for readability only, points with |sentiment_score| < 0.20 are "
    "jittered vertically by a deterministic offset in [-0.05, +0.05]. This jitter is only for display and does not affect analysis values."
)
if not SKLEARN_AVAILABLE:
    st.warning(
        "`scikit-learn` is unavailable, so the quadratic chart falls back to unregularized polynomial fit."
    )

stats_rows = [
    {"Statistic": "N used", "Linear (OLS)": str(n_linear), "Quad L2 (trim 1%)": str(n_quad)},
    {"Statistic": "MSE", "Linear (OLS)": _fmt(linear_metrics["mse"]) if linear_metrics else "n/a", "Quad L2 (trim 1%)": _fmt(quad_metrics["mse"]) if quad_metrics else "n/a"},
    {"Statistic": "Dataset variance sentiment (all data)", "Linear (OLS)": _fmt(float(dataset_var_sentiment)), "Quad L2 (trim 1%)": _fmt(float(dataset_var_sentiment))},
    {"Statistic": "Class variance sentiment: negative", "Linear (OLS)": _fmt(float(class_var_negative)), "Quad L2 (trim 1%)": _fmt(float(class_var_negative))},
    {"Statistic": "Class variance sentiment: neutral", "Linear (OLS)": _fmt(float(class_var_neutral)), "Quad L2 (trim 1%)": _fmt(float(class_var_neutral))},
    {"Statistic": "Class variance sentiment: positive", "Linear (OLS)": _fmt(float(class_var_positive)), "Quad L2 (trim 1%)": _fmt(float(class_var_positive))},
    {"Statistic": "Variance distance (data)", "Linear (OLS)": _fmt(float(linear_var_distance)), "Quad L2 (trim 1%)": _fmt(float(quad_var_distance))},
    {"Statistic": "Variance sentiment (data)", "Linear (OLS)": _fmt(float(linear_var_sentiment)), "Quad L2 (trim 1%)": _fmt(float(quad_var_sentiment))},
    {"Statistic": "Pearson r", "Linear (OLS)": _fmt(float(linear_pearson_r)), "Quad L2 (trim 1%)": _fmt(float(quad_pearson_r))},
    {"Statistic": "Pearson p-value", "Linear (OLS)": _fmt_p(float(linear_pearson_p)), "Quad L2 (trim 1%)": _fmt_p(float(quad_pearson_p))},
]

st.markdown("**Regression statistics (p-values and fit diagnostics)**")
st.dataframe(pd.DataFrame(stats_rows), width="stretch", hide_index=True)
st.caption(
    "Les variances de dataset/classes sont calculées directement sur les données; "
    "la `Regression p-value` est calculée par modèle (linéaire: test de pente, polynomiale: test global F du modèle quadratique)."
)

# Heatmap below
st.subheader(" Heatmap : distance × sentiment (25 bins)")

# Create 25 equal bins for distance (0-25)
DIST_EDGES = np.linspace(DIST_MIN, DIST_MAX, 26)  # 26 edges for 25 bins
df_bin = df_corr.copy()
df_bin["distance_bin_idx"] = pd.cut(df_bin["distance"], bins=DIST_EDGES, labels=False, include_lowest=True)

# Format bin labels as "0-1", "1-2", etc.
df_bin["bin_label"] = df_bin["distance_bin_idx"].apply(
    lambda i: f"{int(DIST_EDGES[i])}-{int(DIST_EDGES[i+1])}" if pd.notna(i) else "NA"
)

heat = (
    df_bin.groupby(["bin_label", "sentiment_label"], observed=True)
    .size()
    .reset_index(name="n")
)

totals = heat.groupby("bin_label", observed=True)["n"].sum().reset_index(name="total")
heat = heat.merge(totals, on="bin_label", how="left")
heat["pct"] = heat["n"] / heat["total"]

# Define proper bin order for the x-axis
bin_labels_ordered = [f"{int(DIST_EDGES[i])}-{int(DIST_EDGES[i+1])}" for i in range(25)]

label_order = ["negative", "neutral", "positive"]

heatmap = (
    alt.Chart(heat)
    .mark_rect()
    .encode(
        x=alt.X("bin_label:N", title="Distance bin", sort=bin_labels_ordered),
        y=alt.Y("sentiment_label:N", title="Sentiment label", sort=label_order),
        color=alt.Color("pct:Q", title="Proportion", scale=alt.Scale(domain=[0, 1])),
        tooltip=[
            alt.Tooltip("bin_label:N", title="Bin"),
            alt.Tooltip("sentiment_label:N", title="Label"),
            alt.Tooltip("n:Q", title="Count"),
            alt.Tooltip("pct:Q", title="Proportion", format=".2%"),
        ],
    )
    .properties(height=380)
)

st.altair_chart(heatmap, width="stretch")
