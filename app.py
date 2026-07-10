"""
Probability of Default (PD) Prediction App
--------------------------------------------
Streamlit app that loads the model artifacts produced by `model_training.ipynb`
(`scaler.pkl`, `onehot_encoder.pkl`, `xgb_model.pkl`) and scores a CSV of loan
applications (same schema as `pd_loan_data_test.csv`) with the trained XGBoost
PD model.

Run with:
    streamlit run app.py

Expected files in the same folder as this script (produced by the notebook):
    - scaler.pkl
    - onehot_encoder.pkl
    - xgb_model.pkl
"""

import os
import io

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="PD Model — Probability of Default Scoring",
    page_icon="📊",
    layout="wide",
)

ARTIFACT_DIR = os.path.dirname(os.path.abspath(__file__))
SCALER_PATH = os.path.join(ARTIFACT_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(ARTIFACT_DIR, "onehot_encoder.pkl")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "xgb_model.pkl")

DROP_COLS = ["annual_inc", "acc_now_delinq", "loan_id"]
ID_COL = "loan_id"


# ----------------------------------------------------------------------------
# Artifact loading
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load the fitted scaler, one-hot encoder, and XGBoost model saved by
    the training notebook. Returns None for any artifact that is missing."""
    scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
    encoder = joblib.load(ENCODER_PATH) if os.path.exists(ENCODER_PATH) else None
    model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    return scaler, encoder, model


def missing_artifacts(scaler, encoder, model):
    missing = []
    if scaler is None:
        missing.append("scaler.pkl")
    if encoder is None:
        missing.append("onehot_encoder.pkl")
    if model is None:
        missing.append("xgb_model.pkl")
    return missing


# ----------------------------------------------------------------------------
# Feature engineering / preprocessing
# ----------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Same steps as the training notebook's feature-engineering cell:
    log-transform annual_inc, then drop annual_inc / acc_now_delinq / loan_id."""
    out = df.copy()
    if "annual_inc" not in out.columns:
        raise ValueError("Uploaded file is missing the required 'annual_inc' column.")
    out["log_annual_inc"] = np.log1p(out["annual_inc"])
    out = out.drop(columns=[c for c in DROP_COLS if c in out.columns])
    return out


def preprocess_for_model(df_engineered: pd.DataFrame, scaler, encoder, model) -> pd.DataFrame:
    """Scale numeric columns and one-hot encode categorical columns using the
    already-fitted training artifacts (transform only, never re-fit), then
    align the resulting columns to what the model expects."""
    num_cols = list(getattr(scaler, "feature_names_in_", []))
    cat_cols = list(getattr(encoder, "feature_names_in_", []))

    missing_num = [c for c in num_cols if c not in df_engineered.columns]
    missing_cat = [c for c in cat_cols if c not in df_engineered.columns]
    if missing_num or missing_cat:
        raise ValueError(
            "Uploaded file is missing columns the model expects: "
            f"{missing_num + missing_cat}"
        )

    processed = df_engineered.copy()
    processed[num_cols] = scaler.transform(processed[num_cols])

    encoded = encoder.transform(processed[cat_cols])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(cat_cols),
        index=processed.index,
    )

    processed = pd.concat(
        [processed.drop(columns=cat_cols), encoded_df],
        axis=1,
    )

    # Align column order/set to exactly what the model was trained on
    expected_cols = list(getattr(model, "feature_names_in_", processed.columns))
    for col in expected_cols:
        if col not in processed.columns:
            processed[col] = 0.0
    processed = processed[expected_cols]

    return processed


def score_dataframe(raw_df: pd.DataFrame, scaler, encoder, model) -> pd.DataFrame:
    engineered = engineer_features(raw_df)
    X = preprocess_for_model(engineered, scaler, encoder, model)

    pred_label = model.predict(X)
    pred_proba_good = model.predict_proba(X)[:, 1]
    pred_proba_default = 1 - pred_proba_good

    results = raw_df.copy()
    results["predicted_good_bad"] = pred_label
    results["predicted_class"] = np.where(pred_label == 1, "Good (Non-Default)", "Bad (Default)")
    results["probability_of_default"] = pred_proba_default.round(4)
    results["probability_of_good"] = pred_proba_good.round(4)
    return results


# ----------------------------------------------------------------------------
# Optional: Gen AI portfolio commentary (Anthropic Claude)
# ----------------------------------------------------------------------------
def generate_ai_commentary(api_key: str, summary_stats: dict) -> str:
    """Ask Claude to write a short, plain-English risk commentary on the
    scored portfolio. Requires the `anthropic` package and a valid API key."""
    try:
        import anthropic
    except ImportError:
        return (
            "The `anthropic` package is not installed. Run `pip install anthropic` "
            "to enable AI-generated commentary."
        )

    client = anthropic.Anthropic(api_key=api_key)

    prompt = (
        "You are a credit risk analyst. Write a concise (5-7 sentence) portfolio "
        "risk commentary for a lending team based on these scored-loan statistics. "
        "Be factual, avoid hype, and call out any risk concentration worth watching.\n\n"
        f"Statistics: {summary_stats}"
    )

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in message.content if block.type == "text")


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title("📊 Probability of Default (PD) Scoring App")
st.caption(
    "Upload a loan-application CSV (same schema as `pd_loan_data_test.csv`) to score it "
    "with the trained XGBoost model."
)

scaler, encoder, model = load_artifacts()
missing = missing_artifacts(scaler, encoder, model)

if missing:
    st.error(
        "Missing model artifact(s): **" + ", ".join(missing) + "**.\n\n"
        "Run `model_development.ipynb` end-to-end first (it saves `scaler.pkl`, "
        "`onehot_encoder.pkl`, and `xgb_model.pkl`), then place those files in the "
        "same folder as `app.py` before restarting this app."
    )
    st.stop()

with st.sidebar:
    st.header("⚙️ Options")
    uploaded_file = st.file_uploader("Upload loan applications CSV", type=["csv"])
    default_threshold = st.slider(
        "Default probability flag threshold",
        min_value=0.0, max_value=1.0, value=0.50, step=0.01,
        help="Loans with a predicted probability of default at or above this "
             "value are flagged as high-risk in the results table.",
    )
    st.divider()
    st.subheader("🤖 Optional: Gen AI Commentary")
    st.caption("Uses Claude to write a short risk summary of the scored portfolio.")
    api_key = st.text_input("Anthropic API key", type="password")
    generate_commentary = st.checkbox("Generate AI commentary after scoring", value=False)

if uploaded_file is None:
    st.info("👈 Upload a CSV file in the sidebar to get started.")
    st.stop()

try:
    raw_df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Could not read the uploaded CSV: {e}")
    st.stop()

st.subheader("Preview of uploaded data")
st.dataframe(raw_df.head(10), use_container_width=True)

with st.spinner("Scoring applications..."):
    try:
        results = score_dataframe(raw_df, scaler, encoder, model)
    except Exception as e:
        st.error(f"Scoring failed: {e}")
        st.stop()

results["flagged_high_risk"] = results["probability_of_default"] >= default_threshold

# ----------------------------------------------------------------------------
# Summary metrics
# ----------------------------------------------------------------------------
st.subheader("Portfolio summary")

total_loans = len(results)
n_default = int((results["predicted_class"] == "Bad (Default)").sum())
n_good = total_loans - n_default
avg_pd = results["probability_of_default"].mean()
n_flagged = int(results["flagged_high_risk"].sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total loans scored", f"{total_loans:,}")
c2.metric("Predicted defaults", f"{n_default:,}", f"{n_default/total_loans:.1%}")
c3.metric("Average PD", f"{avg_pd:.1%}")
c4.metric(f"Flagged ≥ {default_threshold:.0%} PD", f"{n_flagged:,}")

col_a, col_b = st.columns(2)
with col_a:
    fig1 = px.histogram(
        results, x="probability_of_default", nbins=30,
        title="Distribution of predicted default probability",
        labels={"probability_of_default": "Probability of Default"},
    )
    st.plotly_chart(fig1, use_container_width=True)

with col_b:
    fig2 = px.pie(
        results, names="predicted_class",
        title="Predicted class split",
        color="predicted_class",
        color_discrete_map={"Good (Non-Default)": "#2ca02c", "Bad (Default)": "#d62728"},
    )
    st.plotly_chart(fig2, use_container_width=True)

if "grade" in results.columns:
    st.subheader("Average predicted PD by loan grade")
    grade_pd = (
        results.groupby("grade")["probability_of_default"]
        .mean()
        .sort_index()
        .reset_index()
    )
    fig3 = px.bar(
        grade_pd, x="grade", y="probability_of_default",
        labels={"probability_of_default": "Average Probability of Default", "grade": "Grade"},
        title="Average PD by Grade",
    )
    st.plotly_chart(fig3, use_container_width=True)

# ----------------------------------------------------------------------------
# Results table + download
# ----------------------------------------------------------------------------
st.subheader("Scored applications")

show_flagged_only = st.checkbox("Show only high-risk flagged loans", value=False)
display_df = results[results["flagged_high_risk"]] if show_flagged_only else results
st.dataframe(display_df, use_container_width=True)

csv_buffer = io.StringIO()
results.to_csv(csv_buffer, index=False)
st.download_button(
    "⬇️ Download scored results as CSV",
    data=csv_buffer.getvalue(),
    file_name="pd_scored_predictions.csv",
    mime="text/csv",
)

# ----------------------------------------------------------------------------
# Optional Gen AI commentary
# ----------------------------------------------------------------------------
if generate_commentary:
    st.subheader("🤖 AI-generated portfolio commentary")
    if not api_key:
        st.warning("Enter an Anthropic API key in the sidebar to generate commentary.")
    else:
        summary_stats = {
            "total_loans": total_loans,
            "predicted_defaults": n_default,
            "predicted_default_rate": round(n_default / total_loans, 4),
            "average_predicted_pd": round(float(avg_pd), 4),
            "flagged_high_risk_count": n_flagged,
            "flag_threshold": default_threshold,
        }
        if "grade" in results.columns:
            summary_stats["avg_pd_by_grade"] = (
                results.groupby("grade")["probability_of_default"].mean().round(4).to_dict()
            )

        with st.spinner("Asking Claude for a portfolio risk summary..."):
            commentary = generate_ai_commentary(api_key, summary_stats)
        st.markdown(commentary)

st.divider()
st.caption(
    "Model: XGBoost (tuned via RandomizedSearchCV) · "
    "Preprocessing: StandardScaler + OneHotEncoder fitted during training · "
    "See `model_development.ipynb` for full methodology."
)
