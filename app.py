
from pathlib import Path
import pickle

import pandas as pd
import streamlit as st
import tensorflow as tf

from bs4 import BeautifulSoup
from tensorflow.keras.preprocessing.sequence import pad_sequences


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
RESULTS_PATH = BASE_DIR / "results" / "model_results.csv"

MODEL_PATH = MODEL_DIR / "best_sentiment_model.keras"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.pkl"
CONFIG_PATH = MODEL_DIR / "config.pkl"

st.set_page_config(
    page_title="Sentiment AI | IMDB",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# PROFESSIONAL UI
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #0b1120;
    --panel: #111b2e;
    --border: #24334b;
    --text: #edf3fc;
    --muted: #91a3bd;
    --accent: #4fd1c5;
    --blue: #7aa7ff;
}

.stApp {
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', sans-serif;
}

[data-testid="stHeader"] {
    background: rgba(11, 17, 32, 0.95);
}

[data-testid="stSidebar"] {
    background: #0e1728;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    color: #c5d2e5;
}

.block-container {
    padding-top: 2.2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

h1, h2, h3 {
    color: var(--text) !important;
    letter-spacing: -0.5px;
}

p, label, .stMarkdown {
    color: #c1cde0;
}

.hero {
    background: linear-gradient(115deg, #172944, #122b36);
    border: 1px solid #2a455d;
    padding: 30px;
    border-radius: 20px;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 32px;
    font-weight: 800;
    margin: 0 0 10px 0;
}

.hero p {
    font-size: 15px;
    color: #b6c8df;
    margin-bottom: 0;
}

.eyebrow {
    color: var(--accent);
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.metric-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px;
    min-height: 132px;
}

.metric-label {
    color: var(--muted);
    font-size: 13px;
    margin-bottom: 12px;
}

.metric-value {
    color: var(--text);
    font-size: 29px;
    font-weight: 800;
    letter-spacing: -0.7px;
}

.metric-note {
    color: var(--accent);
    font-size: 12px;
    margin-top: 8px;
}

.section-title {
    font-size: 20px;
    font-weight: 700;
    color: var(--text);
    margin: 25px 0 5px 0;
}

.section-subtitle {
    color: var(--muted);
    font-size: 13px;
    margin-bottom: 18px;
}

.result-card {
    background: #13273a;
    border: 1px solid #2d5364;
    border-radius: 18px;
    padding: 26px;
    margin-top: 20px;
}

.result-positive {
    color: #5ee0b5;
    font-size: 27px;
    font-weight: 800;
}

.result-negative {
    color: #ff929e;
    font-size: 27px;
    font-weight: 800;
}

.info-card {
    background: var(--panel);
    border: 1px solid var(--border);
    padding: 22px;
    border-radius: 16px;
}

div[data-testid="stMetric"] {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 18px;
}

div[data-testid="stMetricLabel"] {
    color: var(--muted);
}

div[data-testid="stMetricValue"] {
    color: var(--text);
}

.stTextArea textarea {
    background: #111b2e;
    color: #edf3fc;
    border: 1px solid #2a3b55;
    border-radius: 12px;
    line-height: 1.7;
}

.stTextArea textarea:focus {
    border-color: var(--accent);
}

.stButton > button {
    background: #4fd1c5;
    color: #071521;
    border: none;
    border-radius: 10px;
    padding: 11px 20px;
    font-weight: 800;
    min-height: 44px;
    transition: 0.2s ease;
}

.stButton > button:hover {
    background: #83e5db;
    color: #071521;
    border: none;
}

div[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
}

hr {
    border-color: var(--border);
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD SAVED MODEL - NO TRAINING
# =========================================================

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    with open(CONFIG_PATH, "rb") as f:
        config = pickle.load(f)

    return model, tokenizer, config


@st.cache_data
def load_results():
    if not RESULTS_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(RESULTS_PATH)


def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    return " ".join(text.lower().split())


def metric_card(label, value, note):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# CHECK FILES
# =========================================================

required_files = [
    MODEL_PATH,
    TOKENIZER_PATH,
    CONFIG_PATH,
]

if not all(path.exists() for path in required_files):
    st.error(
        "Model files are missing. Please check the models folder."
    )
    st.code(
        "models/best_sentiment_model.keras\n"
        "models/tokenizer.pkl\n"
        "models/config.pkl"
    )
    st.stop()

try:
    model, tokenizer, config = load_artifacts()
    results = load_results()
except Exception as error:
    st.error(f"Could not load the saved artifacts: {error}")
    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 15px 5px 25px 5px;">
            <div style="font-size: 30px;">🎬</div>
            <div style="font-size: 21px; font-weight: 800;
                        color: #edf3fc; margin-top: 8px;">
                Sentiment<span style="color:#4fd1c5;">AI</span>
            </div>
            <div style="font-size: 12px; color: #91a3bd;
                        margin-top: 5px;">
                IMDB Deep Learning Studio
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("##### WORKSPACE")

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Sentiment Analyzer",
            "Model Performance",
            "About Project",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("##### MODEL STATUS")

    st.markdown(
        """
        <div style="background:#13273a; padding:14px;
                    border-radius:12px; border:1px solid #294359;">
            <span style="color:#5ee0b5;">●</span>
            <span style="color:#edf3fc; font-weight:600;">
                Model loaded
            </span>
            <div style="font-size:12px; color:#91a3bd;
                        margin-top:8px;">
                Running locally
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")
    st.caption("IMDB Sentiment Analysis · RNN / LSTM / GRU")


# =========================================================
# PREPARE RESULTS
# =========================================================

has_results = (
    not results.empty
    and {"Model", "Accuracy", "F1-Score"}.issubset(results.columns)
)

best_row = None

if has_results:
    results = results.copy()
    results["Accuracy"] = pd.to_numeric(
        results["Accuracy"], errors="coerce"
    )
    results["F1-Score"] = pd.to_numeric(
        results["F1-Score"], errors="coerce"
    )

    results = results.dropna(
        subset=["Accuracy", "F1-Score"]
    )

    if not results.empty:
        best_row = results.loc[results["Accuracy"].idxmax()]


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">MACHINE LEARNING / OVERVIEW</div>
            <h1>Sentiment Analytics Dashboard</h1>
            <p>
                Understand model performance and explore movie review
                sentiment predictions in one workspace.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if best_row is not None:
        accuracy_value = f"{best_row['Accuracy']:.2%}"
        f1_value = f"{best_row['F1-Score']:.2%}"
        best_name = str(best_row["Model"])
        model_count = str(len(results))
    else:
        accuracy_value = "N/A"
        f1_value = "N/A"
        best_name = str(config.get("best_model", "Unknown"))
        model_count = "N/A"

    c1, c2, c3, c4 = st.columns(4, gap="medium")

    with c1:
        metric_card("BEST TEST ACCURACY", accuracy_value, "Saved evaluation")

    with c2:
        metric_card("BEST F1-SCORE", f1_value, "Saved evaluation")

    with c3:
        metric_card("SELECTED MODEL", best_name, "Loaded from disk")

    with c4:
        metric_card("MODELS COMPARED", model_count, "Training results")

    st.markdown(
        '<div class="section-title">Performance Overview</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-subtitle">'
        'Compare the recorded test metrics of your trained models.'
        '</div>',
        unsafe_allow_html=True,
    )

    if has_results and not results.empty:
        left, right = st.columns([1.25, 1], gap="large")

        with left:
            st.markdown("#### Model Comparison")

            chart_data = results.set_index("Model")[
                ["Accuracy", "F1-Score"]
            ]

            st.bar_chart(
                chart_data,
                color=["#4fd1c5", "#7aa7ff"],
                height=320,
            )

        with right:
            st.markdown("#### Recorded Results")

            display_df = results.copy()

            for col in ["Accuracy", "Precision", "Recall", "F1-Score"]:
                if col in display_df.columns:
                    display_df[col] = display_df[col].map(
                        lambda x: f"{x:.2%}" if pd.notna(x) else "N/A"
                    )

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info(
            "The model comparison CSV is missing or does not contain "
            "the expected columns."
        )

    st.markdown('<div class="section-title">Quick Actions</div>',
                unsafe_allow_html=True)

    a, b = st.columns(2)

    with a:
        st.markdown(
            """
            <div class="info-card">
                <h3>💬 Sentiment Analyzer</h3>
                <p>Enter a movie review and predict its sentiment.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open Sentiment Analyzer", use_container_width=True):
            st.session_state["next_page"] = "Sentiment Analyzer"
            st.rerun()

    with b:
        st.markdown(
            """
            <div class="info-card">
                <h3>📊 Model Performance</h3>
                <p>Review the saved metrics for the trained models.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Explore Model Performance", use_container_width=True):
            st.session_state["next_page"] = "Model Performance"
            st.rerun()


# =========================================================
# SENTIMENT ANALYZER
# =========================================================

elif page == "Sentiment Analyzer":

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">AI PREDICTION</div>
            <h1>Movie Review Analyzer</h1>
            <p>
                Enter a review in English to classify it as positive
                or negative using your trained neural network.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Your Review")

    review = st.text_area(
        "Review text",
        placeholder=(
            "Example: The acting was excellent and the story "
            "kept me interested from beginning to end..."
        ),
        height=190,
        label_visibility="collapsed",
        key="review_input",
    )

    st.caption(
        f"Characters entered: {len(review)} · "
        f"Selected model: {config.get('best_model', 'Unknown')}"
    )

    if st.button("✨ Analyze Sentiment", use_container_width=True):

        if not review.strip():
            st.warning("Please enter a movie review first.")

        else:
            with st.spinner("Analyzing review..."):

                cleaned_review = clean_text(review)

                sequence = tokenizer.texts_to_sequences(
                    [cleaned_review]
                )

                padded = pad_sequences(
                    sequence,
                    maxlen=config["MAX_LEN"],
                    padding="post",
                    truncating="post",
                )

                probability = float(
                    model.predict(padded, verbose=0)[0][0]
                )

                if probability >= 0.5:
                    sentiment = "Positive"
                    confidence = probability
                    css_class = "result-positive"
                    emoji = "😊"
                else:
                    sentiment = "Negative"
                    confidence = 1 - probability
                    css_class = "result-negative"
                    emoji = "😞"

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="eyebrow">PREDICTION RESULT</div>
                    <div class="{css_class}">
                        {emoji} {sentiment} Sentiment
                    </div>
                    <p style="margin-top:12px;">
                        Confidence score: <b>{confidence:.2%}</b>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.progress(confidence)

            st.caption(
                "Confidence reflects the model's output, "
                "not a guarantee that the prediction is correct."
            )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">EVALUATION CENTER</div>
            <h1>Model Performance</h1>
            <p>
                Review the saved evaluation metrics for RNN, LSTM,
                and GRU. No models are retrained on this page.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if has_results and not results.empty:

        metric_options = [
            col for col in
            ["Accuracy", "Precision", "Recall", "F1-Score"]
            if col in results.columns
        ]

        selected_metrics = st.multiselect(
            "Metrics to compare",
            metric_options,
            default=[
                m for m in ["Accuracy", "F1-Score"]
                if m in metric_options
            ],
        )

        if selected_metrics:
            st.markdown("#### Metric Comparison")

            st.bar_chart(
                results.set_index("Model")[selected_metrics],
                color=["#4fd1c5", "#7aa7ff", "#b49aff", "#f6bd60"],
                height=380,
            )

        st.markdown("#### Full Evaluation Table")

        formatted = results.copy()

        for col in ["Accuracy", "Precision", "Recall", "F1-Score"]:
            if col in formatted.columns:
                formatted[col] = formatted[col].map(
                    lambda x: f"{x:.2%}" if pd.notna(x) else "N/A"
                )

        st.dataframe(
            formatted,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download Evaluation Results CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="model_results.csv",
            mime="text/csv",
        )

        if best_row is not None:
            st.success(
                f"Highest recorded accuracy: {best_row['Model']} "
                f"({best_row['Accuracy']:.2%})."
            )

    else:
        st.warning(
            "No valid saved model results were found. "
            "Check results/model_results.csv."
        )


# =========================================================
# ABOUT PROJECT
# =========================================================

elif page == "About Project":

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">PROJECT INFORMATION</div>
            <h1>About Sentiment AI</h1>
            <p>
                A deep learning project for binary sentiment
                classification of IMDB movie reviews.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Project Overview")

    st.write(
        "The project compares recurrent neural network architectures "
        "for classifying movie reviews as positive or negative."
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            """
            <div class="info-card">
                <h3>🧠 Architectures</h3>
                <p>SimpleRNN</p>
                <p>LSTM</p>
                <p>GRU</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="info-card">
                <h3>⚙️ Text Processing</h3>
                <p>HTML removal and lowercasing</p>
                <p>Tokenization</p>
                <p>Sequence padding and truncation</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Current Model Configuration")

    st.json(
        {
            "Selected model": config.get("best_model"),
            "Maximum vocabulary": config.get("MAX_WORDS"),
            "Maximum review length": config.get("MAX_LEN"),
            "Embedding dimension": config.get("EMBEDDING_DIM"),
            "Recurrent units": config.get("RNN_UNITS"),
        }
    )

    st.caption(
        "This interface loads previously saved artifacts. "
        "Training is performed separately."
    )
