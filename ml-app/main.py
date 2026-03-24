import os
import sys
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(
    page_title="Restaurant Rating Predictor",
    page_icon="🍽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=EB+Garamond:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'EB Garamond', 'Times New Roman', Georgia, serif !important;
    }
    h1, h2, h3, h4 {
        font-family: 'Playfair Display', 'Times New Roman', Georgia, serif !important;
        font-weight: 700;
        color: #000000;
    }
    .stButton > button {
        background-color: #000000;
        color: #FFFFFF;
        border-radius: 4px;
        font-family: 'EB Garamond', serif;
        font-size: 16px;
        padding: 0.4rem 1.2rem;
        border: none;
    }
    .stButton > button:hover {
        background-color: #333333;
        color: #FFFFFF;
    }
    .metric-card {
        background: #F5F5F5;
        border-left: 4px solid #000000;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
        border-radius: 0 4px 4px 0;
    }
    .hero-title {
        font-family: 'Playfair Display', 'Times New Roman', Georgia, serif !important;
        font-size: 3rem;
        font-weight: 700;
        line-height: 1.1;
        color: #000000;
        margin-bottom: 0.5rem;
    }
    .hero-sub {
        font-family: 'EB Garamond', 'Times New Roman', Georgia, serif !important;
        font-size: 1.25rem;
        color: #444444;
        margin-bottom: 2rem;
    }
    .divider {
        border-top: 2px solid #000000;
        margin: 1.5rem 0;
    }
    .sidebar-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.1rem;
        font-weight: 700;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'EB Garamond', serif;
        font-size: 1rem;
    }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

DATA_PATH = os.path.join(os.path.dirname(__file__), "restaurant_data.csv")
FEATURES = ["Average Cost for two", "Price range", "Votes"]
TARGET = "Aggregate rating"


@st.cache_data(show_spinner=False)
def load_data():
    if not os.path.exists(DATA_PATH):
        return None, "Dataset file not found. Please download it first."
    try:
        df = pd.read_csv(DATA_PATH, encoding="latin-1")
        df = df[df[TARGET] > 0].copy()
        df = df.dropna(subset=FEATURES + [TARGET])
        for col in FEATURES + [TARGET]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=FEATURES + [TARGET])
        return df, None
    except Exception as e:
        return None, str(e)


@st.cache_resource(show_spinner=False)
def train_model(data_hash):
    df, err = load_data()
    if err or df is None:
        return None, None, None, err

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "R²": round(r2_score(y_test, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
        "MAE": round(mean_absolute_error(y_test, y_pred), 4),
        "Train Size": len(X_train),
        "Test Size": len(X_test),
    }
    results = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
    return model, metrics, results, None


def download_dataset():
    with st.spinner("Downloading Zomato dataset from Kaggle…"):
        try:
            import kagglehub, shutil
            path = kagglehub.dataset_download("shrutimehta/zomato-restaurants-data")
            for root, dirs, files in os.walk(path):
                for f in files:
                    if f.endswith(".csv"):
                        import shutil
                        shutil.copy(os.path.join(root, f), DATA_PATH)
                        st.success("Dataset downloaded successfully.")
                        st.cache_data.clear()
                        st.cache_resource.clear()
                        st.rerun()
                        return
            st.error("No CSV found in the Kaggle dataset package.")
        except Exception as e:
            st.error(f"Download failed: {e}")


def sidebar():
    with st.sidebar:
        st.markdown('<p class="sidebar-title">Restaurant Rating Predictor</p>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("**Model:** Random Forest Regressor")
        st.markdown("**Dataset:** Zomato Restaurants")
        st.markdown("**Features used:**")
        for f in FEATURES:
            st.markdown(f"- {f}")
        st.markdown(f"**Target:** {TARGET}")
        st.markdown("---")
        st.markdown("**GitHub Repository**")
        st.markdown("[DEIAKO / Restaurant-Recommendation-ML-Project](https://github.com/DEIAKO/Restaurant-Recommendation-ML-Project)")


def tab_home(df):
    st.markdown('<p class="hero-title">Restaurant Rating Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">A machine learning model that predicts restaurant aggregate ratings using the Zomato dataset.</p>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if df is None:
        st.warning("Dataset not found. Please download it below.")
        if st.button("Download Zomato Dataset"):
            download_dataset()
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Restaurants", f"{len(df):,}")
    with col2:
        st.metric("Avg Rating", f"{df[TARGET].mean():.2f}")
    with col3:
        st.metric("Avg Cost (₹)", f"{df['Average Cost for two'].median():,.0f}")
    with col4:
        st.metric("Total Votes", f"{df['Votes'].sum():,}")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Rating Distribution")
        fig = px.histogram(
            df, x=TARGET, nbins=30,
            color_discrete_sequence=["#000000"],
            template="simple_white",
        )
        fig.update_layout(
            xaxis_title="Aggregate Rating",
            yaxis_title="Count",
            font=dict(family="Georgia, serif"),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Cost vs Rating")
        sample = df.sample(min(1500, len(df)), random_state=42)
        fig2 = px.scatter(
            sample, x="Average Cost for two", y=TARGET,
            color="Price range",
            color_continuous_scale="Greys",
            template="simple_white",
            opacity=0.7,
        )
        fig2.update_layout(
            xaxis_title="Average Cost for Two (₹)",
            yaxis_title="Aggregate Rating",
            font=dict(family="Georgia, serif"),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### How it works")
    st.markdown("""
    This application trains a **Random Forest Regressor** on the Zomato restaurant dataset.
    The model learns patterns between a restaurant's price range, average cost, and number of votes
    to predict its aggregate rating on a scale of 1–5.

    Navigate the tabs above to explore the data, evaluate the model, and make your own predictions.
    """)


def tab_data(df):
    st.header("Data Explorer")

    if df is None:
        st.warning("No data available. Please download the dataset from the Home tab.")
        return

    st.markdown(f"**{len(df):,} rated restaurants** in the dataset after filtering out unrated entries.")

    with st.expander("Preview raw data"):
        cols_to_show = ["Restaurant Name", "City", "Cuisines", "Average Cost for two",
                        "Price range", "Votes", "Aggregate rating", "Rating text"]
        show_cols = [c for c in cols_to_show if c in df.columns]
        st.dataframe(df[show_cols].head(100), use_container_width=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Votes vs Rating")
        sample = df.sample(min(2000, len(df)), random_state=1)
        fig = px.scatter(
            sample, x="Votes", y=TARGET,
            color=TARGET,
            color_continuous_scale="Greys",
            template="simple_white",
            opacity=0.6,
        )
        fig.update_layout(
            font=dict(family="Georgia, serif"),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Rating by Price Range")
        fig2 = px.box(
            df, x="Price range", y=TARGET,
            color="Price range",
            color_discrete_sequence=["#333333", "#666666", "#999999", "#CCCCCC"],
            template="simple_white",
        )
        fig2.update_layout(
            font=dict(family="Georgia, serif"),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            showlegend=False,
            xaxis_title="Price Range (1=cheapest, 4=most expensive)",
            yaxis_title="Aggregate Rating",
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Feature Correlation")
    corr_cols = FEATURES + [TARGET]
    corr = df[corr_cols].corr()
    fig3 = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="Greys",
        template="simple_white",
        aspect="auto",
    )
    fig3.update_layout(
        font=dict(family="Georgia, serif"),
        paper_bgcolor="#FFFFFF",
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Descriptive Statistics")
    st.dataframe(df[corr_cols].describe().round(2), use_container_width=True)


def tab_model(df):
    st.header("Model Performance")

    if df is None:
        st.warning("No data available. Please download the dataset from the Home tab.")
        return

    with st.spinner("Training Random Forest model…"):
        data_hash = str(len(df))
        model, metrics, results, err = train_model(data_hash)

    if err:
        st.error(f"Model training failed: {err}")
        return

    st.markdown("### Evaluation Metrics")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("R² Score", metrics["R²"], help="Proportion of variance explained (closer to 1 is better)")
    with c2:
        st.metric("RMSE", metrics["RMSE"], help="Root Mean Squared Error (lower is better)")
    with c3:
        st.metric("MAE", metrics["MAE"], help="Mean Absolute Error (lower is better)")

    st.markdown(f"*Trained on {metrics['Train Size']:,} samples — tested on {metrics['Test Size']:,} samples.*")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Actual vs Predicted")
        fig = px.scatter(
            results, x="Actual", y="Predicted",
            opacity=0.5,
            color_discrete_sequence=["#000000"],
            template="simple_white",
        )
        fig.add_shape(
            type="line",
            x0=results["Actual"].min(), y0=results["Actual"].min(),
            x1=results["Actual"].max(), y1=results["Actual"].max(),
            line=dict(color="#CC0000", dash="dash", width=2),
        )
        fig.update_layout(
            font=dict(family="Georgia, serif"),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            xaxis_title="Actual Rating",
            yaxis_title="Predicted Rating",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Residuals Distribution")
        residuals = results["Actual"] - results["Predicted"]
        fig2 = px.histogram(
            residuals, nbins=40,
            color_discrete_sequence=["#000000"],
            template="simple_white",
        )
        fig2.update_layout(
            xaxis_title="Residual (Actual − Predicted)",
            yaxis_title="Count",
            font=dict(family="Georgia, serif"),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Feature Importance")
    importances = model.feature_importances_
    fi_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": importances
    }).sort_values("Importance", ascending=True)

    fig3 = px.bar(
        fi_df, x="Importance", y="Feature",
        orientation="h",
        color_discrete_sequence=["#000000"],
        template="simple_white",
        text=fi_df["Importance"].apply(lambda x: f"{x:.1%}"),
    )
    fig3.update_layout(
        font=dict(family="Georgia, serif"),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        xaxis_title="Importance Score",
        yaxis_title="",
        xaxis_tickformat=".0%",
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### Model Configuration")
    st.code("""RandomForestRegressor(
    n_estimators = 200,
    max_depth    = 12,
    min_samples_split = 4,
    min_samples_leaf  = 2,
    random_state = 42
)""", language="python")


def tab_predict(df):
    st.header("Make a Prediction")
    st.markdown("Enter restaurant details below to get a predicted aggregate rating.")

    if df is None:
        st.warning("No data available. Please download the dataset from the Home tab.")
        return

    with st.spinner("Loading model…"):
        data_hash = str(len(df))
        model, metrics, results, err = train_model(data_hash)

    if err or model is None:
        st.error("Model could not be loaded.")
        return

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Average Cost for Two (₹)**")
        cost = st.slider(
            "cost", min_value=50, max_value=8000, value=600, step=50,
            label_visibility="collapsed"
        )

    with col2:
        st.markdown("**Price Range**")
        price_range = st.selectbox(
            "price_range",
            options=[1, 2, 3, 4],
            format_func=lambda x: {1: "1 — Budget", 2: "2 — Moderate", 3: "3 — Expensive", 4: "4 — Luxury"}[x],
            index=1,
            label_visibility="collapsed"
        )

    with col3:
        st.markdown("**Number of Votes**")
        votes = st.slider(
            "votes", min_value=0, max_value=15000, value=200, step=50,
            label_visibility="collapsed"
        )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if st.button("Predict Rating", use_container_width=False):
        input_data = pd.DataFrame([{
            "Average Cost for two": cost,
            "Price range": price_range,
            "Votes": votes,
        }])
        prediction = model.predict(input_data)[0]
        prediction = float(np.clip(prediction, 1.0, 5.0))

        col_r, col_i = st.columns([1, 2])
        with col_r:
            stars = "★" * round(prediction) + "☆" * (5 - round(prediction))
            st.markdown(f"""
            <div style="background:#000;color:#fff;padding:2rem;border-radius:8px;text-align:center;">
                <div style="font-size:3rem;font-family:'Playfair Display',serif;font-weight:700;">{prediction:.2f}</div>
                <div style="font-size:1.5rem;letter-spacing:4px;margin-top:0.5rem;">{stars}</div>
                <div style="font-size:0.9rem;margin-top:0.5rem;opacity:0.8;">Predicted Rating / 5.0</div>
            </div>
            """, unsafe_allow_html=True)

        with col_i:
            st.markdown("**Interpretation**")
            if prediction >= 4.5:
                label, desc = "Exceptional", "This restaurant is expected to be among the very best."
            elif prediction >= 4.0:
                label, desc = "Excellent", "Highly rated — customers love this restaurant."
            elif prediction >= 3.5:
                label, desc = "Very Good", "Above average — most customers are satisfied."
            elif prediction >= 3.0:
                label, desc = "Good", "A decent restaurant with room for improvement."
            elif prediction >= 2.5:
                label, desc = "Average", "Mediocre — mixed customer experience."
            else:
                label, desc = "Below Average", "This restaurant needs significant improvement."

            st.markdown(f"**{label}** — {desc}")
            st.markdown("")
            st.markdown(f"- Cost for two: ₹{cost:,}")
            st.markdown(f"- Price range: {price_range}/4")
            st.markdown(f"- Number of votes: {votes:,}")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### Batch Prediction")
    st.markdown("Upload a CSV file with columns `Average Cost for two`, `Price range`, and `Votes` to predict ratings for multiple restaurants.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded)
            missing = [c for c in FEATURES if c not in batch.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                preds = model.predict(batch[FEATURES])
                preds = np.clip(preds, 1.0, 5.0)
                batch["Predicted Rating"] = preds.round(2)
                st.success(f"Predicted ratings for {len(batch):,} restaurants.")
                st.dataframe(batch, use_container_width=True)
                csv_out = batch.to_csv(index=False).encode("utf-8")
                st.download_button("Download predictions", csv_out, "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Error processing file: {e}")


def tab_about():
    st.header("About This Project")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    ### Project Overview

    This application is a **Restaurant Rating Predictor** built as a machine learning portfolio project.
    It demonstrates an end-to-end ML pipeline from data acquisition to a live interactive web app.

    ---

    ### Dataset

    - **Source:** [Zomato Restaurants Data on Kaggle](https://www.kaggle.com/datasets/shrutimehta/zomato-restaurants-data)
    - **Size:** ~9,551 restaurants worldwide
    - **Usable rows:** ~7,403 (after removing unrated entries)
    - **Features used:**

    | Feature | Description |
    |---|---|
    | Average Cost for two | Average dining cost in local currency |
    | Price range | Ordinal scale 1–4 (budget to luxury) |
    | Votes | Number of user votes received |

    ---

    ### Machine Learning Pipeline

    1. **Data Loading** — CSV read with `latin-1` encoding
    2. **Cleaning** — Remove rows with zero or missing target rating
    3. **Feature Engineering** — Numeric coercion, drop nulls
    4. **Train/Test Split** — 80% / 20% stratified split
    5. **Model** — `RandomForestRegressor` with 200 trees
    6. **Evaluation** — R², RMSE, MAE on held-out test set

    ---

    ### Key Findings

    - **Votes** is by far the most important predictor (~74% importance)
    - **Price range** contributes modestly (~5%)
    - **Average Cost for two** fills in the remaining predictive power

    ---

    ### Tech Stack

    | Layer | Tools |
    |---|---|
    | Language | Python 3.11 |
    | ML | scikit-learn |
    | Data | pandas, numpy |
    | Visualisation | plotly |
    | App | Streamlit |
    | Data Source | kagglehub (Kaggle API) |

    ---

    ### Repository

    [https://github.com/DEIAKO/Restaurant-Recommendation-ML-Project](https://github.com/DEIAKO/Restaurant-Recommendation-ML-Project)
    """)


def main():
    sidebar()

    df, err = load_data()
    if err and "not found" not in err:
        st.error(f"Error loading data: {err}")
        df = None

    tabs = st.tabs(["Home", "Data Explorer", "Model Performance", "Predict", "About"])

    with tabs[0]:
        tab_home(df)
    with tabs[1]:
        tab_data(df)
    with tabs[2]:
        tab_model(df)
    with tabs[3]:
        tab_predict(df)
    with tabs[4]:
        tab_about()


if __name__ == "__main__":
    main()
