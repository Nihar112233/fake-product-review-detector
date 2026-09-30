import os
import joblib
import pandas as pd
import streamlit as st

from train_model import (
    MODEL_DIR,
    DATASET_PATH,
    load_and_clean_data,
    train_and_evaluate_pipeline,
    predict_review,
)

st.set_page_config(
    page_title="Fake Review Detector",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
<style>
    /* Global Background */
    .stApp {
        background-color: #f0f2f5;
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Sidebar */
    [data-testid="stSidebar"],
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"] {
        display: none !important;
        visibility: hidden !important;
    }

    /* Header Styling */
    .main-header {
        text-align: center;
        padding: 3rem 1rem 2rem 1rem;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: white;
        border-radius: 0 0 24px 24px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
        letter-spacing: -0.05em;
        color: white;
    }
    .main-header p {
        font-size: 1.15rem;
        color: #94a3b8;
        font-weight: 400;
        max-width: 600px;
        margin: 0 auto;
    }

    /* Cards */
    .card-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid #f1f5f9;
        padding-bottom: 0.8rem;
        margin-top: 0;
    }

    /* Results */
    .result-container {
        margin-top: 1.5rem;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        animation: fadeIn 0.4s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-5px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .result-genuine {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border: 2px solid #10b981;
    }
    .result-fake {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 2px solid #ef4444;
    }
    .result-label {
        font-size: 2rem;
        font-weight: 900;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .genuine-color { color: #047857; }
    .fake-color { color: #b91c1c; }
    .result-conf {
        color: #475569;
        font-size: 1.1rem;
        font-weight: 600;
    }

    /* Analyze Button */
    .stButton>button {
        font-weight: 700;
        font-size: 1.1rem;
        padding: 0.6rem 1rem;
        border-radius: 10px;
        transition: all 0.2s;
    }
    .stButton>button[kind="primary"] {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        border: none;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.4);
    }
    .stButton>button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px -1px rgba(37, 99, 235, 0.5);
    }

    /* Demo Cards */
    .demo-text {
        color: #334155;
        font-size: 0.95rem;
        font-style: italic;
        margin-bottom: 1rem;
        line-height: 1.5;
    }
    .demo-meta {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

def get_models_mtime():
    lr_path = os.path.join(MODEL_DIR, "lr_model.joblib")
    if os.path.exists(lr_path):
        return os.path.getmtime(lr_path)
    return 0

@st.cache_resource(show_spinner="Loading trained models...")
def load_models(_mtime):
    prep_path = os.path.join(MODEL_DIR, "preprocessor.joblib")
    lr_path = os.path.join(MODEL_DIR, "lr_model.joblib")
    nb_path = os.path.join(MODEL_DIR, "nb_model.joblib")
    metrics_path = os.path.join(MODEL_DIR, "metrics.joblib")

    if not all(os.path.exists(p) for p in (prep_path, lr_path, nb_path, metrics_path)):
        if not os.path.exists(DATASET_PATH):
            raise FileNotFoundError(
                "Dataset and trained models are missing. "
                "Add dataset/amazon_reviews.txt and run python train_model.py"
            )
        df = load_and_clean_data(DATASET_PATH)
        artifacts = train_and_evaluate_pipeline(df, save_artifacts=True)
        st.cache_resource.clear()
        return artifacts

    return {
        "preprocessor": joblib.load(prep_path),
        "lr_model": joblib.load(lr_path),
        "nb_model": joblib.load(nb_path),
        "results": joblib.load(metrics_path),
    }

try:
    artifacts = load_models(get_models_mtime())
except Exception as exc:
    st.error(f"Error loading models: {exc}")
    st.stop()

results = artifacts["results"]
lr_m = results["lr_metrics"]
nb_m = results["nb_metrics"]

# Session State Initialization
if "nav" not in st.session_state:
    st.session_state.nav = "Analyzer"
if "review_text" not in st.session_state:
    st.session_state.review_text = ""
if "rating" not in st.session_state:
    st.session_state.rating = 5
if "vp" not in st.session_state:
    st.session_state.vp = "N"
if "category" not in st.session_state:
    st.session_state.category = "Wireless"

# Header
st.markdown("""
    <div class="main-header">
        <h1>FAKE REVIEW DETECTOR</h1>
        <p>AI-powered Product Review Analysis to identify deceptive opinion spam using machine learning.</p>
    </div>
""", unsafe_allow_html=True)

# Main Navigation
nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
with nav_col2:
    st.session_state.nav = st.radio(
        "Navigation",
        ["Analyzer", "Model Performance"],
        horizontal=True,
        label_visibility="collapsed"
    )

st.markdown("<br>", unsafe_allow_html=True)

if st.session_state.nav == "Analyzer":
    # Analyzer Layout
    
    col_main, col_demo = st.columns([55, 45], gap="large")
    
    with col_main:
        with st.container(border=True):
            st.markdown('<div class="card-title">🔍 REVIEW ANALYZER</div>', unsafe_allow_html=True)
            
            review_text = st.text_area(
                "Product Review",
                value=st.session_state.review_text,
                height=200,
                placeholder="Enter or paste your product review here...",
            )
            
            c1, c2 = st.columns(2)
            with c1:
                rating = st.selectbox(
                    "Rating",
                    [1, 2, 3, 4, 5],
                    index=[1, 2, 3, 4, 5].index(st.session_state.rating) if st.session_state.rating in [1, 2, 3, 4, 5] else 4
                )
            with c2:
                vp_index = 0 if st.session_state.vp == "Y" else 1
                verified_purchase = st.selectbox(
                    "Verified Purchase",
                    ["Y", "N"],
                    index=vp_index
                )
                
            product_category = st.text_input(
                "Product Category",
                value=st.session_state.category,
                placeholder="e.g. Wireless, Electronics, Beauty"
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            analyze_btn = st.button("ANALYZE REVIEW", type="primary", use_container_width=True)
            
            if analyze_btn:
                output = predict_review(
                    review_text=review_text,
                    rating=rating,
                    verified_purchase=verified_purchase,
                    product_category=product_category,
                    preprocessor=artifacts["preprocessor"],
                    model=artifacts["lr_model"],
                    algorithm="Logistic Regression",
                )
                
                if not output["ok"]:
                    st.error(output["error"])
                else:
                    is_genuine = output["class_idx"] == 0
                    box_class = "result-genuine" if is_genuine else "result-fake"
                    color_class = "genuine-color" if is_genuine else "fake-color"
                    
                    conf_val = f"{output['confidence'] * 100:.1f}%" if output["confidence"] else "N/A"
                    
                    st.markdown(
                        f"""
                        <div class="result-container {box_class}">
                            <div style="font-size: 1.1rem; font-weight: 700; color: #64748b; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;">Analysis Result</div>
                            <div class="result-label {color_class}">{output["prediction"].upper()}</div>
                            <div class="result-conf">Confidence: {conf_val}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    with col_demo:
        with st.container(border=True):
            st.markdown('<div class="card-title">✨ QUICK DEMO EXAMPLES</div>', unsafe_allow_html=True)
            st.write("Try a sample review to see the model in action:")
            
            def load_example(t, r, v, c):
                st.session_state.review_text = t
                st.session_state.rating = r
                st.session_state.vp = v
                st.session_state.category = c

            demo_tabs = st.tabs(["Potentially Fake", "Potentially Genuine"])
            
            with demo_tabs[0]:
                with st.container(border=True):
                    st.markdown('<div class="demo-text">"This is absolutely the BEST product I have ever purchased! The quality is perfect and far better than every other product available..."</div>', unsafe_allow_html=True)
                    st.markdown('<div class="demo-meta">5 ★ &nbsp;•&nbsp; Unverified (N) &nbsp;•&nbsp; Wireless</div>', unsafe_allow_html=True)
                    if st.button("Use Fake Example 1", use_container_width=True):
                        load_example("This is absolutely the BEST product I have ever purchased! The quality is perfect and far better than every other product available. Everyone should buy this immediately. I have never been so impressed with a product. 100% recommended and definitely worth every penny!", 5, "N", "Wireless")
                        st.rerun()

                with st.container(border=True):
                    st.markdown('<div class="demo-text">"Outstanding product!!! Amazing quality!!! Nothing else even comes close to this. This is easily the greatest purchase I have ever made..."</div>', unsafe_allow_html=True)
                    st.markdown('<div class="demo-meta">5 ★ &nbsp;•&nbsp; Unverified (N) &nbsp;•&nbsp; Beauty</div>', unsafe_allow_html=True)
                    if st.button("Use Fake Example 2", use_container_width=True):
                        load_example("Outstanding product!!! Amazing quality!!! Nothing else even comes close to this. This is easily the greatest purchase I have ever made. Five stars without hesitation. You will not find anything better anywhere!", 5, "N", "Beauty")
                        st.rerun()

                with st.container(border=True):
                    st.markdown('<div class="demo-text">"Perfect in every possible way. Incredible quality, unbelievable performance, and absolutely no problems at all. This product is far superior..."</div>', unsafe_allow_html=True)
                    st.markdown('<div class="demo-meta">5 ★ &nbsp;•&nbsp; Unverified (N) &nbsp;•&nbsp; Office Products</div>', unsafe_allow_html=True)
                    if st.button("Use Fake Example 3", use_container_width=True):
                        load_example("Perfect in every possible way. Incredible quality, unbelievable performance, and absolutely no problems at all. This product is far superior to anything else on the market. Everyone needs to own one!", 5, "N", "Office Products")
                        st.rerun()

            with demo_tabs[1]:
                with st.container(border=True):
                    st.markdown('<div class="demo-text">"I have been using these headphones for about three weeks. The sound quality is good and the battery lasts most of the day..."</div>', unsafe_allow_html=True)
                    st.markdown('<div class="demo-meta">3 ★ &nbsp;•&nbsp; Verified (Y) &nbsp;•&nbsp; Wireless</div>', unsafe_allow_html=True)
                    if st.button("Use Genuine Example 1", use_container_width=True):
                        load_example("I have been using these headphones for about three weeks. The sound quality is good and the battery lasts most of the day. The buttons are sometimes difficult to press, though. Overall they are decent for the price, but I would consider other options if you need better controls.", 3, "Y", "Wireless")
                        st.rerun()

                with st.container(border=True):
                    st.markdown('<div class="demo-text">"The mouse works well for normal daily use. I like the shape and the buttons feel responsive. The only issue I noticed is that the scroll wheel occasionally feels a little rough..."</div>', unsafe_allow_html=True)
                    st.markdown('<div class="demo-meta">4 ★ &nbsp;•&nbsp; Verified (Y) &nbsp;•&nbsp; PC</div>', unsafe_allow_html=True)
                    if st.button("Use Genuine Example 2", use_container_width=True):
                        load_example("The mouse works well for normal daily use. I like the shape and the buttons feel responsive. The only issue I noticed is that the scroll wheel occasionally feels a little rough. It is still a good option for the price.", 4, "Y", "PC")
                        st.rerun()

                with st.container(border=True):
                    st.markdown('<div class="demo-text">"I bought this charger because I needed a spare for my laptop. It charges at the expected speed and has worked reliably for the past month..."</div>', unsafe_allow_html=True)
                    st.markdown('<div class="demo-meta">4 ★ &nbsp;•&nbsp; Verified (Y) &nbsp;•&nbsp; Electronics</div>', unsafe_allow_html=True)
                    if st.button("Use Genuine Example 3", use_container_width=True):
                        load_example("I bought this charger because I needed a spare for my laptop. It charges at the expected speed and has worked reliably for the past month. The cable is a little shorter than I expected, but otherwise I have no major complaints.", 4, "Y", "Electronics")
                        st.rerun()

elif st.session_state.nav == "Model Performance":
    
    with st.container(border=True):
        st.markdown('<div class="card-title">📊 DATASET OVERVIEW</div>', unsafe_allow_html=True)
        
        col_do1, col_do2, col_do3, col_do4, col_do5 = st.columns(5)
        col_do1.metric("Total Reviews", f"{results['total_samples']:,}")
        col_do2.metric("Genuine Reviews", f"{results['genuine_count']:,}")
        col_do3.metric("Fake Reviews", f"{results['fake_count']:,}")
        col_do4.metric("Training Set", f"{results['train_size']:,}")
        col_do5.metric("Testing Set", f"{results['test_size']:,}")
    
    with st.container(border=True):
        st.markdown('<div class="card-title">📈 MODEL COMPARISON</div>', unsafe_allow_html=True)
        
        def format_pct(value):
            return f"{value * 100:.2f}%"

        metrics_table = pd.DataFrame(
            {
                "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
                "Naive Bayes": [
                    format_pct(nb_m["accuracy"]),
                    format_pct(nb_m["precision"]),
                    format_pct(nb_m["recall"]),
                    format_pct(nb_m["f1_score"]),
                ],
                "Logistic Regression": [
                    format_pct(lr_m["accuracy"]),
                    format_pct(lr_m["precision"]),
                    format_pct(lr_m["recall"]),
                    format_pct(lr_m["f1_score"]),
                ],
            }
        ).set_index("Metric")
        
        st.table(metrics_table)

        def confusion_as_frame(cm):
            return pd.DataFrame(
                cm,
                index=["Actual Genuine", "Actual Fake"],
                columns=["Predicted Genuine", "Predicted Fake"],
            )

        cm_col1, cm_col2 = st.columns(2)
        with cm_col1:
            st.markdown("**Naive Bayes Confusion Matrix**")
            st.dataframe(confusion_as_frame(nb_m["confusion_matrix"]), use_container_width=True)
        with cm_col2:
            st.markdown("**Logistic Regression Confusion Matrix**")
            st.dataframe(confusion_as_frame(lr_m["confusion_matrix"]), use_container_width=True)
