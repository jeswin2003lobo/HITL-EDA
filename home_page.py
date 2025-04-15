import streamlit as st

def custom_css():
    st.markdown("""
    <style>
    body {
        background-color: #f5f5f5;
        font-family: "Segoe UI", sans-serif;
    }
    .target-audience {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        margin-top: 20px;
    }
    .audience {
        background-color: #ffffff;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin: 10px;
        padding: 20px;
        width: 200px;
        text-align: center;
        transition: transform 0.2s;
    }
    .audience:hover {
        transform: scale(1.05);
    }
    .audience-icon {
        font-size: 2.5rem;
        margin-bottom: 10px;
    }
    .audience-title {
        font-size: 1.1rem;
        font-weight: 600;
    }
    .thank-you {
        margin-top: 30px;
        font-size: 1.3rem;
        color: #444;
        text-align: center;
    }
    .start-button {
        display: inline-block;
        background-color: #4CAF50;
        color: white;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: bold;
        margin-top: 30px;
        text-decoration: none;
    }
    </style>
    """, unsafe_allow_html=True)

def show_home_page():
    custom_css()

    st.title("🤖 Smart Data Understanding & Synthetic Data Generation")
    st.markdown("A unified AutoEDA + Data Synthesis platform for efficient AI pipelines.")

    # Key Features
    st.subheader("✨ Key Features")
    st.markdown("""
    - 📊 **Interactive Exploration:** Explore datasets with intuitive visualizations.
    - 🧠 **AutoEDA Engine:** Automatically detect patterns, nulls, outliers, and correlations.
    - 🧬 **Synthetic Data Generator:** Generate privacy-preserving datasets using CTGAN, TVAE, etc.
    - 🛠️ **Seamless Preprocessing:** Handle missing values, encoding, and scaling.
    - 🔒 **Privacy Aware:** Built-in compliance for HIPAA/GDPR-like environments.
    """)

    # Target Audience
    st.subheader("👥 Who is this for?")
    st.markdown('<div class="target-audience">', unsafe_allow_html=True)
    audience_list = [
        ("📊", "Data Analysts"),
        ("🔍", "Data Scientists"),
        ("💼", "Business Professionals"),
        ("🎓", "Students & Educators"),
        ("🏥", "Healthcare Experts"),
        ("💻", "ML Engineers"),
    ]
    for icon, title in audience_list:
        st.markdown(
            f"""<div class="audience">
                <div class="audience-icon">{icon}</div>
                <div class="audience-title">{title}</div>
            </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Get Started
    st.subheader("🚀 Get Started with HITL-EDA")
    st.markdown("""
    Upload your dataset or try our built-in sample. Select preprocessing options, run HITL-EDA, and download clean or synthetic datasets for ML pipelines.
    """)

