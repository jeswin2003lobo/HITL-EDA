import streamlit as st

# Inject custom CSS based on SRS UI requirements
def custom_css():
    st.markdown("""
    <style>
    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1f4e79;
        margin-top: 30px;
        margin-bottom: 10px;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 5px solid #1f4e79;
    }
    .start-button {
        background-color: #1f4e79;
        color: white;
        padding: 12px 24px;
        font-size: 16px;
        border-radius: 8px;
        text-align: center;
        display: inline-block;
        margin-top: 20px;
        text-decoration: none;
    }
    .audience-tile {
        display: flex;
        flex-wrap: wrap;
        gap: 15px;
        margin-top: 10px;
    }
    .tile {
        background-color: #ffffff;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border-radius: 10px;
        text-align: center;
        padding: 20px;
        flex: 1 1 200px;
    }
    </style>
    """, unsafe_allow_html=True)

# Home page content aligned with SRS
def show_home_page():
    custom_css()

    st.title("📊 Smart Data Understanding & Synthetic Data Generation")
    st.write("An intelligent platform for AutoEDA and privacy-aware synthetic data pipelines.")

    # Key Features
    st.markdown('<div class="section-title">✨ Core Features</div>', unsafe_allow_html=True)
    st.markdown("""
    - 📁 Upload datasets (CSV/Excel) and auto-detect modality (Tabular/Text).
    - 🧪 AutoEDA summaries: Missing values, outliers, correlations, charts.
    - 🧠 BERT/LLM-driven natural language summaries.
    - ⚙️ Preprocessing: Encoding, Imputation, Scaling.
    - 🧬 Synthetic data generation using CTGAN, TVAE, GaussianCopula.
    - 📊 Side-by-side comparison: Real vs Synthetic (visual & statistical).
    - 📥 Download cleaned/synthetic datasets and reports (CSV, JSON, PDF).
    - 🔐 Privacy toggle (enable GDPR/HIPAA-style synthetic generation).
    """)

    # Target Audience
    st.markdown('<div class="section-title">👥 Who Can Use This?</div>', unsafe_allow_html=True)
    st.markdown('<div class="audience-tile">', unsafe_allow_html=True)
    users = ["Data Scientists", "ML Engineers", "Healthcare Professionals", "Finance Analysts", "Researchers", "Students"]
    icons = ["🔬", "💻", "🏥", "📈", "📚", "🎓"]
    for icon, role in zip(icons, users):
        st.markdown(f'<div class="tile"><div style="font-size:2rem;">{icon}</div><div>{role}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Usage Steps
    st.markdown('<div class="section-title">🚀 Getting Started</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        1. Upload a dataset using the sidebar.<br>
        2. Configure preprocessing options (missing values, encoding, scaling).<br>
        3. Let AutoEDA generate statistical & visual insights.<br>
        4. Optionally, generate synthetic data using deep generative models.<br>
        5. Download your results or compare real vs synthetic.
    </div>
    """, unsafe_allow_html=True)

    # Call to Action
    st.markdown('<div class="thank-you">Your AI journey begins here — get ready to analyze, simulate, and share data confidently!</div>', unsafe_allow_html=True)
    st.markdown('<a href="#file-uploader" class="start-button">📥 Upload Your Dataset</a>', unsafe_allow_html=True)

# Render the home page
if __name__ == "__main__":
    show_home_page()
