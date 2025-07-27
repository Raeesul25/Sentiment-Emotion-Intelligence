import time
import random
import json
import requests
import streamlit as st
from datetime import datetime
from dashboard import RealTimeDashboard

# Define item mappings by product category
ITEM_IDS = {
    "Electronics": [f"elect_{str(i).zfill(4)}" for i in range(1, 11)],
    "Toys": [f"toys_{str(i).zfill(4)}" for i in range(1, 11)],
    "Amazon Alexa": [f"alexa_{str(i).zfill(4)}" for i in range(1, 11)],
    "Tops": [f"tops_{str(i).zfill(4)}" for i in range(1, 11)],
    "Musical Instruments": [f"music_{str(i).zfill(4)}" for i in range(1, 11)]
}

# Page Config
st.set_page_config(page_title="AI-Driven Sentiment & Emotion Intelligence", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
    }
    .title {
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 20px;
        color: #2C3E50;
    }
    .output-card {
        background-color: #F8F9F9;
        border-left: 6px solid #3D365C;
        padding: 15px;
        margin-top: 20px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .output-label {
        font-weight: bold;
        color: #3D365C;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.markdown('<div class="title">Navigation</div>', unsafe_allow_html=True)
page = st.sidebar.radio("Go to", ["Feedback Analysis", "Dashboard"])

# Feedback Analysis Page
if page == "Feedback Analysis":
    st.markdown('<div class="title">AI-Driven Sentiment & Emotion Intelligence for E-Commerce Feedback Optimization</div>', unsafe_allow_html=True)

    st.markdown("### Enter Your Product Review")
    review_input = st.text_area("Write your product feedback...", height=100)

    col1, col2 = st.columns(2)
    with col1:
        product_category = st.selectbox("Select Product Category", options=list(ITEM_IDS.keys()))
    with col2:
        if product_category:
            item_id = st.selectbox("Select Item ID", options=ITEM_IDS[product_category])
        else:
            item_id = st.selectbox("Select Item ID", options=[])

    if st.button("Analyze Feedback", use_container_width=True, type="primary"):
        if not review_input.strip():
            st.warning("Please enter a review before analyzing.")
        else:
            with st.spinner("Analyzing feedback..."):
                payload = {
                    "item_id": item_id,
                    "review": review_input,
                    "category": product_category
                    }
                
                try:
                    response = requests.post("http://localhost:5000/analyze", json=payload)
                    result = response.json()

                    st.success("✅ Analysis Complete")
                    st.markdown("---")
                    st.subheader("Analysis of Your Feedback..")
                    st.markdown(f"<span class='output-label'>📝 Review:</span> {result['review']}", unsafe_allow_html=True)
                    st.markdown(f"<span class='output-label'>📦 Product Category:</span> {result['product_category']}", unsafe_allow_html=True)
                    st.markdown(f"<span class='output-label'>🔢 Item ID:</span> {result['item_id']}", unsafe_allow_html=True)
                    st.markdown(f"<span class='output-label'>🕒 Entered Time:</span> {result['datetime']}", unsafe_allow_html=True)
                    st.markdown(f"<span class='output-label'>😊 Sentiment:</span> {result['sentiment']} (Confidence: {result['sentiment_confidence']:.2f})", unsafe_allow_html=True)
                    st.markdown(f"<span class='output-label'>🎭 Emotion:</span> {result['emotion']} (Confidence: {result['emotion_confidence']:.2f})", unsafe_allow_html=True)
                    st.markdown(f"<span class='output-label'>🔍 ABSA:</span> {result['aspect_sentiment']}", unsafe_allow_html=True)
    
                except requests.exceptions.ConnectionError:
                    st.error("❌ Could not connect to Flask API. Is it running?")
        
# Dashboard Page
elif page == "Dashboard":
    st.markdown('<div class="title">AI-Driven Sentiment & Emotion Intelligence for E-Commerce Feedback Optimization</div>', unsafe_allow_html=True)
    st.markdown("### 📊 Real-Time Dashboard")
    RealTimeDashboard = RealTimeDashboard()
    RealTimeDashboard.dashboard()    
