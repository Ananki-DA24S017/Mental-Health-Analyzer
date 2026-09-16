import streamlit as st
import requests
import json
import pandas as pd
import time
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/frontend.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("frontend")

# Configuration
API_URL = "http://backend:8000"  # This refers to the service name in docker-compose

def get_prediction(text):
    """Send text to the API and get prediction"""
    try:
        start_time = time.time()
        response = requests.post(
            f"{API_URL}/predict",
            json={"text": text}
        )
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            logger.info(f"Prediction received in {response_time:.4f}s")
            return response.json()
        else:
            logger.error(f"Error from API: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error connecting to API: {str(e)}")
        return None

def get_health_status():
    """Check if the API is available"""
    try:
        response = requests.get(f"{API_URL}/health")
        return response.status_code == 200
    except:
        return False

def main():
    st.set_page_config(
        page_title="Mental Health Analyzer",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 Mental Health Analyzer")
    st.write("Express your thoughts and our AI will analyze your mental health state.")
    
    # Check API health
    api_available = get_health_status()
    if not api_available:
        st.error("⚠️ Backend API is not available. Please try again later.")
        st.stop()
    else:
        st.success("✅ Backend API is connected and ready")
    
    # User input
    user_input = st.text_area("Enter your thoughts here:", height=150)
    
    # Process button
    if st.button("Analyze"):
        if not user_input.strip():
            st.warning("Please enter some text to analyze")
        else:
            with st.spinner("Analyzing your thoughts..."):
                prediction = get_prediction(user_input)
                
            if prediction:
                # Display result with appropriate styling
                category = prediction["category"]
                confidence = prediction["confidence"]
                
                # Define colors for different categories
                category_colors = {
                    "Normal": "green",
                    "Depression": "blue",
                    "Anxiety": "orange",
                    "Bipolar": "purple",
                    "Personality disorder": "pink",
                    "Stress": "yellow",
                    "Suicidal": "red"
                } 
                
                color = category_colors.get(category, "gray")
                
                # Create three columns for better layout
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col2:
                    st.markdown(f"""
                    <div style="padding: 20px; border-radius: 10px; background-color: rgba(0, 0, 0, 0.05);">
                        <h3 style="text-align: center;">Analysis Result</h3>
                        <h1 style="text-align: center; color: {color};">{category}</h1>
                        <p style="text-align: center;">Confidence: {confidence:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Provide different responses based on the category
                    if category == "Normal":
                        st.info("Your mental state appears to be within normal range. Continue maintaining healthy habits!")
                    elif category == "Depression":
                        st.warning("Your text indicates signs of depression. Consider reaching out to a mental health professional.")
                    elif category == "Anxiety":
                        st.warning("Your text shows patterns associated with anxiety. Try relaxation techniques and consider professional support.")
                    elif category == "Bipolar":
                        st.warning("Your text has patterns that may be associated with bipolar tendencies. Professional evaluation is recommended.")
                    elif category == "Personality disorder":
                        st.warning("Your text shows signs that may be associated with a personality disorder. Professional evaluation is recommended.")
                    elif category == "Stress":
                        st.warning("Your text indicates high levels of stress. Consider stress management techniques and self-care.")
                    elif category == "Suicidal":
                        st.error("⚠️ Your text contains concerning elements. Please reach out to a crisis helpline or mental health professional immediately.")
                        st.markdown("**National Suicide Prevention Lifeline**: 988 or 1-800-273-8255")
            else:
                st.error("Failed to get prediction. Please try again.")
    
    # Add disclaimer at the bottom
    st.markdown("---")
    st.caption("""
    **Disclaimer**: This tool is for educational purposes only and is not a substitute for professional medical advice, 
    diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions 
    you may have regarding a medical condition.
    """)

if __name__ == "__main__":
    main()