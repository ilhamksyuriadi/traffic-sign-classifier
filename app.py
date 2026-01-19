"""
Traffic Sign Classification - Streamlit Web Interface
====================================================

A user-friendly web interface for classifying traffic signs using
the trained CNN model via Flask API.

Usage:
    streamlit run app.py
"""

import streamlit as st
import requests
from PIL import Image
import io
import pandas as pd
import plotly.express as px

# Configuration
API_URL = "http://localhost:5000"

# Page configuration
st.set_page_config(
    page_title="Traffic Sign Classifier",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        padding: 0.5rem;
        border-radius: 0.5rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


def check_api_health():
    """Check if the Flask API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def get_prediction(image_file):
    """Send image to API and get prediction"""
    try:
        files = {'file': image_file}
        response = requests.post(f"{API_URL}/predict", files=files, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None


def display_prediction_results(result):
    """Display prediction results in a beautiful format"""
    
    # Main prediction
    prediction = result['prediction']
    
    st.markdown("---")
    
    # Big result box
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎯 Prediction Result")
        st.markdown(f"<div class='prediction-box'>", unsafe_allow_html=True)
        st.markdown(f"**Traffic Sign:** {prediction['class_name']}")
        st.markdown(f"**Confidence:** {prediction['confidence']:.2%}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Confidence indicator
        confidence_color = "green" if prediction['confidence'] > 0.8 else "orange" if prediction['confidence'] > 0.5 else "red"
        st.markdown(f"**Confidence Level:** :{confidence_color}[{'●' * int(prediction['confidence'] * 10)}]")
    
    with col2:
        # Confidence gauge
        st.metric(
            label="Confidence Score",
            value=f"{prediction['confidence']:.1%}",
            delta=None
        )
    
    # Top 5 predictions
    st.markdown("---")
    st.markdown("### 📊 Top 5 Predictions")
    
    top_5 = result['top_5_predictions']
    
    # Create DataFrame for better display
    df = pd.DataFrame(top_5)
    df['confidence'] = df['confidence'].apply(lambda x: f"{x:.2%}")
    df.columns = ['Class ID', 'Traffic Sign', 'Confidence']
    
    # Display as table
    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True
    )
    
    # Visualization - Horizontal bar chart
    chart_df = pd.DataFrame(top_5)
    fig = px.bar(
        chart_df,
        x='confidence',
        y='class_name',
        orientation='h',
        title='Confidence Distribution - Top 5 Predictions',
        labels={'confidence': 'Confidence Score', 'class_name': 'Traffic Sign'},
        color='confidence',
        color_continuous_scale='RdYlGn',
        text=chart_df['confidence'].apply(lambda x: f"{x:.1%}")
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    fig.update_traces(textposition='outside')
    
    st.plotly_chart(fig, use_container_width=True)


def main():
    """Main Streamlit app"""
    
    # Header
    st.title("🚦 Traffic Sign Classifier")
    st.markdown("### Upload a traffic sign image to classify it using deep learning")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ℹ️ About")
        st.markdown("""
        This application uses a **Convolutional Neural Network (CNN)** with custom architecture
        trained on the German Traffic Sign Recognition Benchmark (GTSRB) dataset.
        
        **Features:**
        - 43 different traffic sign classes
        - Real-time classification
        - Confidence scores
        - Top 5 predictions

        """)
        
        st.markdown("---")
        
        # API status
        st.markdown("## 🔌 API Status")
        if check_api_health():
            st.success("✅ API is online")
        else:
            st.error("❌ API is offline")
            st.warning("Please start the Flask API:\n```python predict.py```")
        
        st.markdown("---")
        
        # Stats (optional)
        st.markdown("## 📈 Model Stats")
        st.metric("Total Classes", "43")
        st.metric("Input Size", "48x48")
        st.metric("Model Type", "CNN")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a traffic sign image...",
            type=['png', 'jpg', 'jpeg', 'ppm'],
            help="Upload an image of a traffic sign (PNG, JPG, or JPEG format)"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Traffic Sign', use_container_width=True)
            
            # Image info
            st.caption(f"📏 Image size: {image.size[0]} x {image.size[1]} pixels")
            st.caption(f"📁 File size: {uploaded_file.size / 1024:.2f} KB")
    
    with col2:
        st.markdown("### 🔍 Classification")
        
        if uploaded_file is not None:
            # Check API first
            if not check_api_health():
                st.error("⚠️ Flask API is not running!")
                st.info("Please start the API first:\n```bash\npython predict.py\n```")
                return
            
            # Classify button
            if st.button("🚀 Classify Traffic Sign", type="primary"):
                with st.spinner("🔄 Analyzing image..."):
                    # Reset file pointer
                    uploaded_file.seek(0)
                    
                    # Get prediction
                    result = get_prediction(uploaded_file)
                    
                    if result and result.get('success'):
                        st.success("✅ Classification complete!")
                        display_prediction_results(result)
                    else:
                        st.error("❌ Classification failed. Please try another image.")
        else:
            st.info("👈 Please upload an image to start classification")
            
            # Example instructions
            st.markdown("---")
            st.markdown("### 💡 Tips")
            st.markdown("""
            - Use clear, well-lit images
            - Ensure the traffic sign is centered
            - Avoid blurry or obscured images
            - Supported formats: PNG, JPG, JPEG
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Built with ❤️ using Streamlit | Powered by TensorFlow & Flask"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()