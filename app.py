import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import cv2
from PIL import Image
import os
import json
from datetime import datetime
import time

from model import ModelManager, preprocess_image

# Page configuration
st.set_page_config(
    page_title="TrustLens - AI Model Trust Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .trust-score-high {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
    }
    
    .trust-score-medium {
        background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%);
    }
    
    .trust-score-low {
        background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_manager' not in st.session_state:
    st.session_state.model_manager = ModelManager()
if 'inference_history' not in st.session_state:
    st.session_state.inference_history = []
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = {
        'pytorch': False,
        'onnx': False
    }

def load_models():
    """Load available models"""
    model_files = {
        'final_weights.pth': 'Final trained weights',
        'trustlens.pth': 'TrustLens model weights',
        'hybrid_model.onnx': 'ONNX optimized model'
    }
    
    loaded_models = []
    
    for file_name, description in model_files.items():
        file_path = os.path.join(".", file_name)
        if os.path.exists(file_path):
            if file_name.endswith('.pth'):
                success = st.session_state.model_manager.load_pytorch_model(file_path)
                if success:
                    st.session_state.models_loaded['pytorch'] = True
                    loaded_models.append(f"✅ {description}")
                else:
                    loaded_models.append(f"❌ {description} (Failed to load)")
            elif file_name.endswith('.onnx'):
                success = st.session_state.model_manager.load_onnx_model(file_path)
                if success:
                    st.session_state.models_loaded['onnx'] = True
                    loaded_models.append(f"✅ {description}")
                else:
                    loaded_models.append(f"❌ {description} (Failed to load)")
        else:
            loaded_models.append(f"⚠️ {description} (File not found)")
    
    return loaded_models

def create_trust_visualization(trust_score, confidence_scores):
    """Create trust score visualization"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Trust Score', 'Confidence Distribution'),
        specs=[[{"type": "indicator"}, {"type": "bar"}]]
    )
    
    # Trust score gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=trust_score * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Trust Score (%)"},
            delta={'reference': 80},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ),
        row=1, col=1
    )
    
    # Confidence distribution
    classes = ['REAL', 'FAKE']  # Class 0 = REAL, Class 1 = FAKE (matches model training)
    fig.add_trace(
        go.Bar(
            x=classes,
            y=confidence_scores,
            marker_color=['#FF6B6B', '#4ECDC4'],
            name="Confidence"
        ),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False)
    return fig

def create_feature_visualization(features):
    """Create feature visualization"""
    # Reduce dimensionality for visualization
    feature_sample = features[:100] if len(features) > 100 else features
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=feature_sample,
        mode='lines+markers',
        name='Feature Values',
        line=dict(color='#667eea', width=2),
        marker=dict(size=4)
    ))
    
    fig.update_layout(
        title="Feature Activation Pattern",
        xaxis_title="Feature Index",
        yaxis_title="Activation Value",
        height=300
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 TrustLens</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI Model Trust and Transparency Analysis Platform</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🚀 Model Management")
        
        # Model loading section
        if st.button("🔄 Load Models", type="primary"):
            with st.spinner("Loading models..."):
                loaded_models = load_models()
                st.success("Model loading completed!")
                for model_status in loaded_models:
                    st.write(model_status)
        
        # Model information
        st.header("📊 Model Information")
        model_info = st.session_state.model_manager.get_model_info()
        
        col1, col2 = st.columns(2)
        with col1:
            pytorch_status = "✅" if model_info['pytorch_loaded'] else "❌"
            st.metric("PyTorch Model", pytorch_status)
        with col2:
            onnx_status = "✅" if model_info['onnx_loaded'] else "❌"
            st.metric("ONNX Model", onnx_status)
        
        st.write(f"**Device:** {model_info['device']}")
        
        # Show ONNX input dimensions if available
        if model_info['onnx_loaded'] and 'onnx_inputs' in model_info:
            st.write("**ONNX Model Input Info:**")
            for inp in model_info['onnx_inputs']:
                st.write(f"- {inp['name']}: {inp['shape']}")
        
        # Settings
        st.header("⚙️ Settings")
        model_type = st.selectbox(
            "Select Model Type",
            ["PyTorch", "ONNX"],
            disabled=not any(st.session_state.models_loaded.values())
        )
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05
        )
        
        trust_threshold = st.slider(
            "Trust Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05
        )
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Inference", "📈 Analytics", "📊 Model Comparison", "📋 History"])
    
    with tab1:
        st.header("Model Inference")
        
        # Check if models are loaded
        if not any(st.session_state.models_loaded.values()):
            st.warning("⚠️ Please load models first using the sidebar.")
            st.stop()
        
        # Input methods
        input_method = st.radio(
            "Choose input method:",
            ["Upload Image", "Use Sample Data", "Random Input"]
        )
        
        input_data = None
        
        if input_method == "Upload Image":
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['png', 'jpg', 'jpeg', 'bmp', 'tiff']
            )
            
            if uploaded_file is not None:
                # Display uploaded image
                image = Image.open(uploaded_file)
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.image(image, caption="Uploaded Image", use_container_width=True)
                
                # Convert to numpy array
                image_array = np.array(image)
                if len(image_array.shape) == 3 and image_array.shape[2] == 4:
                    # Remove alpha channel if present
                    image_array = image_array[:, :, :3]
                
                # Use appropriate input size based on model type
                if model_type == "ONNX" and st.session_state.models_loaded['onnx']:
                    target_size = st.session_state.model_manager.get_onnx_input_size()
                else:
                    target_size = (32, 32)  # Both models trained on 32x32 images
                
                input_data = preprocess_image(image_array, target_size=target_size)
        
        elif input_method == "Use Sample Data":
            st.info("Using synthetic sample data for demonstration")
            
            # Use appropriate input size based on model type
            if model_type == "ONNX" and st.session_state.models_loaded['onnx']:
                target_size = st.session_state.model_manager.get_onnx_input_size()
            else:
                target_size = (32, 32)  # Both models trained on 32x32 images
            
            # Create sample data with correct size
            sample_data = np.random.rand(target_size[0], target_size[1], 3) * 255
            sample_data = sample_data.astype(np.uint8)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(sample_data, caption="Sample Data", use_container_width=True)
            
            input_data = preprocess_image(sample_data, target_size=target_size)
        
        elif input_method == "Random Input":
            st.info("Using random tensor input")
            
            # Use appropriate input size based on model type
            if model_type == "ONNX" and st.session_state.models_loaded['onnx']:
                target_size = st.session_state.model_manager.get_onnx_input_size()
            else:
                target_size = (32, 32)  # Both models trained on 32x32 images
            
            input_data = torch.randn(1, 3, target_size[0], target_size[1])
        
        # Run inference
        if input_data is not None and st.button("🚀 Run Inference", type="primary"):
            with st.spinner("Running inference..."):
                try:
                    if model_type == "PyTorch" and st.session_state.models_loaded['pytorch']:
                        results = st.session_state.model_manager.predict_pytorch(input_data)
                    elif model_type == "ONNX" and st.session_state.models_loaded['onnx']:
                        input_array = input_data.numpy()
                        results = st.session_state.model_manager.predict_onnx(input_array)
                    else:
                        st.error("Selected model type is not available!")
                        st.stop()
                    
                    # Display results
                    st.success("✅ Inference completed!")
                    
                    # Validate results
                    if not results or 'predicted_class' not in results or 'classification_probs' not in results:
                        st.error("❌ Invalid inference results: Missing required keys")
                        st.stop()
                    
                    # Create columns for results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        predicted_class_idx = results['predicted_class'][0]
                        predicted_class_name = 'REAL' if predicted_class_idx == 0 else 'FAKE'  # Class 0 = REAL, Class 1 = FAKE
                        confidence = np.max(results['classification_probs'][0])
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>Predicted Class</h3>
                            <h2>{predicted_class_name}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>Confidence</h3>
                            <h2>{confidence:.2%}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        if 'trust_score' in results:
                            trust_score = results['trust_score'][0][0]
                            trust_class = "trust-score-high" if trust_score > trust_threshold else "trust-score-medium" if trust_score > 0.5 else "trust-score-low"
                            
                            st.markdown(f"""
                            <div class="metric-card {trust_class}">
                                <h3>Trust Score</h3>
                                <h2>{trust_score:.2%}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>Trust Score</h3>
                                <h2>N/A</h2>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Detailed results
                    st.subheader("📊 Detailed Analysis")
                    
                    # Trust visualization
                    if 'trust_score' in results:
                        trust_fig = create_trust_visualization(
                            results['trust_score'][0][0],
                            results['classification_probs'][0]
                        )
                        st.plotly_chart(trust_fig, use_container_width=True)
                    
                    # Feature visualization
                    if 'features' in results:
                        feature_fig = create_feature_visualization(results['features'][0])
                        st.plotly_chart(feature_fig, use_container_width=True)
                    
                    # Classification probabilities
                    st.subheader("🎯 Classification Probabilities")
                    prob_df = pd.DataFrame({
                        'Class': ['REAL', 'FAKE'],  # Class 0 = REAL, Class 1 = FAKE
                        'Probability': results['classification_probs'][0]
                    })
                    
                    fig_bar = px.bar(
                        prob_df, 
                        x='Class', 
                        y='Probability',
                        color='Probability',
                        color_continuous_scale='viridis'
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
                    # Save to history
                    inference_record = {
                        'timestamp': datetime.now().isoformat(),
                        'model_type': model_type,
                        'predicted_class': int(predicted_class_idx),
                        'confidence': float(confidence),
                        'trust_score': float(results['trust_score'][0][0]) if 'trust_score' in results else None,
                        'probabilities': results['classification_probs'][0].tolist()
                    }
                    st.session_state.inference_history.append(inference_record)
                    
                except Exception as e:
                    st.error(f"❌ Inference failed: {str(e)}")
    
    with tab2:
        st.header("📈 Analytics Dashboard")
        
        if not st.session_state.inference_history:
            st.info("No inference history available. Run some inferences first!")
        else:
            # Convert history to DataFrame
            df = pd.DataFrame(st.session_state.inference_history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Inferences", len(df))
            with col2:
                avg_confidence = df['confidence'].mean()
                st.metric("Avg Confidence", f"{avg_confidence:.2%}")
            with col3:
                if 'trust_score' in df.columns and df['trust_score'].notna().any():
                    avg_trust = df['trust_score'].mean()
                    st.metric("Avg Trust Score", f"{avg_trust:.2%}")
                else:
                    st.metric("Avg Trust Score", "N/A")
            with col4:
                high_conf_count = (df['confidence'] > confidence_threshold).sum()
                st.metric("High Confidence", f"{high_conf_count}/{len(df)}")
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Confidence over time
                fig_time = px.line(
                    df, 
                    x='timestamp', 
                    y='confidence',
                    title='Confidence Over Time',
                    markers=True
                )
                st.plotly_chart(fig_time, use_container_width=True)
            
            with col2:
                # Class distribution
                class_counts = df['predicted_class'].value_counts()
                class_names = ['REAL' if i == 0 else 'FAKE' for i in class_counts.index]  # Class 0 = REAL, Class 1 = FAKE
                fig_pie = px.pie(
                    values=class_counts.values,
                    names=class_names,
                    title='Predicted Class Distribution'
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Trust score analysis (if available)
            if 'trust_score' in df.columns and df['trust_score'].notna().any():
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_trust_hist = px.histogram(
                        df, 
                        x='trust_score',
                        title='Trust Score Distribution',
                        nbins=20
                    )
                    st.plotly_chart(fig_trust_hist, use_container_width=True)
                
                with col2:
                    fig_scatter = px.scatter(
                        df,
                        x='confidence',
                        y='trust_score',
                        title='Confidence vs Trust Score',
                        trendline='ols'
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tab3:
        st.header("📊 Model Comparison")
        
        if len(st.session_state.inference_history) < 2:
            st.info("Need at least 2 inferences to compare models. Run more inferences!")
        else:
            df = pd.DataFrame(st.session_state.inference_history)
            
            # Group by model type
            model_comparison = df.groupby('model_type').agg({
                'confidence': ['mean', 'std', 'count'],
                'trust_score': ['mean', 'std'] if 'trust_score' in df.columns else ['count']
            }).round(4)
            
            st.subheader("Model Performance Comparison")
            st.dataframe(model_comparison)
            
            # Visualization
            if len(df['model_type'].unique()) > 1:
                fig_box = px.box(
                    df,
                    x='model_type',
                    y='confidence',
                    title='Confidence Distribution by Model Type'
                )
                st.plotly_chart(fig_box, use_container_width=True)
    
    with tab4:
        st.header("📋 Inference History")
        
        if not st.session_state.inference_history:
            st.info("No inference history available.")
        else:
            # Display history table
            df = pd.DataFrame(st.session_state.inference_history)
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            st.dataframe(df, use_container_width=True)
            
            # Export options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📥 Export CSV"):
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"trustlens_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("📥 Export JSON"):
                    json_str = json.dumps(st.session_state.inference_history, indent=2)
                    st.download_button(
                        label="Download JSON",
                        data=json_str,
                        file_name=f"trustlens_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            
            with col3:
                if st.button("🗑️ Clear History"):
                    st.session_state.inference_history = []
                    st.success("History cleared!")
                    st.rerun()

if __name__ == "__main__":
    main()