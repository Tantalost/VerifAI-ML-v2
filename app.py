import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import zipfile
import tempfile
import os
from pathlib import Path
import io
import base64
from PIL import Image
import torch
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from explainability_module import YOLOv8Explainer

# Page configuration
st.set_page_config(
    page_title="VerifAI-ML - AI Image Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .result-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #e1e5e9;
        margin-bottom: 1rem;
    }
    .confidence-high {
        color: #2ecc71;
        font-weight: bold;
    }
    .confidence-medium {
        color: #f39c12;
        font-weight: bold;
    }
    .confidence-low {
        color: #e74c3c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class VerifAIApp:
    def __init__(self):
        self.model_path = None
        self.explainer = None
        self.results_history = []
        
    def load_model(self):
        """Load the trained YOLOv8 model"""
        try:
            # Try to find the best model
            model_paths = [
                "runs/classify/yolov8_classification_*/weights/best.pt",
                "best.pt",
                "yolov8m-cls.pt"
            ]
            
            for path_pattern in model_paths:
                if "*" in path_pattern:
                    # Find the latest run
                    from glob import glob
                    matches = glob(path_pattern)
                    if matches:
                        self.model_path = matches[-1]  # Get the latest
                        break
                else:
                    if os.path.exists(path_pattern):
                        self.model_path = path_pattern
                        break
            
            if self.model_path is None:
                st.error("❌ No trained model found! Please run training_pipeline.py first.")
                return False
            
            self.explainer = YOLOv8Explainer(self.model_path)
            st.success(f"✅ Model loaded successfully: {self.model_path}")
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            return False
    
    def process_single_image(self, image_file, method='eigencam'):
        """Process a single image and generate explanation"""
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(image_file.getvalue())
                tmp_path = tmp_file.name
            
            # Generate explanation
            result = self.explainer.explain_image(tmp_path, method)
            
            # Clean up
            os.unlink(tmp_path)
            
            return result
            
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
            return None
    
    def process_batch_images(self, uploaded_files, method='eigencam'):
        """Process multiple images in batch"""
        results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, image_file in enumerate(uploaded_files):
            status_text.text(f"Processing image {i+1}/{len(uploaded_files)}...")
            progress_bar.progress((i + 1) / len(uploaded_files))
            
            result = self.process_single_image(image_file, method)
            if result:
                result['filename'] = image_file.name
                results.append(result)
        
        status_text.text("✅ Batch processing completed!")
        progress_bar.empty()
        
        return results
    
    def display_single_result(self, result, image_name="Uploaded Image"):
        """Display results for a single image"""
        if not result:
            return
        
        # Create columns for layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📸 Original Image")
            st.image(result['original_image'], caption=image_name, use_column_width=True)
        
        with col2:
            st.subheader("🔍 Evidence Heatmap")
            st.image(result['overlay'], caption="AI Detection Evidence", use_column_width=True)
        
        # Prediction results
        st.subheader("🎯 Classification Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Prediction card
            prediction_class = result['predicted_class']
            confidence = result['confidence']
            
            # Color code based on confidence
            if confidence >= 0.8:
                confidence_class = "confidence-high"
                confidence_text = "High Confidence"
            elif confidence >= 0.6:
                confidence_class = "confidence-medium"
                confidence_text = "Medium Confidence"
            else:
                confidence_class = "confidence-low"
                confidence_text = "Low Confidence"
            
            st.markdown(f"""
            <div class="result-card">
                <h3>Prediction</h3>
                <h2 style="color: {'#e74c3c' if prediction_class == 'ai_generated' else '#2ecc71'};">
                    {prediction_class.upper().replace('_', ' ')}
                </h2>
                <p class="{confidence_class}">
                    {confidence:.3f} ({confidence_text})
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Confidence bar chart
            st.markdown("**Confidence Scores**")
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Real', 'AI Generated'],
                y=[
                    result['class_probabilities']['real'],
                    result['class_probabilities']['ai_generated']
                ],
                marker_color=['#2ecc71', '#e74c3c']
            ))
            
            fig.update_layout(
                title="Class Probabilities",
                yaxis_title="Confidence",
                yaxis=dict(range=[0, 1]),
                showlegend=False,
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            # Detailed metrics
            st.markdown("**Detailed Analysis**")
            
            real_prob = result['class_probabilities']['real']
            ai_prob = result['class_probabilities']['ai_generated']
            
            st.metric("Real Probability", f"{real_prob:.3f}")
            st.metric("AI Generated Probability", f"{ai_prob:.3f}")
            st.metric("Prediction Margin", f"{abs(real_prob - ai_prob):.3f}")
        
        # Heatmap visualization
        st.subheader("🌡️ Detailed Heatmap Analysis")
        
        # Create matplotlib figure for better control
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(result['original_image'])
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Raw heatmap
        axes[1].imshow(result['heatmap'], cmap='jet')
        axes[1].set_title('Raw Heatmap')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(result['overlay'])
        axes[2].set_title(f'Prediction: {prediction_class}')
        axes[2].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    def display_batch_results(self, results):
        """Display results for batch processing"""
        if not results:
            st.warning("No results to display.")
            return
        
        st.subheader("📊 Batch Analysis Results")
        
        # Summary statistics
        total_images = len(results)
        ai_count = sum(1 for r in results if r['predicted_class'] == 'ai_generated')
        real_count = total_images - ai_count
        avg_confidence = np.mean([r['confidence'] for r in results])
        
        # Display summary cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Images", total_images)
        
        with col2:
            st.metric("AI Generated", ai_count)
        
        with col3:
            st.metric("Real Images", real_count)
        
        with col4:
            st.metric("Avg Confidence", f"{avg_confidence:.3f}")
        
        # Distribution chart
        st.subheader("📈 Classification Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            fig = go.Figure(data=[go.Pie(
                labels=['AI Generated', 'Real'],
                values=[ai_count, real_count],
                hole=0.3,
                marker_colors=['#e74c3c', '#2ecc71']
            )])
            
            fig.update_layout(
                title="Image Classification Distribution",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Confidence distribution
            confidences = [r['confidence'] for r in results]
            predictions = [r['predicted_class'] for r in results]
            
            fig = go.Figure()
            
            # Add histograms for each class
            ai_confidences = [r['confidence'] for r in results if r['predicted_class'] == 'ai_generated']
            real_confidences = [r['confidence'] for r in results if r['predicted_class'] == 'real']
            
            if ai_confidences:
                fig.add_trace(go.Histogram(
                    x=ai_confidences,
                    name='AI Generated',
                    marker_color='#e74c3c',
                    opacity=0.7,
                    nbinsx=20
                ))
            
            if real_confidences:
                fig.add_trace(go.Histogram(
                    x=real_confidences,
                    name='Real',
                    marker_color='#2ecc71',
                    opacity=0.7,
                    nbinsx=20
                ))
            
            fig.update_layout(
                title="Confidence Score Distribution",
                xaxis_title="Confidence Score",
                yaxis_title="Count",
                barmode='overlay',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Individual results
        st.subheader("🔍 Individual Image Results")
        
        # Create expandable sections for each result
        for i, result in enumerate(results):
            with st.expander(f"📷 {result['filename']} - {result['predicted_class'].upper()} ({result['confidence']:.3f})"):
                self.display_single_result(result, result['filename'])
        
        # Download results
        st.subheader("💾 Export Results")
        
        # Create CSV of results
        results_df = pd.DataFrame([
            {
                'Filename': r['filename'],
                'Prediction': r['predicted_class'],
                'Confidence': r['confidence'],
                'Real_Probability': r['class_probabilities']['real'],
                'AI_Probability': r['class_probabilities']['ai_generated']
            }
            for r in results
        ])
        
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results (CSV)",
            data=csv,
            file_name="verifai_results.csv",
            mime="text/csv"
        )
    
    def run(self):
        """Main application runner"""
        # Header
        st.markdown('<h1 class="main-header">🔍 VerifAI-ML</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Generated Image Detection with Explainable AI</p>', unsafe_allow_html=True)
        
        # Load model
        if not self.explainer:
            if not self.load_model():
                st.stop()
        
        # Sidebar
        st.sidebar.title("⚙️ Settings")
        
        # Upload options
        upload_type = st.sidebar.radio(
            "Upload Type:",
            ["Single Image", "Batch Upload"]
        )
        
        # Explainability method
        method = st.sidebar.selectbox(
            "Explainability Method:",
            ["eigencam", "gradcam"],
            help="EigenCAM shows overall patterns, Grad-CAM shows class-specific features"
        )
        
        # Main content
        if upload_type == "Single Image":
            st.header("📤 Single Image Analysis")
            
            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                help="Upload an image to analyze whether it's AI-generated or real"
            )
            
            if uploaded_file is not None:
                # Display uploaded image
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
                
                # Process button
                if st.button("🔍 Analyze Image", type="primary"):
                    with st.spinner("Analyzing image..."):
                        result = self.process_single_image(uploaded_file, method)
                        
                        if result:
                            self.display_single_result(result, uploaded_file.name)
                            self.results_history.append(result)
        
        else:  # Batch Upload
            st.header("📦 Batch Image Analysis")
            
            uploaded_files = st.file_uploader(
                "Choose images...",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                accept_multiple_files=True,
                help="Upload multiple images for batch analysis"
            )
            
            # Alternative: Zip file upload
            zip_file = st.file_uploader(
                "Or upload a ZIP file containing images...",
                type=['zip'],
                help="Upload a ZIP file containing multiple images"
            )
            
            if zip_file is not None:
                # Extract zip file
                with tempfile.TemporaryDirectory() as temp_dir:
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    
                    # Find image files
                    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
                    image_files = []
                    
                    for root, dirs, files in os.walk(temp_dir):
                        for file in files:
                            if any(file.lower().endswith(ext) for ext in image_extensions):
                                image_files.append(os.path.join(root, file))
                    
                    # Convert to file-like objects
                    uploaded_files = []
                    for img_path in image_files:
                        with open(img_path, 'rb') as f:
                            uploaded_files.append(io.BytesIO(f.read()))
            
            if uploaded_files and len(uploaded_files) > 0:
                st.info(f"📁 {len(uploaded_files)} images ready for analysis")
                
                # Process button
                if st.button("🔍 Analyze Batch", type="primary"):
                    with st.spinner(f"Processing {len(uploaded_files)} images..."):
                        results = self.process_batch_images(uploaded_files, method)
                        
                        if results:
                            self.display_batch_results(results)
                            self.results_history.extend(results)
        
        # Footer
        st.markdown("---")
        st.markdown(
            "<p style='text-align: center; color: #666;'>"
            "VerifAI-ML - Advanced AI Image Detection System | "
            "Powered by YOLOv8 & Explainable AI"
            "</p>",
            unsafe_allow_html=True
        )

# Run the app
if __name__ == "__main__":
    app = VerifAIApp()
    app.run()
