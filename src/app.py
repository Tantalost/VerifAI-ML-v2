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
import json
from datetime import datetime

# Import our custom modules
from models.explainability_module import YOLOv8Explainer
from utils.metrics_tracker import MetricsTracker
from utils.history_manager import HistoryManager

class MetricsTracker:
    """Track FRR and FAR metrics for model evaluation"""
    
    def __init__(self):
        self.metrics_file = "metrics_log.json"
        self.load_metrics()
    
    def load_metrics(self):
        """Load existing metrics from file"""
        try:
            with open(self.metrics_file, 'r') as f:
                self.metrics = json.load(f)
        except FileNotFoundError:
            self.metrics = {
                'true_positives': 0,  # AI correctly identified as AI
                'true_negatives': 0,  # Real correctly identified as Real  
                'false_positives': 0,  # Real incorrectly identified as AI
                'false_negatives': 0,  # AI incorrectly identified as Real
                'total_processed': 0,
                'session_history': []
            }
    
    def save_metrics(self):
        """Save metrics to file"""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def add_result(self, predicted_class, confidence, actual_class=None, filename="unknown"):
        """Add a prediction result for tracking"""
        self.metrics['total_processed'] += 1
        
        # If actual class is provided, calculate confusion matrix
        if actual_class:
            if actual_class.lower() == 'ai_generated' and predicted_class.lower() == 'ai_generated':
                self.metrics['true_positives'] += 1
            elif actual_class.lower() == 'real' and predicted_class.lower() == 'real':
                self.metrics['true_negatives'] += 1
            elif actual_class.lower() == 'real' and predicted_class.lower() == 'ai_generated':
                self.metrics['false_positives'] += 1
            elif actual_class.lower() == 'ai_generated' and predicted_class.lower() == 'real':
                self.metrics['false_negatives'] += 1
        
        # Add to session history
        self.metrics['session_history'].append({
            'timestamp': datetime.now().isoformat(),
            'filename': filename,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'actual_class': actual_class
        })
        
        self.save_metrics()
    
    def calculate_rates(self):
        """Calculate FRR and FAR"""
        tp = self.metrics['true_positives']
        tn = self.metrics['true_negatives']
        fp = self.metrics['false_positives']
        fn = self.metrics['false_negatives']
        
        # False Acceptance Rate: Real images incorrectly accepted as AI
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # False Rejection Rate: AI images incorrectly rejected as Real
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        return {
            'far': far,
            'frr': frr, 
            'accuracy': accuracy,
            'total_samples': tp + tn + fp + fn,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn
        }

# Page configuration
st.set_page_config(
    page_title="VerifAI-ML - AI Image Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Navigation
page = st.sidebar.selectbox("🧭 Navigate", ["🔍 Image Analysis", "📊 Metrics Dashboard"])

if page == "📊 Metrics Dashboard":
    # Import and run dashboard
    from dashboard import MetricsDashboard
    dashboard = MetricsDashboard()
    dashboard.run_dashboard()
    st.stop()

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
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .result-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #e1e5e9;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .confidence-high {
        color: #27ae60;
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
    .method-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e9ecef;
        margin: 0.5rem 0;
    }
    .heatmap-container {
        border: 2px solid #e9ecef;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

class VerifAIApp:
    def __init__(self):
        self.model_path = None
        self.explainer = None
        self.results_history = []
        self.metrics_tracker = MetricsTracker()  # Add metrics tracking
        self.history_manager = HistoryManager()  # Add history management
        
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
    
    def process_single_image(self, image_file, actual_class=None):
        """Process a single image and generate explanations for both methods"""
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(image_file.getvalue())
                tmp_path = tmp_file.name
            
            # Generate explanations for both methods
            eigencam_result = self.explainer.explain_image(tmp_path, 'eigencam')
            gradcam_result = self.explainer.explain_image(tmp_path, 'gradcam')
            
            # Combine results
            combined_result = {
                'predicted_class': eigencam_result['predicted_class'],
                'confidence': eigencam_result['confidence'],
                'eigencam': eigencam_result,
                'gradcam': gradcam_result
            }
            
            # Track metrics
            if combined_result:
                self.metrics_tracker.add_result(
                    predicted_class=combined_result['predicted_class'],
                    confidence=combined_result['confidence'],
                    actual_class=actual_class,
                    filename=getattr(image_file, 'name', 'uploaded_file')
                )
                
                # Save to history
                self.history_manager.add_analysis(
                    result=combined_result,
                    filename=getattr(image_file, 'name', 'uploaded_file'),
                    analysis_type="single",
                    method="both"
                )
            
            # Clean up
            os.unlink(tmp_path)
            
            return combined_result
            
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
                
                # Save to history
                self.history_manager.add_analysis(
                    result=result,
                    filename=image_file.name,
                    analysis_type="batch",
                    method=method
                )
        
        status_text.text("✅ Batch processing completed!")
        progress_bar.empty()
        
        return results
    
    def display_single_result(self, result, image_name="Uploaded Image"):
        """Display results for a single image with both EigenCAM and Grad-CAM"""
        if not result:
            return
        
        # Original Image
        st.subheader("📸 Original Image")
        st.image(result['eigencam']['original_image'], caption=image_name, use_container_width=True)
        
        # Bounding Box Analysis
        st.subheader("🎯 AI Region Detection")
        st.markdown("*YOLOv8-style bounding boxes highlight areas most likely to be AI-generated*")
        
        # Get detected regions from both methods
        eigencam_regions = result['eigencam'].get('detected_regions', [])
        gradcam_regions = result['gradcam'].get('detected_regions', [])
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 🌊 EigenCAM Regions")
            if eigencam_regions:
                st.image(result['eigencam']['image_with_boxes'], caption="EigenCAM Detected Regions", use_container_width=True)
                
                # Region details
                with st.expander("📋 Region Details"):
                    for i, region in enumerate(eigencam_regions[:5]):  # Show top 5 regions
                        bbox = region['bbox']
                        st.write(f"**Region {i+1}:** {region['label']}")
                        st.write(f"   - Confidence: {region['confidence']:.3f}")
                        st.write(f"   - Area: {region['area']:.0f} pixels")
                        st.write(f"   - Bounding Box: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
            else:
                st.info("No AI regions detected by EigenCAM")
        
        with col2:
            st.markdown("### 🎯 Grad-CAM Regions")
            if gradcam_regions:
                st.image(result['gradcam']['image_with_boxes'], caption="Grad-CAM Detected Regions", use_container_width=True)
                
                # Region details
                with st.expander("📋 Region Details"):
                    for i, region in enumerate(gradcam_regions[:5]):  # Show top 5 regions
                        bbox = region['bbox']
                        st.write(f"**Region {i+1}:** {region['label']}")
                        st.write(f"   - Confidence: {region['confidence']:.3f}")
                        st.write(f"   - Area: {region['area']:.0f} pixels")
                        st.write(f"   - Bounding Box: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
            else:
                st.info("No AI regions detected by Grad-CAM")
        
        # Prediction results
        st.subheader("🎯 Classification Results")
        
        col1, col2, col3 = st.columns([1.2, 1.2, 1])
        
        with col1:
            # Prediction card with better styling
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
            
            # Better prediction display
            st.markdown(f"""
            <div class="result-card">
                <h3 style="margin-bottom: 0.5rem;">Prediction</h3>
                <h2 style="color: {'#e74c3c' if prediction_class == 'ai_generated' else '#2ecc71'}; margin: 0.5rem 0;">
                    {prediction_class.upper().replace('_', ' ')}
                </h2>
                <p class="{confidence_class}">{confidence:.3f} ({confidence_text})</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Class probabilities with better chart
            st.markdown("**Class Probabilities**")
            if 'class_probabilities' in result['eigencam']:
                probs = result['eigencam']['class_probabilities']
                fig = go.Figure(data=[
                    go.Bar(name='Real', x=['Real'], y=[probs['real']], marker_color='#2ecc71'),
                    go.Bar(name='AI Generated', x=['AI Generated'], y=[probs['ai_generated']], marker_color='#e74c3c')
                ])
                fig.update_layout(
                    barmode='group', 
                    yaxis_title="Probability", 
                    height=250,
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            # Detailed metrics with better styling
            st.markdown("**Detailed Analysis**")
            if 'class_probabilities' in result['eigencam']:
                probs = result['eigencam']['class_probabilities']
                st.metric("Real Probability", f"{probs['real']:.3f}")
                st.metric("AI Generated Probability", f"{probs['ai_generated']:.3f}")
                margin = abs(probs['real'] - probs['ai_generated'])
                st.metric("Prediction Margin", f"{margin:.3f}")
        
        # Both explanations side by side
        st.subheader("🔍 Explainability Analysis")
        st.markdown("*Compare both methods to understand different aspects of the AI's decision*")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="method-card">', unsafe_allow_html=True)
            st.markdown("### 🌊 EigenCAM Analysis")
            st.markdown("*Shows overall patterns and features that influenced the decision*")
            st.image(result['eigencam']['overlay'], caption="EigenCAM Heatmap", use_container_width=True)
            
            # EigenCAM details
            with st.expander("📋 EigenCAM Details"):
                st.write("**Method**: Principal Component Analysis on feature maps")
                st.write("**Focus**: Overall decision patterns")
                st.write("**Strengths**: Good for showing broad evidence regions")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="method-card">', unsafe_allow_html=True)
            st.markdown("### 🎯 Grad-CAM Analysis")
            st.markdown("*Shows class-specific features that are most relevant to the prediction*")
            st.image(result['gradcam']['overlay'], caption="Grad-CAM Heatmap", use_container_width=True)
            
            # Grad-CAM details
            with st.expander("📋 Grad-CAM Details"):
                st.write("**Method**: Gradient-based class activation mapping")
                st.write("**Focus**: Class-specific features")
                st.write("**Strengths**: Precise localization of decision-critical areas")
            st.markdown('</div>', unsafe_allow_html=True)
    
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
    
    def display_metrics_dashboard(self):
        """Display FRR/FAR metrics in sidebar"""
        rates = self.metrics_tracker.calculate_rates()
        
        # Display key metrics
        st.metric("📊 Accuracy", f"{rates['accuracy']:.2%}")
        st.metric("🚫 FAR", f"{rates['far']:.2%}")
        st.metric("❌ FRR", f"{rates['frr']:.2%}")
        
        # Confusion matrix details
        with st.expander("📋 Confusion Matrix"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("✅ True Positives", rates['true_positives'])
                st.metric("✅ True Negatives", rates['true_negatives'])
            with col2:
                st.metric("🚫 False Positives", rates['false_positives'])
                st.metric("❌ False Negatives", rates['false_negatives'])
        
        # Total samples
        st.info(f"📈 Total Validated Samples: {rates['total_samples']}")
        
        # Clear metrics button
        if st.button("🗑️ Clear Metrics", key="clear_metrics"):
            self.metrics_tracker.metrics = {
                'true_positives': 0,
                'true_negatives': 0,
                'false_positives': 0,
                'false_negatives': 0,
                'total_processed': 0,
                'session_history': []
            }
            self.metrics_tracker.save_metrics()
            st.rerun()
    
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
                st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
                
                # Ground truth input for testing
                st.subheader("🧪 Test Mode (Optional)")
                actual_class = st.selectbox(
                    "Actual Class (for accuracy testing):",
                    ["Unknown", "Real", "AI-Generated"],
                    help="Select the ground truth to calculate FRR/FAR metrics"
                )
                
                # Process button
                if st.button("🔍 Analyze Image", type="primary"):
                    with st.spinner("Analyzing image with both EigenCAM and Grad-CAM..."):
                        actual = None if actual_class == "Unknown" else actual_class.lower().replace("-", "_")
                        result = self.process_single_image(uploaded_file, actual)
                        
                        if result:
                            # Store result in session state for verification
                            st.session_state.current_result = result
                            st.session_state.current_filename = uploaded_file.name
                            self.display_single_result(result, uploaded_file.name)
                            self.results_history.append(result)
                            
                            # Verification step
                            st.subheader("✅ Verification - Is this prediction correct?")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                if st.button("✅ Yes, Correct", type="primary", key="verify_correct"):
                                    # Save with ground truth
                                    actual_class = result['predicted_class'].lower().replace("-", "_")
                                    self.metrics_tracker.add_result(
                                        predicted_class=result['predicted_class'],
                                        confidence=result['confidence'],
                                        actual_class=actual_class,
                                        filename=uploaded_file.name
                                    )
                                    st.success("✅ Prediction validated and saved to dashboard!")
                                    st.session_state.current_result = None
                                    st.rerun()
                            
                            with col2:
                                if st.button("❌ No, Incorrect", type="secondary", key="verify_incorrect"):
                                    # Save with corrected ground truth
                                    actual_class = "ai_generated" if result['predicted_class'].lower() == "real" else "real"
                                    self.metrics_tracker.add_result(
                                        predicted_class=result['predicted_class'],
                                        confidence=result['confidence'],
                                        actual_class=actual_class,
                                        filename=uploaded_file.name
                                    )
                                    st.success("✅ Correction saved to dashboard!")
                                    st.session_state.current_result = None
                                    st.rerun()
                            
                            with col3:
                                if st.button("⏭️ Skip", type="secondary", key="verify_skip"):
                                    st.info("⏭️ Prediction not saved to metrics")
                                    st.session_state.current_result = None
                                    st.rerun()
        
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
