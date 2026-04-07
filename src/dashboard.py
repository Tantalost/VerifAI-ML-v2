import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.history_manager import HistoryManager
from datetime import datetime
import os
import json

class MetricsDashboard:
    """Dashboard for displaying FRR/FAR metrics and analytics"""
    
    def __init__(self, metrics_file="metrics_log.json"):
        self.metrics_file = metrics_file
        self.history_manager = HistoryManager()
    
    def load_metrics(self):
        """Load metrics from file"""
        try:
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                'true_positives': 0,
                'true_negatives': 0,
                'false_positives': 0,
                'false_negatives': 0,
                'total_processed': 0,
                'session_history': []
            }
    
    def calculate_rates(self, metrics):
        """Calculate FRR, FAR, and accuracy"""
        tp = metrics['true_positives']
        tn = metrics['true_negatives']
        fp = metrics['false_positives']
        fn = metrics['false_negatives']
        
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0
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
    
    def run_dashboard(self):
        """Main dashboard page"""
        st.set_page_config(
            page_title="VerifAI-ML - Metrics Dashboard",
            page_icon="📊",
            layout="wide"
        )
        
        st.title("📊 VerifAI-ML Metrics Dashboard")
        st.markdown("---")
        
        # Load metrics
        metrics = self.load_metrics()
        rates = self.calculate_rates(metrics)
        
        # Key Metrics Overview
        st.header("🎯 Performance Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "📊 Accuracy", 
                f"{rates['accuracy']:.2%}",
                delta=None,
                help="Overall prediction accuracy"
            )
        
        with col2:
            st.metric(
                "🚫 False Acceptance Rate", 
                f"{rates['far']:.2%}",
                delta=None,
                help="Real images incorrectly classified as AI"
            )
        
        with col3:
            st.metric(
                "❌ False Rejection Rate", 
                f"{rates['frr']:.2%}",
                delta=None,
                help="AI images incorrectly classified as Real"
            )
        
        with col4:
            st.metric(
                "📈 Total Validated", 
                rates['total_samples'],
                delta=None,
                help="Number of images with ground truth validation"
            )
        
        # Confusion Matrix
        st.header("📋 Confusion Matrix")
        
        # Create confusion matrix data
        confusion_data = [
            ["Predicted AI", "Predicted Real"],
            [rates['true_positives'], rates['false_negatives']],
            [rates['false_positives'], rates['true_negatives']]
        ]
        
        fig = go.Figure(data=go.Heatmap(
            z=[[rates['true_positives'], rates['false_negatives']],
               [rates['false_positives'], rates['true_negatives']]],
            x=['Predicted AI', 'Predicted Real'],
            y=['Actual AI', 'Actual Real'],
            colorscale='Blues',
            text=[[f"TP: {rates['true_positives']}", f"FN: {rates['false_negatives']}"],
                  [f"FP: {rates['false_positives']}", f"TN: {rates['true_negatives']}"]],
            texttemplate="%{text}",
            textfont={"size": 14},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance Trends
        st.header("📈 Performance Trends")
        
        if metrics['session_history']:
            # Convert history to DataFrame
            df = pd.DataFrame(metrics['session_history'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.date
            
            # Daily accuracy trend
            daily_stats = df.groupby('date').agg({
                'predicted_class': 'count',
                'actual_class': lambda x: x.notna().sum()
            }).rename(columns={'predicted_class': 'total_processed', 'actual_class': 'validated'})
            
            if not daily_stats.empty:
                fig_trend = px.line(
                    daily_stats, 
                    x=daily_stats.index, 
                    y='validated',
                    title="Daily Validated Predictions Over Time",
                    markers=True
                )
                fig_trend.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Number of Validated Predictions"
                )
                st.plotly_chart(fig_trend, use_container_width=True)
        
        # Recent Predictions
        st.header("🕐 Recent Validated Predictions")
        
        if metrics['session_history']:
            # Get recent validated predictions
            validated_predictions = [
                pred for pred in metrics['session_history'] 
                if pred['actual_class'] is not None
            ]
            
            if validated_predictions:
                recent_df = pd.DataFrame(validated_predictions[-10:])  # Last 10
                recent_df['timestamp'] = pd.to_datetime(recent_df['timestamp'])
                recent_df['correct'] = (
                    (recent_df['predicted_class'].str.lower() == 'ai_generated') & 
                    (recent_df['actual_class'].str.lower() == 'ai_generated')
                ) | (
                    (recent_df['predicted_class'].str.lower() == 'real') & 
                    (recent_df['actual_class'].str.lower() == 'real')
                )
                
                recent_df['result'] = recent_df['correct'].apply(lambda x: "✅ Correct" if x else "❌ Incorrect")
                
                st.dataframe(
                    recent_df[['timestamp', 'filename', 'predicted_class', 'actual_class', 'confidence', 'result']],
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No validated predictions yet. Start analyzing images and providing ground truth!")
        else:
            st.info("No predictions yet. Start analyzing images!")
        
        # Export/Reset Options
        st.header("⚙️ Management Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Export Metrics (JSON)", type="secondary"):
                st.download_button(
                    label="Download metrics_log.json",
                    data=json.dumps(metrics, indent=2),
                    file_name=f"verifai_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📊 Export CSV", type="secondary"):
                if metrics['session_history']:
                    df = pd.DataFrame(metrics['session_history'])
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download predictions.csv",
                        data=csv,
                        file_name=f"verifai_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No data to export")
        
        with col3:
            if st.button("🗑️ Reset All Metrics", type="secondary"):
                if st.session_state.get('confirm_reset', False):
                    # Reset metrics
                    reset_metrics = {
                        'true_positives': 0,
                        'true_negatives': 0,
                        'false_positives': 0,
                        'false_negatives': 0,
                        'total_processed': 0,
                        'session_history': []
                    }
                    
                    with open(self.metrics_file, 'w') as f:
                        json.dump(reset_metrics, f, indent=2)
                    
                    st.success("✅ Metrics reset successfully!")
                    st.rerun()
                else:
                    st.session_state.confirm_reset = True
                    st.warning("⚠️ Click again to confirm reset")

    def display_history_section(self):
        """Display comprehensive analysis history"""
        st.header("📚 Analysis History")
        st.markdown("*Browse and search your previous image analyses*")
        
        # History controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Search functionality
            st.markdown("**🔍 Search History**")
            search_query = st.text_input(
                "Search by filename or prediction...",
                placeholder="Enter search term...",
                help="Search through your analysis history"
            )
        
        with col2:
            # Filter by analysis type
            st.markdown("**📋 Filter by Type**")
            analysis_type = st.selectbox(
                "Analysis Type:",
                ["All", "Single", "Batch"],
                help="Filter by analysis type"
            )
        
        with col3:
            # Filter by method
            st.markdown("**🔬 Filter by Method**")
            method_filter = st.selectbox(
                "Method:",
                ["All", "eigencam", "gradcam", "both"],
                help="Filter by explainability method"
            )
        
        # Apply filters
        history = self.history_manager.get_history()
        
        if search_query:
            history = self.history_manager.search_history(search_query)
        
        if analysis_type != "All":
            history = [entry for entry in history 
                     if entry.get('analysis_type') == analysis_type.lower()]
        
        if method_filter != "All":
            history = [entry for entry in history 
                     if entry.get('method') == method_filter]
        
        # Number of results to show
        col1, col2 = st.columns(2)
        
        with col1:
            limit = st.slider(
                "Show last:",
                min_value=1,
                max_value=min(100, len(history)),
                value=10,
                help="Number of recent analyses to display"
            )
        
        with col2:
            st.markdown(f"**Found {len(history)} analyses**")
            if st.button("🔄 Refresh History"):
                st.rerun()
        
        # Display history
        if history:
            # Show limited results
            display_history = history[:limit]
            
            for i, entry in enumerate(display_history):
                with st.expander(f"📸 {entry.get('filename', 'Unknown')} - {entry.get('timestamp', '')[:10]}"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Basic info
                        st.markdown("**📊 Analysis Details**")
                        st.write(f"**Type:** {entry.get('analysis_type', 'Unknown').title()}")
                        st.write(f"**Method:** {entry.get('method', 'Unknown').title()}")
                        st.write(f"**Prediction:** {entry.get('predicted_class', 'Unknown').title()}")
                        st.write(f"**Confidence:** {entry.get('confidence', 0):.3f}")
                        
                        # Ground truth if available
                        if entry.get('ground_truth'):
                            st.write(f"**Ground Truth:** {entry.get('ground_truth').title()}")
                            if entry.get('verified'):
                                st.success("✅ Verified")
                            else:
                                st.info("⏳ Pending verification")
                        
                        # Detected regions summary
                        eigencam_regions = entry.get('detected_regions', {}).get('eigencam', [])
                        gradcam_regions = entry.get('detected_regions', {}).get('gradcam', [])
                        
                        st.write(f"**EigenCAM Regions:** {len(eigencam_regions)} detected")
                        st.write(f"**Grad-CAM Regions:** {len(gradcam_regions)} detected")
                    
                    with col2:
                        # Quick stats
                        st.markdown("**⚡ Quick Stats**")
                        
                        # Confidence indicator
                        confidence = entry.get('confidence', 0)
                        if confidence >= 0.8:
                            conf_color = "🟢"
                            conf_text = "High"
                        elif confidence >= 0.6:
                            conf_color = "🟡"
                            conf_text = "Medium"
                        else:
                            conf_color = "🔴"
                            conf_text = "Low"
                        
                        st.write(f"**Confidence:** {conf_color} {conf_text}")
                        
                        # Prediction indicator
                        pred_class = entry.get('predicted_class', '').lower()
                        if pred_class == 'ai_generated':
                            pred_color = "🤖"
                        else:
                            pred_color = "👤"
                        
                        st.write(f"**Prediction:** {pred_color} {pred_class.replace('_', ' ').title()}")
                        
                        # Time info
                        timestamp = entry.get('timestamp', '')
                        if timestamp:
                            try:
                                dt = datetime.fromisoformat(timestamp)
                                st.write(f"**Analyzed:** {dt.strftime('%b %d, %Y %H:%M')}")
                            except:
                                st.write(f"**Time:** {timestamp[:19]}")
        else:
            st.info("🔍 No analysis history found. Start analyzing images to build your history!")
        
        # History statistics
        if history:
            st.subheader("📈 History Statistics")
            stats = self.history_manager.get_statistics()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📸 Total Analyses", stats['total_analyses'])
            
            with col2:
                st.metric("🔬 Single Analyses", stats['single_analyses'])
            
            with col3:
                st.metric("📦 Batch Analyses", stats['batch_analyses'])
            
            with col4:
                st.metric("📁 Unique Files", stats['unique_files'])
            
            # Additional stats
            col1, col2 = st.columns(2)
            
            with col1:
                if stats['most_common_prediction']:
                    st.metric("🎯 Most Common", stats['most_common_prediction'].replace('_', ' ').title())
            
            with col2:
                st.metric("📊 Avg Confidence", f"{stats['average_confidence']:.3f}")
            
            # Export history section
            st.subheader("💾 Export History")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📥 Export History (JSON)", type="secondary"):
                    history_data = self.history_manager.get_history(limit=None)
                    st.download_button(
                        label="Download analysis_history.json",
                        data=json.dumps(history_data, indent=2),
                        file_name=f"analysis_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            
            with col2:
                if st.button("🗑️ Clear History", type="secondary"):
                    if st.session_state.get('confirm_clear_history', False):
                        if self.history_manager.clear_history():
                            st.success("✅ History cleared successfully!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to clear history")
                    else:
                        st.session_state.confirm_clear_history = True
                        st.warning("⚠️ Click again to confirm clear history")

    def run_dashboard(self, show_history=False):
        """Main dashboard page"""
        st.set_page_config(
            page_title="VerifAI-ML - Metrics Dashboard",
            page_icon="📊",
            layout="wide"
        )
        
        st.title("📊 VerifAI-ML Metrics Dashboard")
        st.markdown("---")
        
        # Load metrics
        metrics = self.load_metrics()
        rates = self.calculate_rates(metrics)
        
        # Key Metrics Overview
        st.header("🎯 Performance Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "📊 Accuracy", 
                f"{rates['accuracy']:.2%}",
                delta=None,
                help="Overall prediction accuracy"
            )
        
        with col2:
            st.metric(
                "🚫 False Acceptance Rate", 
                f"{rates['far']:.2%}",
                delta=None,
                help="Real images incorrectly classified as AI"
            )
        
        with col3:
            st.metric(
                "❌ False Rejection Rate", 
                f"{rates['frr']:.2%}",
                delta=None,
                help="AI images incorrectly classified as Real"
            )
        
        with col4:
            st.metric(
                "📈 Total Validated", 
                rates['total_samples'],
                delta=None,
                help="Number of images with ground truth validation"
            )
        
        # Confusion Matrix
        st.header("📋 Confusion Matrix")
        
        # Create confusion matrix data
        confusion_data = [
            ["Predicted AI", "Predicted Real"],
            [rates['true_positives'], rates['false_negatives']],
            [rates['false_positives'], rates['true_negatives']]
        ]
        
        fig = go.Figure(data=go.Heatmap(
            z=[[rates['true_positives'], rates['false_negatives']],
               [rates['false_positives'], rates['true_negatives']]],
            x=['Predicted AI', 'Predicted Real'],
            y=['Actual AI', 'Actual Real'],
            colorscale='Blues',
            text=[[f"TP: {rates['true_positives']}", f"FN: {rates['false_negatives']}"],
                  [f"FP: {rates['false_positives']}", f"TN: {rates['true_negatives']}"]],
            texttemplate="%{text}",
            textfont={"size": 14},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance Trends
        st.header("📈 Performance Trends")
        
        if metrics['session_history']:
            # Convert history to DataFrame
            df = pd.DataFrame(metrics['session_history'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.date
            
            # Daily accuracy trend
            daily_stats = df.groupby('date').agg({
                'predicted_class': 'count',
                'actual_class': lambda x: x.notna().sum()
            }).rename(columns={'predicted_class': 'total_processed', 'actual_class': 'validated'})
            
            if not daily_stats.empty:
                fig_trend = px.line(
                    daily_stats, 
                    x=daily_stats.index, 
                    y='validated',
                    title="Daily Validated Predictions Over Time",
                    markers=True
                )
                fig_trend.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Number of Validated Predictions"
                )
                st.plotly_chart(fig_trend, use_container_width=True)
        
        # Recent Predictions
        st.header("🕐 Recent Validated Predictions")
        
        if metrics['session_history']:
            # Get recent validated predictions
            validated_predictions = [
                pred for pred in metrics['session_history'] 
                if pred['actual_class'] is not None
            ]
            
            if validated_predictions:
                recent_df = pd.DataFrame(validated_predictions[-10:])  # Last 10
                recent_df['timestamp'] = pd.to_datetime(recent_df['timestamp'])
                recent_df['correct'] = (
                    (recent_df['predicted_class'].str.lower() == 'ai_generated') & 
                    (recent_df['actual_class'].str.lower() == 'ai_generated')
                ) | (
                    (recent_df['predicted_class'].str.lower() == 'real') & 
                    (recent_df['actual_class'].str.lower() == 'real')
                )
                
                recent_df['result'] = recent_df['correct'].apply(lambda x: "✅ Correct" if x else "❌ Incorrect")
                
                st.dataframe(
                    recent_df[['timestamp', 'filename', 'predicted_class', 'actual_class', 'confidence', 'result']],
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No validated predictions yet. Start analyzing images and providing ground truth!")
        else:
            st.info("No predictions yet. Start analyzing images!")
        
        # Export/Reset Options
        st.header("⚙️ Management Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Export Metrics (JSON)", type="secondary"):
                st.download_button(
                    label="Download metrics_log.json",
                    data=json.dumps(metrics, indent=2),
                    file_name=f"verifai_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📊 Export CSV", type="secondary"):
                if metrics['session_history']:
                    df = pd.DataFrame(metrics['session_history'])
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download predictions.csv",
                        data=csv,
                        file_name=f"verifai_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No data to export")
        
        with col3:
            if st.button("🗑️ Reset All Metrics", type="secondary"):
                if st.session_state.get('confirm_reset', False):
                    # Reset metrics
                    reset_metrics = {
                        'true_positives': 0,
                        'true_negatives': 0,
                        'false_positives': 0,
                        'false_negatives': 0,
                        'total_processed': 0,
                        'session_history': []
                    }
                    
                    with open(self.metrics_file, 'w') as f:
                        json.dump(reset_metrics, f, indent=2)
                    
                    st.success("✅ Metrics reset successfully!")
                    st.rerun()
                else:
                    st.session_state.confirm_reset = True
                    st.warning("⚠️ Click again to confirm reset")
        
        if show_history:
            self.display_history_section()

if __name__ == "__main__":
    dashboard = MetricsDashboard()
    dashboard.run_dashboard(show_history=True)
