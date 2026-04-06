import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime
import os

class MetricsDashboard:
    """Dashboard for displaying FRR/FAR metrics and analytics"""
    
    def __init__(self, metrics_file="metrics_log.json"):
        self.metrics_file = metrics_file
    
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

if __name__ == "__main__":
    dashboard = MetricsDashboard()
    dashboard.run_dashboard()
