import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class MetricsTracker:
    """Track FRR and FAR metrics for model evaluation"""
    
    def __init__(self):
        self.metrics_file = "metrics_log.json"
        self.load_metrics()
    
    def load_metrics(self):
        """Load existing metrics from file"""
        try:
            if os.path.exists(self.metrics_file):
                with open(self.metrics_file, 'r') as f:
                    self.metrics = json.load(f)
            else:
                self.metrics = {
                    'true_positives': 0,
                    'true_negatives': 0,
                    'false_positives': 0,
                    'false_negatives': 0,
                    'total_processed': 0,
                    'session_history': []
                }
        except Exception as e:
            logger.error(f"Error loading metrics: {str(e)}")
            self.metrics = {
                'true_positives': 0,
                'true_negatives': 0,
                'false_positives': 0,
                'false_negatives': 0,
                'total_processed': 0,
                'session_history': []
            }
    
    def save_metrics(self):
        """Save metrics to file"""
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")
    
    def add_result(self, predicted_class: str, confidence: float, 
                   actual_class: Optional[str] = None, filename: str = "unknown"):
        """Add a new prediction result"""
        self.metrics['total_processed'] += 1
        
        # Add to session history
        result_entry = {
            'timestamp': datetime.now().isoformat(),
            'filename': filename,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'actual_class': actual_class
        }
        
        self.metrics['session_history'].append(result_entry)
        
        # Update confusion matrix if ground truth provided
        if actual_class:
            pred = predicted_class.lower()
            actual = actual_class.lower()
            
            if pred == 'ai_generated' and actual == 'ai_generated':
                self.metrics['true_positives'] += 1
            elif pred == 'real' and actual == 'real':
                self.metrics['true_negatives'] += 1
            elif pred == 'ai_generated' and actual == 'real':
                self.metrics['false_positives'] += 1
            elif pred == 'real' and actual == 'ai_generated':
                self.metrics['false_negatives'] += 1
        
        self.save_metrics()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return self.metrics.copy()
    
    def calculate_rates(self) -> Dict[str, float]:
        """Calculate FRR, FAR, and accuracy rates"""
        tp = self.metrics['true_positives']
        tn = self.metrics['true_negatives']
        fp = self.metrics['false_positives']
        fn = self.metrics['false_negatives']
        
        total_positives = tp + fn
        total_negatives = tn + fp
        total_validated = tp + tn + fp + fn
        
        rates = {
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'total_samples': total_validated,
            'accuracy': 0.0,
            'far': 0.0,  # False Acceptance Rate
            'frr': 0.0   # False Rejection Rate
        }
        
        if total_validated > 0:
            rates['accuracy'] = (tp + tn) / total_validated
        
        if total_negatives > 0:
            rates['far'] = fp / total_negatives  # Real incorrectly classified as AI
        
        if total_positives > 0:
            rates['frr'] = fn / total_positives  # AI incorrectly classified as Real
        
        return rates
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.metrics = {
            'true_positives': 0,
            'true_negatives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'total_processed': 0,
            'session_history': []
        }
        self.save_metrics()
