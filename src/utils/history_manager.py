import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class HistoryManager:
    """Manages analysis history with persistence and search capabilities"""
    
    def __init__(self, history_file: str = "analysis_history.json"):
        self.history_file = history_file
        self.history = self._load_history()
    
    def _load_history(self) -> List[Dict[str, Any]]:
        """Load history from file"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"Error loading history: {str(e)}")
            return []
    
    def _save_history(self) -> bool:
        """Save history to file"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving history: {str(e)}")
            return False
    
    def add_analysis(self, result: Dict[str, Any], filename: str, 
                   analysis_type: str = "single", method: str = "both") -> bool:
        """Add a new analysis to history"""
        try:
            history_entry = {
                "timestamp": datetime.now().isoformat(),
                "filename": filename,
                "analysis_type": analysis_type,  # "single" or "batch"
                "method": method,  # "eigencam", "gradcam", or "both"
                "predicted_class": result.get('predicted_class', 'Unknown'),
                "confidence": result.get('confidence', 0.0),
                "class_probabilities": result.get('class_probabilities', {}),
                "detected_regions": {
                    "eigencam": result.get('eigencam', {}).get('detected_regions', []),
                    "gradcam": result.get('gradcam', {}).get('detected_regions', [])
                },
                "ground_truth": result.get('ground_truth', None),
                "verified": result.get('verified', False)
            }
            
            self.history.append(history_entry)
            return self._save_history()
            
        except Exception as e:
            logger.error(f"Error adding analysis to history: {str(e)}")
            return False
    
    def get_history(self, limit: Optional[int] = None, 
                   analysis_type: Optional[str] = None,
                   method: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get filtered history"""
        filtered_history = self.history.copy()
        
        # Filter by analysis type
        if analysis_type:
            filtered_history = [entry for entry in filtered_history 
                           if entry.get('analysis_type') == analysis_type]
        
        # Filter by method
        if method:
            filtered_history = [entry for entry in filtered_history 
                           if entry.get('method') == method]
        
        # Sort by timestamp (newest first)
        filtered_history.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        # Apply limit
        if limit:
            filtered_history = filtered_history[:limit]
        
        return filtered_history
    
    def search_history(self, query: str, search_fields: List[str] = None) -> List[Dict[str, Any]]:
        """Search history by filename or predicted class"""
        if search_fields is None:
            search_fields = ['filename', 'predicted_class', 'method']
        
        query_lower = query.lower()
        results = []
        
        for entry in self.history:
            for field in search_fields:
                field_value = str(entry.get(field, '')).lower()
                if query_lower in field_value:
                    results.append(entry)
                    break
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get analysis statistics"""
        if not self.history:
            return {
                "total_analyses": 0,
                "single_analyses": 0,
                "batch_analyses": 0,
                "unique_files": 0,
                "most_common_prediction": None,
                "average_confidence": 0.0
            }
        
        total_analyses = len(self.history)
        single_analyses = len([e for e in self.history if e.get('analysis_type') == 'single'])
        batch_analyses = len([e for e in self.history if e.get('analysis_type') == 'batch'])
        unique_files = len(set(e.get('filename', '') for e in self.history))
        
        # Most common prediction
        predictions = [e.get('predicted_class') for e in self.history]
        most_common_prediction = max(set(predictions), key=predictions.count) if predictions else None
        
        # Average confidence
        confidences = [e.get('confidence', 0.0) for e in self.history]
        average_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            "total_analyses": total_analyses,
            "single_analyses": single_analyses,
            "batch_analyses": batch_analyses,
            "unique_files": unique_files,
            "most_common_prediction": most_common_prediction,
            "average_confidence": average_confidence
        }
    
    def clear_history(self) -> bool:
        """Clear all history"""
        self.history = []
        return self._save_history()
    
    def export_history(self, export_path: str, format: str = "json") -> bool:
        """Export history to file"""
        try:
            if format.lower() == "json":
                with open(export_path, 'w') as f:
                    json.dump(self.history, f, indent=2)
            elif format.lower() == "csv":
                import pandas as pd
                df = pd.DataFrame(self.history)
                df.to_csv(export_path, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            return True
        except Exception as e:
            logger.error(f"Error exporting history: {str(e)}")
            return False
