"""
Real-Time DNS Spoofing Detection Module
Optimized inference pipeline for live traffic analysis (<100ms latency)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import time
import logging
from typing import Dict, List, Optional, Union
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RealTimeDNSDetector:
    """
    Real-time DNS spoofing detector with optimized inference
    Designed for low-latency detection (<100ms)
    """
    
    def __init__(self, model_path: str):
        """
        Initialize real-time detector
        
        Args:
            model_path: Path to trained LightGBM model
        """
        self.model_path = Path(model_path)
        self.model = None
        self.feature_names = None
        self.label_names = None
        self.required_features = None
        self.inference_times = []
        
        self._load_model()
    
    def _load_model(self):
        """Load model and metadata"""
        logger.info(f"Loading model from {self.model_path}")
        
        # Load LightGBM model
        self.model = lgb.Booster(model_file=str(self.model_path))
        
        # Load metadata
        metadata_path = self.model_path.parent / f"{self.model_path.stem}_metadata.json"
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.feature_names = metadata.get('feature_names', [])
            self.label_names = metadata.get('label_names', {})
            
            logger.info(f"Model loaded successfully")
            logger.info(f"Expected features: {len(self.feature_names)}")
            logger.info(f"Label mapping: {self.label_names}")
        else:
            logger.warning(f"Metadata file not found at {metadata_path}")
            self.feature_names = []
    
    def preprocess_flow(self, flow_data: Dict) -> pd.DataFrame:
        """
        Preprocess single DNS flow for prediction
        
        Args:
            flow_data: Dictionary containing flow features
            
        Returns:
            Preprocessed DataFrame ready for prediction
        """
        # Convert to DataFrame
        df = pd.DataFrame([flow_data])
        
        # Handle missing features (fill with 0)
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0
        
        # Select only required features in correct order
        df = df[self.feature_names].copy()
        
        # Convert any non-numeric columns to numeric
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle any NaN values
        df = df.fillna(0)
        
        return df
    
    def predict_single(self, flow_data: Dict, measure_latency: bool = True) -> Dict:
        """
        Predict single DNS flow (optimized for speed)
        
        Args:
            flow_data: Dictionary containing flow features
            measure_latency: Whether to measure inference time
            
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time() if measure_latency else None
        
        # Preprocess
        X = self.preprocess_flow(flow_data)
        
        # Predict
        prediction_proba = self.model.predict(X, num_iteration=self.model.best_iteration)[0]
        
        # Get predicted class
        if len(prediction_proba.shape) == 0 or isinstance(prediction_proba, (int, float)):
            # Binary classification
            predicted_class = int(prediction_proba > 0.5)
            confidence = float(prediction_proba) if predicted_class == 1 else float(1 - prediction_proba)
        else:
            # Multi-class
            predicted_class = int(np.argmax(prediction_proba))
            confidence = float(prediction_proba[predicted_class])
        
        # Calculate latency
        latency_ms = None
        if measure_latency:
            latency_ms = (time.time() - start_time) * 1000
            self.inference_times.append(latency_ms)
        
        # Get label name
        label_name = self.label_names.get(str(predicted_class), f"Class_{predicted_class}")
        
        result = {
            'predicted_class': predicted_class,
            'predicted_label': label_name,
            'confidence': confidence,
            'is_malicious': predicted_class != 0,  # Assuming 0 is benign
            'latency_ms': latency_ms
        }
        
        return result
    
    def predict_batch(self, flows_data: List[Dict], 
                     measure_latency: bool = True) -> List[Dict]:
        """
        Predict batch of DNS flows
        
        Args:
            flows_data: List of flow dictionaries
            measure_latency: Whether to measure inference time
            
        Returns:
            List of prediction results
        """
        start_time = time.time() if measure_latency else None
        
        # Preprocess all flows
        X_list = [self.preprocess_flow(flow) for flow in flows_data]
        X_batch = pd.concat(X_list, ignore_index=True)
        
        # Batch prediction
        predictions_proba = self.model.predict(X_batch, num_iteration=self.model.best_iteration)
        
        results = []
        for i, proba in enumerate(predictions_proba):
            if len(proba.shape) == 0 or isinstance(proba, (int, float)):
                # Binary
                predicted_class = int(proba > 0.5)
                confidence = float(proba) if predicted_class == 1 else float(1 - proba)
            else:
                # Multi-class
                predicted_class = int(np.argmax(proba))
                confidence = float(proba[predicted_class])
            
            label_name = self.label_names.get(str(predicted_class), f"Class_{predicted_class}")
            
            result = {
                'predicted_class': predicted_class,
                'predicted_label': label_name,
                'confidence': confidence,
                'is_malicious': predicted_class != 0
            }
            results.append(result)
        
        # Calculate average latency per sample
        if measure_latency:
            total_time = (time.time() - start_time) * 1000
            avg_latency = total_time / len(flows_data)
            
            for result in results:
                result['latency_ms'] = avg_latency
        
        return results
    
    def get_performance_stats(self) -> Dict:
        """
        Get inference performance statistics
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.inference_times:
            return {'message': 'No inference times recorded'}
        
        stats = {
            'total_predictions': len(self.inference_times),
            'avg_latency_ms': np.mean(self.inference_times),
            'min_latency_ms': np.min(self.inference_times),
            'max_latency_ms': np.max(self.inference_times),
            'median_latency_ms': np.median(self.inference_times),
            'std_latency_ms': np.std(self.inference_times),
            'p95_latency_ms': np.percentile(self.inference_times, 95),
            'p99_latency_ms': np.percentile(self.inference_times, 99),
            'meets_sla': np.mean(self.inference_times) < 100  # <100ms requirement
        }
        
        return stats
    
    def reset_performance_stats(self):
        """Reset inference time tracking"""
        self.inference_times = []
    
    def stream_detection(self, flow_generator, callback=None, max_flows: Optional[int] = None):
        """
        Continuous stream detection (simulated real-time)
        
        Args:
            flow_generator: Generator yielding flow dictionaries
            callback: Optional callback function for each prediction
            max_flows: Maximum flows to process (for testing)
        """
        logger.info("Starting stream detection...")
        
        flows_processed = 0
        malicious_detected = 0
        
        try:
            for flow_data in flow_generator:
                # Predict
                result = self.predict_single(flow_data, measure_latency=True)
                
                # Count malicious detections
                if result['is_malicious']:
                    malicious_detected += 1
                    logger.warning(f"MALICIOUS TRAFFIC DETECTED: {result['predicted_label']} "
                                 f"(confidence: {result['confidence']:.4f})")
                
                # Callback
                if callback:
                    callback(flow_data, result)
                
                flows_processed += 1
                
                # Progress logging
                if flows_processed % 1000 == 0:
                    stats = self.get_performance_stats()
                    logger.info(f"Processed {flows_processed} flows | "
                              f"Malicious: {malicious_detected} | "
                              f"Avg latency: {stats['avg_latency_ms']:.2f}ms")
                
                # Stop condition
                if max_flows and flows_processed >= max_flows:
                    break
        
        except KeyboardInterrupt:
            logger.info("Stream detection interrupted by user")
        
        finally:
            # Final statistics
            logger.info("\n" + "="*50)
            logger.info("STREAM DETECTION SUMMARY")
            logger.info("="*50)
            logger.info(f"Total flows processed: {flows_processed}")
            logger.info(f"Malicious flows detected: {malicious_detected} "
                       f"({malicious_detected/flows_processed*100:.2f}%)")
            
            stats = self.get_performance_stats()
            logger.info(f"\nPerformance Statistics:")
            logger.info(f"  Average latency: {stats['avg_latency_ms']:.2f}ms")
            logger.info(f"  Min latency: {stats['min_latency_ms']:.2f}ms")
            logger.info(f"  Max latency: {stats['max_latency_ms']:.2f}ms")
            logger.info(f"  P95 latency: {stats['p95_latency_ms']:.2f}ms")
            logger.info(f"  P99 latency: {stats['p99_latency_ms']:.2f}ms")
            logger.info(f"  Meets <100ms SLA: {stats['meets_sla']}")
            logger.info("="*50)


class DNSFlowSimulator:
    """Simulate DNS flows for testing real-time detection"""
    
    def __init__(self, data_path: str, chunk_size: int = 1):
        """
        Initialize flow simulator
        
        Args:
            data_path: Path to CSV file with DNS flows
            chunk_size: Number of flows to yield at once
        """
        self.data_path = Path(data_path)
        self.chunk_size = chunk_size
    
    def generate_flows(self):
        """Generator yielding DNS flows"""
        logger.info(f"Reading flows from {self.data_path}")
        
        # Read CSV in chunks
        for chunk in pd.read_csv(self.data_path, chunksize=self.chunk_size, low_memory=False):
            for _, row in chunk.iterrows():
                yield row.to_dict()


def main():
    """Test real-time detection"""
    logger.info("Real-time detection module ready")


if __name__ == "__main__":
    main()
