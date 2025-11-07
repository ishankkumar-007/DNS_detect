"""
Flask Web Application for DNS Spoofing Detection
LightGBM Models Only - Simplified & Clean
"""

from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = Path(__file__).parent / 'uploads'
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)

# Global model storage
loaded_models = {}


def discover_lightgbm_models():
    """Discover all available LightGBM models"""
    results_dir = Path(__file__).parent / 'results'
    models = []
    
    if not results_dir.exists():
        return models
    
    for result_dir in results_dir.iterdir():
        if not result_dir.is_dir():
            continue
        
        dir_name = result_dir.name.lower()
        
        # Only LightGBM models
        if 'lightgbm' in dir_name:
            model_file = result_dir / 'models' / 'lightgbm_detector.txt'
            if model_file.exists():
                models.append({
                    'name': result_dir.name,
                    'path': str(model_file),
                    'metadata_path': str(result_dir / 'models' / 'lightgbm_detector_metadata.json')
                })
    
    return models


@app.route('/')
def index():
    """Main page"""
    models = discover_lightgbm_models()
    return render_template('index_dashboard.html', models=models)


@app.route('/api/models')
def get_models():
    """API endpoint to get available models"""
    models = discover_lightgbm_models()
    return jsonify({'lightgbm': models})


@app.route('/api/load_model', methods=['POST'])
def load_model_endpoint():
    """Load a specific LightGBM model"""
    data = request.json
    model_path = data.get('model_path')
    model_name = data.get('model_name')
    
    if not all([model_path, model_name]):
        return jsonify({'error': 'Missing required parameters'}), 400
    
    try:
        from real_time_detection_lightgbm import RealTimeDNSDetector
        
        detector = RealTimeDNSDetector(model_path)
        loaded_models[model_name] = detector
        
        logger.info(f"Loaded LightGBM model: {model_name}")
        return jsonify({'success': True, 'message': f'Model {model_name} loaded successfully'})
    
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """Run prediction on uploaded data"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    model_name = request.form.get('model_name')
    max_rows = request.form.get('max_rows', type=int)
    
    if not model_name or model_name not in loaded_models:
        return jsonify({'error': 'Model not loaded'}), 400
    
    try:
        # Read CSV
        df = pd.read_csv(file, low_memory=False)
        
        if max_rows and max_rows > 0:
            df = df.head(max_rows)
        
        # Get model
        detector = loaded_models[model_name]
        
        # Run predictions
        start_time = datetime.now()
        flows_data = df.to_dict('records')
        results = detector.predict_batch(flows_data, measure_latency=True)
        prediction_time = (datetime.now() - start_time).total_seconds()
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Calculate statistics
        total_flows = len(results_df)
        malicious_count = results_df['is_malicious'].sum()
        detection_rate = (malicious_count / total_flows * 100) if total_flows > 0 else 0
        
        # Get prediction distribution
        prediction_dist = results_df['predicted_label'].value_counts().to_dict()
        
        # Combine with original data
        output_df = pd.concat([df, results_df], axis=1)
        
        # Store results temporarily
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_file = app.config['UPLOAD_FOLDER'] / f'predictions_{timestamp}.csv'
        output_df.to_csv(temp_file, index=False)
        
        # Get performance stats
        perf_stats = detector.get_performance_stats()
        
        response = {
            'success': True,
            'statistics': {
                'total_flows': int(total_flows),
                'malicious_detected': int(malicious_count),
                'detection_rate': float(detection_rate),
                'prediction_time': float(prediction_time),
                'avg_latency_ms': perf_stats.get('avg_latency_ms', prediction_time * 1000 / total_flows) if total_flows > 0 else 0
            },
            'prediction_distribution': prediction_dist,
            'download_url': f'/api/download/{temp_file.name}',
            'sample_predictions': results_df.head(10).to_dict('records'),  # For table display
            'all_predictions': results_df.to_dict('records')  # For charts (all flows)
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/download/<filename>')
def download(filename):
    """Download prediction results"""
    file_path = app.config['UPLOAD_FOLDER'] / filename
    
    if not file_path.exists():
        return jsonify({'error': 'File not found'}), 404
    
    return send_file(file_path, as_attachment=True)


@app.route('/api/model_info/<model_name>')
def model_info(model_name):
    """Get information about a loaded model"""
    if model_name not in loaded_models:
        return jsonify({'error': 'Model not loaded'}), 404
    
    detector = loaded_models[model_name]
    
    return jsonify({
        'model_name': model_name,
        'model_type': 'lightgbm',
        'feature_count': len(detector.feature_names) if detector.feature_names else 0,
        'label_mapping': detector.label_names
    })


if __name__ == '__main__':
    logger.info("Starting DNS Spoofing Detection Flask App (LightGBM Only)")
    logger.info("Discovering available models...")
    
    models = discover_lightgbm_models()
    logger.info(f"  Found {len(models)} LightGBM models")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
