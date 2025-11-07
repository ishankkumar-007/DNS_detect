"""
Flask Web Application for DNS Spoofing Detection
Supports multiple model types: LightGBM, Random Forest, Hybrid, BiLSTM, Isolation Forest, OCSVM
"""

from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
import io
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


def discover_models():
    """Discover all available trained models"""
    results_dir = Path(__file__).parent / 'results'
    models = {
        'lightgbm': [],
        'random_forest': [],
        'hybrid': [],
        'bilstm': [],
        'isolation_forest': [],
        'ocsvm': []
    }
    
    if not results_dir.exists():
        return models
    
    for result_dir in results_dir.iterdir():
        if not result_dir.is_dir():
            continue
        
        dir_name = result_dir.name.lower()
        
        # LightGBM models
        if 'lightgbm' in dir_name:
            model_file = result_dir / 'models' / 'lightgbm_detector.txt'
            if model_file.exists():
                models['lightgbm'].append({
                    'name': result_dir.name,
                    'path': str(model_file),
                    'metadata_path': str(result_dir / 'models' / 'lightgbm_detector_metadata.json')
                })
        
        # # Random Forest models
        # elif 'rf_' in dir_name or 'random_forest' in dir_name or 'rf20_' in dir_name:
        #     model_file = result_dir / 'models' / 'random_forest_detector.pkl'
        #     if model_file.exists():
        #         models['random_forest'].append({
        #             'name': result_dir.name,
        #             'path': str(model_file),
        #             'metadata_path': str(result_dir / 'models' / 'random_forest_detector_metadata.json')
        #         })
        
        # Hybrid models
        elif 'hybrid' in dir_name:
            model_file = result_dir / 'models' / 'hybrid_detector.pkl'
            if model_file.exists():
                models['hybrid'].append({
                    'name': result_dir.name,
                    'path': str(model_file),
                    'metadata_path': str(result_dir / 'models' / 'hybrid_detector_metadata.json')
                })
        
        # # BiLSTM models
        # elif 'bilstm' in dir_name:
        #     model_file = result_dir / 'models' / 'bilstm_detector.h5'
        #     if model_file.exists():
        #         models['bilstm'].append({
        #             'name': result_dir.name,
        #             'path': str(model_file),
        #             'metadata_path': str(result_dir / 'models' / 'bilstm_detector_metadata.json')
        #         })
        
        # # Isolation Forest models
        # elif 'iforest' in dir_name:
        #     model_file = result_dir / 'models' / 'isolation_forest_detector.pkl'
        #     if model_file.exists():
        #         models['isolation_forest'].append({
        #             'name': result_dir.name,
        #             'path': str(model_file),
        #             'metadata_path': str(result_dir / 'models' / 'isolation_forest_detector_metadata.json')
        #         })
        
        # # OCSVM models
        # elif 'ocsvm' in dir_name:
        #     model_file = result_dir / 'models' / 'ocsvm_detector.pkl'
        #     if model_file.exists():
        #         models['ocsvm'].append({
        #             'name': result_dir.name,
        #             'path': str(model_file),
        #             'metadata_path': str(result_dir / 'models' / 'ocsvm_detector_metadata.json')
        #         })
    
    return models


def load_model(model_type, model_path):
    """Load a model based on type"""
    try:
        if model_type == 'lightgbm':
            from real_time_detection_lightgbm import RealTimeDNSDetector
            detector = RealTimeDNSDetector(model_path)
            return detector
        
        elif model_type == 'random_forest':
            import pickle
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Load metadata if available
            metadata_path = Path(model_path).parent / f"{Path(model_path).stem}_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    model_data['label_names'] = metadata.get('label_names', {})
            
            return model_data
        
        elif model_type == 'hybrid':
            # Load hybrid model pickle file
            import pickle
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Load metadata if available
            metadata_path = Path(model_path).parent / f"{Path(model_path).stem}_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    if 'metadata' not in model_data:
                        model_data['metadata'] = {}
                    model_data['metadata'].update(metadata)
            
            return model_data
        
        elif model_type == 'bilstm':
            from tensorflow import keras
            model = keras.models.load_model(model_path)
            return model
        
        elif model_type == 'isolation_forest':
            import pickle
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Load metadata if available
            metadata_path = Path(model_path).parent / f"{Path(model_path).stem}_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    model_data['label_names'] = metadata.get('label_names', {})
            
            return model_data
        
        elif model_type == 'ocsvm':
            import pickle
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Load metadata if available
            metadata_path = Path(model_path).parent / f"{Path(model_path).stem}_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    model_data['label_names'] = metadata.get('label_names', {})
            
            return model_data
        
    except Exception as e:
        logger.error(f"Error loading {model_type} model: {e}")
        raise


def predict_with_model(model_type, model, data_df):
    """Run predictions with the specified model type"""
    try:
        if model_type == 'lightgbm':
            # Use the RealTimeDNSDetector's predict_batch method
            flows_data = data_df.to_dict('records')
            results = model.predict_batch(flows_data, measure_latency=True)
            return pd.DataFrame(results)
        
        elif model_type == 'hybrid':
            # Hybrid models have supervised_model, unsupervised_model, etc.
            # Need to reconstruct the detector or use the models directly
            from models.hybrid_model import HybridDetector
            
            # Reconstruct the HybridDetector from loaded data
            detector = HybridDetector(
                supervised_model=model['supervised_model'],
                unsupervised_model=model['unsupervised_model'],
                fusion_strategy=model.get('fusion_strategy', 'weighted'),
                config=model.get('config', {})
            )
            
            # Get feature names from supervised model (RandomForestDetector has feature_names)
            supervised_model = model['supervised_model']
            
            # Get the actual trained sklearn model to find number of features it expects
            if hasattr(supervised_model, 'model'):
                # It's a wrapper like RandomForestDetector
                sklearn_model = supervised_model.model
                n_features_expected = sklearn_model.n_features_in_
                
                # Get feature names from wrapper
                if hasattr(supervised_model, 'feature_names') and supervised_model.feature_names:
                    all_feature_names = supervised_model.feature_names
                    # Only use the first n_features_expected features
                    feature_names = all_feature_names[:n_features_expected]
                    logger.info(f"Model expects {n_features_expected} features, selected from {len(all_feature_names)} available")
                else:
                    feature_names = [f'feature_{i}' for i in range(n_features_expected)]
            elif hasattr(supervised_model, 'feature_names'):
                feature_names = supervised_model.feature_names
            else:
                # Fallback to metadata
                metadata = model.get('metadata', {})
                feature_names = metadata.get('feature_names', [])
            
            # Get label names from metadata
            metadata = model.get('metadata', {})
            label_names = metadata.get('label_names', {})
            
            # Prepare features using the CORRECT feature subset
            if feature_names:
                # Handle missing features - add them as zeros
                missing_features = [f for f in feature_names if f not in data_df.columns]
                if missing_features:
                    logger.warning(f"Missing {len(missing_features)} features in input data, filling with zeros: {missing_features[:5]}...")
                    for feat in missing_features:
                        data_df[feat] = 0
                
                # Select only the features that were used during training
                X = data_df[feature_names].copy()
                
                # Convert any non-numeric columns to numeric
                for col in X.columns:
                    if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                        # Try to convert to numeric, fill non-convertible with 0
                        X[col] = pd.to_numeric(X[col], errors='coerce')
                
                # Fill NaN values with 0
                X = X.fillna(0)
                
                logger.info(f"Using {len(feature_names)} features for hybrid model prediction")
            else:
                # Use all numeric columns if no feature names specified
                X = data_df.select_dtypes(include=[np.number]).fillna(0)
                logger.warning(f"No feature names found for hybrid model, using all {X.shape[1]} numeric columns")
            
            # Predict
            predictions = detector.predict(X)
            
            # Get probabilities if available
            try:
                probabilities = detector.predict_proba(X)
                confidences = np.max(probabilities, axis=1)
            except:
                confidences = np.ones(len(predictions))
            
            # Build results
            results = []
            for pred, conf in zip(predictions, confidences):
                label_name = label_names.get(str(int(pred)), f"Class_{int(pred)}")
                results.append({
                    'predicted_class': int(pred),
                    'predicted_label': label_name,
                    'confidence': float(conf),
                    'is_malicious': int(pred) != 0
                })
            
            return pd.DataFrame(results)
        
        elif model_type in ['random_forest', 'isolation_forest', 'ocsvm']:
            # These models are stored as dictionaries with model and feature info
            clf = model['model']
            feature_names = model.get('feature_names', [])
            label_names = model.get('label_names', {})
            
            # Prepare features - handle missing features
            if feature_names:
                missing_features = [f for f in feature_names if f not in data_df.columns]
                if missing_features:
                    logger.warning(f"Missing {len(missing_features)} features in input data for {model_type}, filling with zeros")
                    for feat in missing_features:
                        data_df[feat] = 0
                
                X = data_df[feature_names].copy()
                
                # Convert any non-numeric columns to numeric
                for col in X.columns:
                    if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                        X[col] = pd.to_numeric(X[col], errors='coerce')
                
                X = X.fillna(0)
            else:
                X = data_df.select_dtypes(include=[np.number]).fillna(0)
            
            # Predict
            predictions = clf.predict(X)
            
            # Get probabilities if available
            if hasattr(clf, 'predict_proba'):
                probabilities = clf.predict_proba(X)
                confidences = np.max(probabilities, axis=1)
            else:
                confidences = np.ones(len(predictions))
            
            # Build results
            results = []
            for pred, conf in zip(predictions, confidences):
                label_name = label_names.get(str(int(pred)), f"Class_{int(pred)}")
                results.append({
                    'predicted_class': int(pred),
                    'predicted_label': label_name,
                    'confidence': float(conf),
                    'is_malicious': int(pred) != 0
                })
            
            return pd.DataFrame(results)
        
        elif model_type == 'bilstm':
            # BiLSTM requires special preprocessing
            # This is a simplified version - you may need to adjust based on your actual implementation
            logger.warning("BiLSTM prediction not fully implemented in web app")
            return pd.DataFrame([{
                'predicted_class': 0,
                'predicted_label': 'Not Implemented',
                'confidence': 0.0,
                'is_malicious': False
            }] * len(data_df))
        
    except Exception as e:
        logger.error(f"Error during prediction with {model_type}: {e}")
        raise


@app.route('/')
def index():
    """Main page"""
    models = discover_models()
    return render_template('index.html', models=models)


@app.route('/api/models')
def get_models():
    """API endpoint to get available models"""
    models = discover_models()
    return jsonify(models)


@app.route('/api/load_model', methods=['POST'])
def load_model_endpoint():
    """Load a specific model"""
    data = request.json
    model_type = data.get('model_type')
    model_path = data.get('model_path')
    model_name = data.get('model_name')
    
    if not all([model_type, model_path, model_name]):
        return jsonify({'error': 'Missing required parameters'}), 400
    
    try:
        model = load_model(model_type, model_path)
        loaded_models[model_name] = {
            'type': model_type,
            'model': model,
            'path': model_path
        }
        
        logger.info(f"Loaded {model_type} model: {model_name}")
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
        model_info = loaded_models[model_name]
        model_type = model_info['type']
        model = model_info['model']
        
        # Run predictions
        start_time = datetime.now()
        results_df = predict_with_model(model_type, model, df)
        prediction_time = (datetime.now() - start_time).total_seconds()
        
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
        
        response = {
            'success': True,
            'statistics': {
                'total_flows': int(total_flows),
                'malicious_detected': int(malicious_count),
                'detection_rate': float(detection_rate),
                'prediction_time': float(prediction_time),
                'avg_latency_ms': float(prediction_time * 1000 / total_flows) if total_flows > 0 else 0
            },
            'prediction_distribution': prediction_dist,
            'download_url': f'/api/download/{temp_file.name}',
            'sample_predictions': results_df.head(10).to_dict('records')
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


@app.route('/api/predict_json', methods=['POST'])
def predict_json():
    """Run prediction on JSON data (for single flow or small batches)"""
    data = request.json
    model_name = data.get('model_name')
    flows = data.get('flows', [])
    
    if not model_name or model_name not in loaded_models:
        return jsonify({'error': 'Model not loaded'}), 400
    
    if not flows:
        return jsonify({'error': 'No flows provided'}), 400
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(flows)
        
        # Get model
        model_info = loaded_models[model_name]
        model_type = model_info['type']
        model = model_info['model']
        
        # Run predictions
        results_df = predict_with_model(model_type, model, df)
        
        return jsonify({
            'success': True,
            'predictions': results_df.to_dict('records')
        })
    
    except Exception as e:
        logger.error(f"JSON prediction failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/model_info/<model_name>')
def model_info(model_name):
    """Get information about a loaded model"""
    if model_name not in loaded_models:
        return jsonify({'error': 'Model not loaded'}), 404
    
    model_info = loaded_models[model_name]
    
    # Try to load metadata
    metadata_path = Path(model_info['path']).parent / f"{Path(model_info['path']).stem}_metadata.json"
    metadata = {}
    
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return jsonify({
        'model_name': model_name,
        'model_type': model_info['type'],
        'model_path': model_info['path'],
        'metadata': metadata
    })


if __name__ == '__main__':
    logger.info("Starting DNS Spoofing Detection Flask App")
    logger.info("Discovering available models...")
    
    models = discover_models()
    for model_type, model_list in models.items():
        logger.info(f"  {model_type}: {len(model_list)} models found")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
