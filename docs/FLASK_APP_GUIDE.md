# DNS Spoofing Detection - Flask Web Application

A user-friendly web interface for DNS spoofing detection using multiple machine learning models.

## Features

- 🔵 **Multiple Model Support**: LightGBM, Random Forest, Hybrid, BiLSTM, Isolation Forest, OCSVM
- 📊 **Real-time Predictions**: Upload CSV files and get instant predictions
- 📈 **Visual Analytics**: View detection rates, prediction distributions, and performance metrics
- 💾 **Export Results**: Download prediction results as CSV
- 🎯 **Easy Model Management**: Select and load models through intuitive UI

## Installation

1. Install Flask dependencies:
```bash
pip install -r requirements_flask.txt
```

Or install individually:
```bash
pip install flask==3.0.0 werkzeug==3.0.1 flask-cors==4.0.0
```

## Usage

1. **Start the Flask server:**
```bash
python app.py
```

2. **Open your browser:**
Navigate to `http://localhost:5000`

3. **Use the interface:**
   - **Step 1**: Select a trained model from the available categories
   - **Step 2**: Click "Load Selected Model" to load it into memory
   - **Step 3**: Upload a CSV file with DNS traffic data
   - **Step 4**: Click "Run Prediction" to analyze the data
   - **Step 5**: View results and download the full predictions

## API Endpoints

### GET `/`
Main web interface

### GET `/api/models`
Returns list of all available models
```json
{
  "lightgbm": [...],
  "random_forest": [...],
  "hybrid": [...],
  ...
}
```

### POST `/api/load_model`
Load a specific model into memory
```json
{
  "model_type": "lightgbm",
  "model_path": "path/to/model.txt",
  "model_name": "model_name"
}
```

### POST `/api/predict`
Run predictions on uploaded CSV file
- **Form Data:**
  - `file`: CSV file with DNS flows
  - `model_name`: Name of loaded model
  - `max_rows`: (optional) Limit number of rows to process

**Response:**
```json
{
  "success": true,
  "statistics": {
    "total_flows": 1000,
    "malicious_detected": 50,
    "detection_rate": 5.0,
    "avg_latency_ms": 2.5
  },
  "prediction_distribution": {...},
  "sample_predictions": [...],
  "download_url": "/api/download/predictions_xxx.csv"
}
```

### POST `/api/predict_json`
Run predictions on JSON data (for API integration)
```json
{
  "model_name": "lightgbm_complete_20251106_082854",
  "flows": [
    {
      "feature1": value1,
      "feature2": value2,
      ...
    }
  ]
}
```

### GET `/api/download/<filename>`
Download prediction results CSV

### GET `/api/model_info/<model_name>`
Get detailed information about a loaded model

## File Structure

```
├── app.py                          # Flask application
├── templates/
│   └── index.html                  # Web interface
├── uploads/                        # Temporary file storage
├── results/                        # Contains trained models
│   ├── lightgbm_*/
│   ├── hybrid_*/
│   └── ...
└── src/
    └── real_time_detection_lightgbm.py
```

## Model Directory Structure

Models are automatically discovered from the `results/` directory. Each model should follow this structure:

```
results/
└── <model_name>/
    └── models/
        ├── <model_type>_detector.{txt|pkl|h5}
        └── <model_type>_detector_metadata.json
```

## Supported Model Types

| Model Type | File Extension | Description |
|------------|---------------|-------------|
| LightGBM | `.txt` | Gradient boosting model |
| Random Forest | `.pkl` | Random forest classifier |
| Hybrid | `.pkl` | Combined supervised/unsupervised |
| BiLSTM | `.h5` | Bidirectional LSTM |
| Isolation Forest | `.pkl` | Anomaly detection |
| OCSVM | `.pkl` | One-class SVM |

## Configuration

### Max Upload Size
Default: 100MB. Change in `app.py`:
```python
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
```

### Port and Host
Default: `http://0.0.0.0:5000`. Change in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

## Example Usage with API

### Using curl:
```bash
# Load model
curl -X POST http://localhost:5000/api/load_model \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "lightgbm",
    "model_path": "./results/lightgbm_complete/models/lightgbm_detector.txt",
    "model_name": "lightgbm_complete"
  }'

# Run prediction
curl -X POST http://localhost:5000/api/predict \
  -F "file=@test_data.csv" \
  -F "model_name=lightgbm_complete"
```

### Using Python:
```python
import requests

# Load model
response = requests.post('http://localhost:5000/api/load_model', json={
    'model_type': 'lightgbm',
    'model_path': './results/lightgbm_complete/models/lightgbm_detector.txt',
    'model_name': 'lightgbm_complete'
})

# Run prediction
with open('test_data.csv', 'rb') as f:
    response = requests.post('http://localhost:5000/api/predict', 
        files={'file': f},
        data={'model_name': 'lightgbm_complete'}
    )
    results = response.json()
    print(results['statistics'])
```

## Troubleshooting

### Models not appearing
- Check that models exist in `results/` directory
- Verify model files follow naming convention: `<model_type>_detector.{ext}`

### Prediction errors
- Ensure CSV has same features as training data
- Check model metadata JSON file exists
- Verify model is loaded before prediction

### Performance issues
- Use `max_rows` parameter to limit data size
- Process large files in batches
- Consider increasing server timeout

## Production Deployment

For production use:

1. **Use Gunicorn or uWSGI:**
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

2. **Set up HTTPS** with nginx reverse proxy

3. **Disable debug mode:**
```python
app.run(debug=False, host='0.0.0.0', port=5000)
```

4. **Add authentication** for sensitive deployments

## License

Part of the DNS Spoofing Detection Project
