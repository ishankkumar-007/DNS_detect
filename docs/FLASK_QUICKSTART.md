# Flask Web Application - Quick Start Guide

## 🎉 Successfully Created!

A full-featured Flask web application for DNS spoofing detection has been created with support for multiple model types.

## 📁 Files Created

1. **`app.py`** - Main Flask application (backend)
2. **`templates/index.html`** - Web interface (frontend)
3. **`test_flask_app_setup.py`** - Setup verification script
4. **`start_flask_app.py`** - Quick start script
5. **`requirements_flask.txt`** - Flask dependencies
6. **`FLASK_APP_GUIDE.md`** - Comprehensive documentation

## ✅ Setup Verified

All tests passed! The app is ready to run with:
- ✓ Flask 3.1.2 installed
- ✓ 19 trained models discovered
- ✓ All directories configured

## 🚀 How to Start

### Option 1: Direct Run
```powershell
python app.py
```

### Option 2: Using Start Script
```powershell
python start_flask_app.py
```

Then open your browser to: **http://localhost:5000**

## 🎯 Features

### Supported Models (Auto-Discovered)
- **LightGBM** (3 models)
- **Random Forest** (4 models)
- **Hybrid** (4 models)
- **BiLSTM** (3 models)
- **Isolation Forest** (3 models)
- **OCSVM** (2 models)

### Web Interface Features
1. **Model Selection**: Browse and select from all available trained models
2. **Easy Loading**: One-click model loading into memory
3. **File Upload**: Upload CSV files with DNS traffic data
4. **Batch Processing**: Optional row limiting for large files
5. **Real-time Results**: View statistics, detection rates, and performance metrics
6. **Visual Analytics**: 
   - Detection rate percentage
   - Prediction distribution
   - Sample predictions table
   - Performance metrics (latency)
7. **Export**: Download full prediction results as CSV

### API Endpoints
- `GET /` - Web interface
- `GET /api/models` - List available models
- `POST /api/load_model` - Load a model
- `POST /api/predict` - Run predictions (file upload)
- `POST /api/predict_json` - Run predictions (JSON API)
- `GET /api/download/<filename>` - Download results
- `GET /api/model_info/<model_name>` - Get model details

## 📊 Usage Workflow

### Via Web Interface:

1. **Start the server**: `python app.py`
2. **Open browser**: Navigate to `http://localhost:5000`
3. **Select model**: Click on any model from the grid
4. **Load model**: Click "Load Selected Model" button
5. **Upload data**: Choose a CSV file with DNS flows
6. **Set options**: (Optional) Limit max rows to process
7. **Predict**: Click "Run Prediction"
8. **View results**: See statistics, distribution, and sample predictions
9. **Download**: Get full results as CSV

### Via API (Python):

```python
import requests

# 1. Load model
response = requests.post('http://localhost:5000/api/load_model', json={
    'model_type': 'lightgbm',
    'model_path': './results/lightgbm_complete_with_pkl_20251106_160526/models/lightgbm_detector.txt',
    'model_name': 'lightgbm_complete_with_pkl_20251106_160526'
})

# 2. Run prediction
with open('test_data.csv', 'rb') as f:
    response = requests.post('http://localhost:5000/api/predict',
        files={'file': f},
        data={'model_name': 'lightgbm_complete_with_pkl_20251106_160526'}
    )
    results = response.json()
    
print(f"Detected {results['statistics']['malicious_detected']} malicious flows")
print(f"Detection rate: {results['statistics']['detection_rate']:.2f}%")
```

### Via Command Line (curl):

```powershell
# Load model
curl -X POST http://localhost:5000/api/load_model `
  -H "Content-Type: application/json" `
  -d '{\"model_type\": \"lightgbm\", \"model_path\": \"./results/lightgbm_complete_with_pkl_20251106_160526/models/lightgbm_detector.txt\", \"model_name\": \"lightgbm_complete\"}'

# Run prediction
curl -X POST http://localhost:5000/api/predict `
  -F "file=@BCCC-CIC-Bell-DNS-2024/test_unknown/benign_1.csv" `
  -F "model_name=lightgbm_complete"
```

## 🎨 Web Interface Preview

The interface includes:
- **Beautiful gradient design** with purple/blue theme
- **Responsive grid layout** for model selection
- **Real-time status indicators** (selected, loaded models)
- **Interactive statistics cards** showing key metrics
- **Sample predictions table** with color-coded badges
- **Smooth animations** and transitions
- **Alert notifications** for user feedback

## 📈 Example Output

```
Statistics:
- Total Flows: 10,000
- Malicious Detected: 523
- Detection Rate: 5.23%
- Avg Latency: 2.15ms

Prediction Distribution:
- Benign: 9,477 flows (94.77%)
- Malware: 312 flows (3.12%)
- Phishing: 156 flows (1.56%)
- Spam: 55 flows (0.55%)
```

## 🔧 Configuration

### Change Port
Edit `app.py` line 375:
```python
app.run(debug=True, host='0.0.0.0', port=5000)  # Change 5000 to desired port
```

### Max Upload Size
Edit `app.py` line 24:
```python
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
```

### Enable External Access
Already configured! The app binds to `0.0.0.0`, making it accessible from other machines on your network at:
```
http://<your-ip-address>:5000
```

## 🐛 Troubleshooting

### Models not showing up
- Check `results/` directory exists
- Verify models follow naming: `<type>_detector.{txt|pkl|h5}`
- Run `python test_flask_app_setup.py` to verify

### Prediction errors
- Ensure CSV has all required features (121 features)
- Check model metadata JSON exists
- Verify model is loaded before prediction

### Import errors
```powershell
pip install flask flask-cors werkzeug
```

## 📚 Documentation

See **`FLASK_APP_GUIDE.md`** for:
- Complete API documentation
- Advanced configuration options
- Production deployment guide
- Security best practices
- Troubleshooting guide

## 🎯 Next Steps

1. **Test the interface**: 
   ```powershell
   python app.py
   ```

2. **Try different models**: Compare LightGBM vs Hybrid vs Random Forest

3. **Test with your data**: Upload your own DNS traffic CSV files

4. **Integrate with systems**: Use the API for automated detection

5. **Production deployment**: Follow the guide for Gunicorn/nginx setup

## 💡 Tips

- **Start with LightGBM**: Fastest inference, great accuracy
- **Use max_rows**: Limit rows when testing with large files
- **Monitor latency**: Check if meets <100ms SLA requirement
- **Compare models**: Load different models to see performance differences
- **Export results**: Download CSV for further analysis

## 🎊 Success!

Your Flask web application is ready to use! You now have a professional-grade web interface for DNS spoofing detection with support for multiple ML models.

**Start the app and open http://localhost:5000 to begin!**
