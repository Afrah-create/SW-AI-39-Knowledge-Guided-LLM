# Agricultural API - Local Run Instructions

## ✅ Installation Complete!

All required packages have been installed and the application is running.

## 🌐 Access Your Application

**Web Interface**: http://localhost:5000

**API Endpoints**:
- `GET /` - Web interface
- `POST /api/recommend` - Get crop recommendations
- `POST /api/download_pdf` - Download PDF report

## 📦 Installed Packages

- ✅ Flask 3.1.2
- ✅ Flask-CORS 6.0.1
- ✅ Pandas 2.0.3
- ✅ NumPy 1.24.3
- ✅ scikit-learn 1.3.0
- ✅ SciPy 1.10.1
- ✅ PyTorch 2.9.0
- ✅ Sentence Transformers 2.2.2
- ✅ Google Generative AI 0.3.2
- ✅ ReportLab 4.0.4
- ✅ Gunicorn 21.2.0
- ✅ OR-Tools 9.7.2996

## 🚀 Running the Application

### Method 1: From the deployment directory
```bash
cd C:\Users\HP\Desktop\Final\deployment
python app/main.py
```

### Method 2: Using Gunicorn (Production)
```bash
cd C:\Users\HP\Desktop\Final\deployment
gunicorn -w 4 -b 0.0.0.0:5000 app.main:app
```

## 🛑 Stopping the Application

Press `Ctrl+C` in the terminal where it's running.

Or find and kill the process:
```bash
taskkill /F /PID <process_id>
# To find process_id: netstat -ano | findstr :5000
```

## 📝 Testing the API

### Using curl:
```bash
curl -X POST http://localhost:5000/api/recommend \
  -H "Content-Type: application/json" \
  -d "{\"soil_properties\":{\"pH\":6.5,\"organic_matter\":2.1,\"texture_class\":\"loam\",\"nitrogen\":120,\"phosphorus\":35,\"potassium\":180},\"climate_conditions\":{\"temperature_mean\":24,\"rainfall_mean\":1200},\"farming_conditions\":{\"available_land\":5.0}}"
```

### Using Python requests:
```python
import requests

response = requests.post('http://localhost:5000/api/recommend', 
    json={
        "soil_properties": {
            "pH": 6.5,
            "organic_matter": 2.1,
            "texture_class": "loam",
            "nitrogen": 120,
            "phosphorus": 35,
            "potassium": 180
        },
        "climate_conditions": {
            "temperature_mean": 24,
            "rainfall_mean": 1200
        },
        "farming_conditions": {
            "available_land": 5.0
        }
    }
)

print(response.json())
```

## 🔧 Troubleshooting

### Port Already in Use
If port 5000 is busy:
```bash
# Change the port in app/main.py
app.run(host='0.0.0.0', port=8000, debug=True)
```

### Module Import Errors
Reinstall requirements:
```bash
pip install -r requirements.txt --force-reinstall
```

### Memory Issues
The application uses significant memory for ML models. Close other applications if needed.

## 🌍 Next Steps: Deploy to Cloud

To deploy your application to the cloud without Docker:

1. **Read**: `CLOUD_DEPLOYMENT_GUIDE.md`
2. **Quick Start**: `QUICK_START.md`
3. **Recommended**: Deploy to Railway.app (easiest option)

## 📊 Application Features

- ✅ Multi-AI Integration (GNN, Constraint Engine, RAG, LLM)
- ✅ Crop Suitability Analysis
- ✅ PDF Report Generation
- ✅ Knowledge Graph (175K+ triples)
- ✅ Evidence-Based Recommendations
- ✅ Land Allocation Planning

---

**Status**: ✅ Running on http://localhost:5000
**Last Updated**: January 2025
