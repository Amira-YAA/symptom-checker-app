# 🏥 Disease Prediction System

An AI-powered medical diagnosis assistant that predicts diseases based on patient symptoms using machine learning.

## 📋 Features

- 🔮 Real-time symptom-based disease prediction
- 📊 Interactive data visualization dashboard
- 🤖 Random Forest model with 82.5% accuracy (on full scale sample of 96,088 pateints)
- 🎯 Top 3 predictions with confidence scores
- 📚 Comprehensive symptom database (230 symptoms)
- 🏥 100 different diseases supported

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
```bash
git clone https://github.com/Amira-YAA/symptom-checker-app.git
cd symptom-checker-app

symptom-checker-app/
├── .streamlit/
│   └── config.toml
├── .env
├── .gitignore
├── requirements.txt
├── setup.py
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── model_trainer.py
│   ├── symptom_categories.py
│   └── utils.py
├── app.py
├── train_model.py
└── README.md
