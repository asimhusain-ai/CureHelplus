# CureHelp+ | AI-Powered Health Risk Analyzer

<div align="center">

![CureHelp+](https://img.shields.io/badge/CureHelp+-Healthcare_AI-blue?style=for-the-badge&logo=medical)
![Flask](https://img.shields.io/badge/Built%20with-Flask-000000?style=for-the-badge&logo=flask)
![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn)

**Your Personal Health Companion for Predictive Diagnostics and Medical Assistance**

</div>

## 🌟 Overview

CureHelp+ is a Flask-based healthcare analytics platform for multi-disease risk prediction, medical chatbot guidance, consultant discovery, and PDF medical reporting. It now includes medical report autofill (CSV/PDF/XLS/XLSX), JSON-backed profile storage with session sync, and an admin console for operational visibility.

- **Live Link** https://www.curehelplus.me
- **Auze Container Link** https://curehelplus-app.lemonmoss-d3a1a3a9.centralindia.azurecontainerapps.io/
- **Docker Image Link** https://hub.decker.com/r/asimhusain/curehelplus

---

## ✅ What’s New (Recent Updates)

- 🛡️ **Admin Panel** with login, dashboard metrics, and patient management
- 📥 **Medical Report Upload & Autofill** (CSV/PDF/XLS/XLSX) with size validation (200 MB)
- 📄 **Enhanced PDF Reporting** with risk gauges and clinical protocols
- 🗂️ **Profile Management** with JSON-backed storage and session sync
- 🔎 **Consultant Directory Search** (hospitals + doctors)

---

### Key Features

- 🩺 **Multi-Disease Risk Prediction** for Diabetes, Heart Disease, Fever, and Anemia
- 🤖 **AI Medical Assistant** using curated datasets in bot_data/
- 📊 **Interactive Results** with risk scores, severity, and clinical guidance
- 👨‍⚕️ **Consultant Directory** with hospitals and doctors + map links
- 🗂️ **Patient Profile Management** with JSON-backed storage and session sync
- 📄 **PDF Report Generation** with risk gauges and protocols
- 📥 **Medical Report Autofill** (CSV/PDF/XLS/XLSX)
- 🔐 **Admin Panel** with monitoring and patient management


---

## 🧠 Machine Learning Models

| Disease | Algorithm | Notes |
|---------|-----------|-------|
| Diabetes | XGBoost | Gender-aware pregnancy handling |
| Heart Disease | Random Forest | Encoded categorical fields |
| Fever | Dual Random Forest | Severity + risk prediction |
| Anemia | Multi-output RF | Risk + type classification |

---

## Usage

1.  **Landing Page:** Upon launching the application, you will be greeted by the landing page. Click on "Get Started" to proceed.
2.  **Patient Details:** Fill in your personal details to create a profile. This information will be used to personalize the predictions and reports.
3.  **Input Health Metrics:** Navigate through the different tabs for each disease (Diabetes, Heart Disease, Fever, Anemia) and enter your health metrics.
4.  **Predict Risk:** Click on the "Predict" button to get your risk assessment.
5.  **View Results:** Review interactive gauges, comparator cards, and AI-powered recommendations.
6.  **Generate Report:** Download a consolidated PDF report with risk protocols and clinical interventions.
7.  **Explore Directory:** Browse nearby hospitals and doctors, or use search to filter specialists.

---

## 🔐 Admin Panel

- **URL:** /admin
- **Default credentials:** admin / curehelp
- **Override with environment variables:**
   - CUREHELP_ADMIN_USER
   - CUREHELP_ADMIN_PASS

The admin console includes:
- Total profiles, predictions, high-risk alerts
- Disease and gender distribution charts
- Recent patient activity with delete actions

---

## 🧾 Reports & Medical Uploads

- **Upload formats:** CSV, PDF, XLS, XLSX
- **Max size:** 200 MB
- **Autofill** supported across diabetes, heart, fever, and anemia fields
- **PDF reports** include gauges, risk protocols, and interventions

---

## 🔗 API Endpoints (Core)

- POST /api/profile → create profile (+ optional report upload)
- GET /api/profile → current profile
- GET /api/profiles → list/search profiles
- DELETE /api/profiles/<profile_id>
- POST /api/diabetes
- POST /api/heart
- POST /api/fever
- POST /api/anemia
- GET /api/report
- GET /api/report/pdf?disease=Diabetes,Heart Disease
- POST /api/chat
- GET /api/consultants?q=search
- POST /api/reset

---

## 📁 Project Structure (High Level)

- app.py → Flask entry point + API routes
- admin/ → Admin blueprint + dashboard templates
- chatbot.py → Rule-based medical assistant
- report_parser.py → Medical report extraction & mapping
- profile_manager.py → JSON-backed patient storage
- makepdf.py → PDF report generation
- static/ + templates/ → UI assets
- models/ → Trained ML artifacts
- bot_data/ → Chatbot datasets

## 🌟 Contributing

I welcome contributions from the community!

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Git

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/asimhusain-ai/CureHelpPlus.git
   cd CureHelpPlus
   ```

2. **Create and Activate a Virtual Environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate      # Windows
   source .venv/bin/activate     # macOS / Linux
   ```

3. **Install Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Prepare Datasets and Models**
   - Place chatbot CSVs inside `bot_data/`
   - Ensure trained model artifacts exist in `models/`
   - Optional: keep sample medical reports in `Sample_inputs/`

5. **Set Environment Variables (optional but recommended)**
   ```bash
   set CUREHELP_SECRET_KEY=change-me      # Windows PowerShell
   set CUREHELP_ADMIN_USER=admin          # Windows PowerShell
   set CUREHELP_ADMIN_PASS=change-me      # Windows PowerShell
   export CUREHELP_SECRET_KEY=change-me   # macOS / Linux
   export CUREHELP_ADMIN_USER=admin       # macOS / Linux
   export CUREHELP_ADMIN_PASS=change-me   # macOS / Linux
   ```

6. **Run the Flask Server**
   ```bash
   flask --app app run
   ```

7. **Access the Dashboard**
   - http://127.0.0.1:5000

---

## ⚙️ Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| CUREHELP_SECRET_KEY | Flask session secret | curehelp-secret-key |
| CUREHELP_ADMIN_USER | Admin username | admin |
| CUREHELP_ADMIN_PASS | Admin password | curehelp |

---

## 🧪 Tests

```bash
pytest
```

---

## Deployment

### Container Deployment (Azure Container Apps)

1. **Build the Docker Image**
   ```bash
   docker build -t curehelplus:latest .
   ```

2. **Tag and Push to Azure Container Registry**
   ```bash
   az acr login --name cureacr
   docker tag curehelplus:latest cureacr.azurecr.io/curehelplus:latest
   docker push cureacr.azurecr.io/curehelplus:latest
   ```

3. **Deploy to Azure Container Apps**
   ```bash
   az containerapp up --name curehelplus --resource-group curehelplus --location central-india --image cureacr.azurecr.io/curehelplus:latest --target-port 5000 --ingress external --environment managedEnvironment-curehelplus-ade7
   ```

4. **Configure Secrets and Storage**
   - Set `CUREHELP_SECRET_KEY` and other env vars with `az containerapp secret set`
   - Mount persistent storage if the container needs to retain `user_profiles.json`

5. **Verify Deployment**
   - Use `az containerapp show --name curehelp-plus --resource-group curehelplus` to fetch the HTTPS endpoint
   - Confirm health with the `/` route and exercise prediction + chatbot flows

---

## ⚠️ Medical Disclaimer

CureHelp+ is intended for informational purposes only and does not provide medical diagnosis. Always consult qualified healthcare professionals for medical advice or treatment.

---

## Author

Made with ❤️ by **Asim Husain** — https://www.asimhusain.dev
