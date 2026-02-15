# CureHelp+ | AI-Powered Health Risk Analyzer

<div align="center">

![CureHelp+](https://img.shields.io/badge/CureHelp+-Healthcare_AI-blue?style=for-the-badge&logo=medical)
![Flask](https://img.shields.io/badge/Built%20with-Flask-000000?style=for-the-badge&logo=flask)
![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn)

**Your Personal Health Companion for Predictive Diagnostics and Medical Assistance**

</div>

## 🌟 Overview

CureHelp+ is a Flask-based healthcare analytics platform that combines classical ML risk prediction, chest X-ray screening, medical report parsing, chatbot support, consultant discovery, and downloadable PDF reporting in one workflow.

It supports:

- **Patient profile management** with persistent storage and session sync
- **Multi-disease risk prediction** for tabular clinical inputs
- **Pneumonia & Tuberculosis X-ray analysis** via image upload endpoints
- **Medical report autofill** from uploaded CSV/PDF/XLS/XLSX files
- **Rule-based healthcare chatbot** powered by curated datasets
- **Consultant directory search** (hospitals + doctors)
- **Admin dashboard** for operational visibility and patient management

## 🔗 Project Links (Legacy)

- **Live Link:** https://www.curehelplus.me
- **Azure Container Link:** https://curehelplus-app.lemonmoss-d3a1a3a9.centralindia.azurecontainerapps.io/
- **Docker Image Link:** https://hub.decker.com/r/asimhusain/curehelplus

## ✅ What’s New / Current Highlights

- Admin panel with login, dashboard metrics, and patient deletion actions
- Medical report upload with format validation and 200 MB size limit
- Enhanced PDF reports with risk gauges and detailed protocols
- Profile persistence layer with **JSON** and optional **PostgreSQL** backend
- Chest X-ray endpoints for pneumonia and tuberculosis risk scoring

## 🧠 Machine Learning Modules

| Disease / Task | Input Type | Model/Approach | Output |
|---|---|---|---|
| Type-2 Diabetes | Tabular | Serialized sklearn pipeline | Probability (%) |
| Coronary Artery Disease | Tabular | Serialized sklearn pipeline | Probability (%) |
| Anemia | Tabular | Risk + type model stack | Probability (%) + severity/type |
| Pneumonia | X-ray Image | TensorFlow/Keras model (`.keras`) | Probability + Normal/Pneumonia |
| Tuberculosis | X-ray Image | PyTorch model (`.pth`) | Probability + Normal/Tuberculosis + confidence band |

## 🧩 Core Features

### 1) Patient Profile Workflow

- Create profile using JSON or multipart form data
- Optional medical report upload during profile creation
- Session tracks active profile and disease predictions
- Search, list, and delete stored profiles

### 2) Prediction Workflow

- Tabular predictions:
  - Type-2 Diabetes
  - Coronary Artery Disease
  - Anemia
- Image predictions:
  - Pneumonia (`multipart/form-data` with `image`)
  - Tuberculosis (`multipart/form-data` with `image`)

### 3) Reports and Guidance

- Session prediction summary endpoint
- Filterable PDF export by selected diseases
- Risk-sensitive recommendation text generation

### 4) Chatbot and Consultant Directory

- Rule-based chatbot using `bot_data/` datasets
- Cached responses (TTL controlled by environment variable)
- Consultant directory with hospitals/doctors and search support

### 5) Admin Panel

- Admin login/logout and protected routes
- Dashboard metrics: profile counts, prediction counts, high-risk indicators
- Disease and gender breakdowns
- Recent patient activity with delete operation

## 📁 Project Structure (High Level)

- `app.py` → Flask entry point, route definitions, model inference orchestration
- `profile_manager.py` → profile persistence manager (JSON/PostgreSQL + fallback)
- `report_parser.py` → extraction + field mapping for CSV/PDF/XLS/XLSX reports
- `chatbot.py` → dataset loading, query processing, response formatting, cache
- `consultant.py` → hospitals/doctors catalog and provider search
- `makepdf.py` → PDF generation with gauge visualization and recommendations
- `admin/` → admin blueprint, auth flow, templates, dashboard logic
- `models/` → trained model artifacts used at runtime
- `bot_data/` → chatbot knowledge datasets
- `templates/`, `static/` → frontend templates and static assets

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip
- Git
- Optional: Docker
- Optional: PostgreSQL (only if using DB-backed profile storage)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/asimhusain-ai/CureHelpPlus.git
cd CureHelpPlus
```

2. **Create and activate a virtual environment**

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate    # macOS / Linux
```

3. **Install dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Run the application**

```bash
flask --app app run
```

5. **Open in browser**

- http://127.0.0.1:5000

## ⚙️ Environment Variables

### Core App

| Variable | Description | Default |
|---|---|---|
| `CUREHELP_SECRET_KEY` | Flask session secret key | `curehelp-secret-key` |
| `TB_THRESHOLD` | Tuberculosis decision threshold (0-1) | `0.50` |
| `MODEL_WARMUP_ENABLED` | Enable startup model warmup thread | `true` |
| `MODEL_HEALTH_CHECK_INTERVAL_SECONDS` | Periodic model health-check interval | `300` |

### Admin Authentication

| Variable | Description | Default |
|---|---|---|
| `CUREHELP_ADMIN_USER` | Admin username | `admin` |
| `CUREHELP_ADMIN_PASS` | Admin password | `curehelp` |

### Chatbot

| Variable | Description | Default |
|---|---|---|
| `CHAT_CACHE_TTL_SECONDS` | Chat response cache TTL in seconds | `300` |

### Profile Storage Backend

| Variable | Description | Default |
|---|---|---|
| `PROFILE_STORAGE_BACKEND` | Force backend (`json` / `postgres`) | auto-detect |
| `DATABASE_URL` | PostgreSQL connection URL | empty |
| `PROFILE_STORAGE_STRICT` | Fail if postgres requested but unavailable | `false` |
| `PROFILE_RUNTIME_JSON_FALLBACK` | Enable runtime fallback to JSON on DB errors | `true` |
| `PROFILE_AUTO_MIGRATE_JSON` | Migrate JSON profiles into postgres on startup | `true` |
| `PROFILE_DB_CONNECT_TIMEOUT` | PostgreSQL connect timeout in seconds | `2` |
| `PROFILE_DB_STATEMENT_TIMEOUT_MS` | PostgreSQL statement timeout in milliseconds | `3000` |
| `PROFILE_SYNC_QUEUE_MAXSIZE` | Max queued async postgres sync operations | `2000` |

These timeout controls help reduce login/reload delays when PostgreSQL is slow or temporarily unavailable by failing fast and allowing JSON fallback behavior.

## 🧾 Upload/Input Constraints

### Medical Report Upload (`POST /api/profile`)

- Allowed formats: `.csv`, `.pdf`, `.xls`, `.xlsx`
- Maximum size: **200 MB**

### X-ray Upload (`POST /api/pneumonia`, `POST /api/tuberculosis`)

- Allowed formats: `.jpg`, `.jpeg`, `.png`
- Maximum size: **10 MB**
- Request type: `multipart/form-data`

## 🔌 API Endpoints (Current)

### General

- `GET /` → frontend landing page
- `GET /api/config` → baseline normal values for selected diseases

### Profile and Session

- `POST /api/profile` → create profile (+ optional report upload)
- `GET /api/profile` → get currently active profile from session
- `GET /api/profiles?q=<name>` → list/search profiles
- `DELETE /api/profiles/<profile_id>` → delete profile (if not active)
- `POST /api/reset` → clear session profile and predictions

### Predictions

- `POST /api/diabetes`
- `POST /api/heart`
- `POST /api/anemia`
- `POST /api/pneumonia` (image upload)
- `POST /api/tuberculosis` (image upload)

### Reporting, Chat, and Directory

- `GET /api/report` → prediction summary for active session
- `GET /api/report/pdf?disease=Type-2 Diabetes,Coronary Artery Disease`
- `POST /api/chat` → chatbot response
- `GET /api/consultants?q=<text>` → provider search
- `GET /api/metrics/latency` → p50/p95/p99 latency summary + model health snapshot

### Admin

- `GET/POST /admin/login`
- `POST /admin/logout`
- `GET /admin/`
- `POST /admin/patients/<profile_id>/delete`

## 📦 Required Runtime Artifacts

Ensure these model files are available under `models/`:

- `diabetes_model.pkl`
- `diabetes_scaler.pkl`
- `heart_model.pkl`
- `heart_scaler.pkl`
- `anemia_risk_model.pkl`
- `anemia_type_model.pkl`
- `feature_scaler.pkl`
- `label_encoder.pkl`
- `pneumonia_model.keras`
- `tb_model.pth`

Chatbot dataset files expected in `bot_data/`:

- `Disease precaution.csv`
- `DiseaseAndSymptoms.csv`
- `medquad.csv`
- `humanqa.csv`
- Optional: `Final_Augmented.csv`

## 🧪 Testing

Run all tests:

```bash
pytest
```

## 🐳 Docker

Build and run locally:

```bash
docker build -t curehelplus:latest .
docker run -p 5000:5000 curehelplus:latest
```

## ☁️ Deployment (Azure Container Apps - Legacy Flow)

1. Build image

```bash
docker build -t curehelplus:latest .
```

2. Push to Azure Container Registry

```bash
az acr login --name cureacr
docker tag curehelplus:latest cureacr.azurecr.io/curehelplus:latest
docker push cureacr.azurecr.io/curehelplus:latest
```

3. Deploy

```bash
az containerapp up --name curehelplus --resource-group curehelplus --location central-india --image cureacr.azurecr.io/curehelplus:latest --target-port 5000 --ingress external --environment managedEnvironment-curehelplus-ade7
```

## 🌟 Contributing

Contributions are welcome.

- Fork the repository
- Create a feature branch
- Add/modify tests where relevant
- Open a pull request with a clear description

## ⚠️ Medical Disclaimer

CureHelp+ is intended for informational and educational support only. It does not provide a medical diagnosis and is not a substitute for professional healthcare advice.

## Author

Made with ❤️ by **Asim Husain** — https://www.asimhusain.dev
