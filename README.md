# PeakWhale™ Harbor

### Valuation and Forecasting Sandbox (Local First, Open Source, Demo First)

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Flask](https://img.shields.io/badge/Flask-Web%20App-black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Docker](https://img.shields.io/badge/Docker-Containerization-informational)
![Gunicorn](https://img.shields.io/badge/Gunicorn-WSGI%20Server-success)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-CI%20Deploy-2088FF)
![Heroku](https://img.shields.io/badge/Heroku-Docker%20Deploy-79589F)
![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-green)

PeakWhale™ Harbor is a local first valuation and forecasting demo that serves a Machine Learning (ML) model through a Flask web User Interface (UI) and a JSON Application Programming Interface (API).

It is intentionally lightweight so you can use it as a reference for model artifact packaging, deterministic inference, and container deployment.

## Enterprise Business Problem

In real teams, model demos often fail to become usable services because:

* artifacts are scattered or not versioned
* feature ordering is ambiguous at inference time
* local runs are not repeatable across machines
* deployments drift from development environments

Harbor is a small, readable example of doing the basics cleanly.

## Important Notice

This repository uses a classic housing dataset for demonstration only.

Do not treat outputs as real world pricing advice.

## What Harbor Does

Harbor provides two ways to run predictions:

* UI form based prediction from a browser
* API based prediction from JSON requests

It also includes a simple health endpoint so deployments can be monitored.

## App Endpoints

* `/` renders the UI
* `/health` returns a health check JSON
* `/predict` accepts HTML form posts
* `/predict_api` accepts JSON payloads

## Example API Request

```bash
curl -X POST http://localhost:5000/predict_api \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "CRIM": 0.1,
      "ZN": 0.0,
      "INDUS": 8.0,
      "CHAS": 0,
      "NOX": 0.5,
      "RM": 6.0,
      "AGE": 65.0,
      "DIS": 4.0,
      "RAD": 4.0,
      "TAX": 300.0,
      "PTRATIO": 18.0,
      "B": 390.0,
      "LSTAT": 12.0
    }
  }'
```

## Tech Stack

* Python 3.11+
* Flask web server
* scikit learn for the model and preprocessing
* Gunicorn Web Server Gateway Interface (WSGI) server for production
* Docker for container builds
* GitHub Actions for Continuous Integration (CI) deployment workflow
* Heroku deployment via Docker image push

## Repository Structure

```text
harbor/
  app.py
  artifacts/
    regmodel.pkl
    scaling.pkl
  templates/
    home.html
  .github/
    workflows/
      main.yaml
  Dockerfile
  requirements.txt
  README.md
  LICENSE
```

## Model Artifacts

Harbor loads two artifacts at runtime:

* `artifacts/regmodel.pkl` contains the trained model
* `artifacts/scaling.pkl` contains the fitted scaler

Inference enforces a fixed feature order to keep predictions deterministic.

## Quick Start

### Prerequisites

* Python 3.11 or newer
* pip installed

### Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the app

```bash
python app.py
```

Open:

```text
http://localhost:5000
```

## Docker

### Build

```bash
docker build -t peakwhale-harbor .
```

### Run

```bash
docker run -p 8000:8000 -e PORT=8000 peakwhale-harbor
```

Open:

```text
http://localhost:8000
```

## Deploy to Heroku via GitHub Actions

This repository deploys on pushes to `main` using GitHub Actions.

Required GitHub repository secrets:

* `HEROKU_EMAIL`
* `HEROKU_API_KEY`
* `HEROKU_APP_NAME`

## Why This Project Exists

Harbor demonstrates practical patterns you will reuse in production ML services:

* artifact management for model and scaler
* deterministic feature ordering at inference time
* local first development and repeatable runs
* container based deployment path

## Part of the PeakWhale™ Ecosystem

* PeakWhale™ Helm, multi agent financial intelligence
* PeakWhale™ Orca, real time fraud detection
* PeakWhale™ Harbor, valuation and forecasting sandbox

## License

Apache 2.0 License
© 2025 PeakWhale™

## Author

Built by Addy
