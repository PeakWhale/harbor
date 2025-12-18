# PeakWhale™ Harbor

### Valuation and Forecasting Sandbox (Local First, Open Source, Demo First)

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Flask](https://img.shields.io/badge/Flask-Web%20App-black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Docker](https://img.shields.io/badge/Docker-Containerization-informational)
![Gunicorn](https://img.shields.io/badge/Gunicorn-WSGI%20Server-success)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-CI-2088FF)
![GHCR](https://img.shields.io/badge/GHCR-Container%20Registry-6f42c1)
![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-green)

PeakWhale™ Harbor is a local first demo that serves a scikit learn regression model through a Flask web User Interface (UI) and a JSON Application Programming Interface (API).

The current model predicts the Boston housing target MEDV, median home value in thousands of US dollars, using the classic feature set.

## Enterprise Business Problem

In real teams, demos often fail to become usable services because:

* artifacts are scattered or not versioned
* feature ordering is ambiguous at inference time
* local runs are not repeatable across machines
* deployments drift from development environments

Harbor is a small, readable example of doing the basics cleanly.

## Important Notice

This repository uses a classic housing dataset for demonstration only.

Do not treat outputs as real world pricing advice.

## What Harbor Does

Harbor provides:

* UI based prediction from a browser
* API based prediction from JSON requests
* a health endpoint for monitoring
* a schema endpoint that documents inputs and outputs

## App Endpoints

* `/` renders the UI
* `/health` returns a health check JSON
* `/schema` returns model metadata and required input fields
* `/predict` accepts HTML form posts
* `/predict_api` accepts JSON payloads

## Model Inputs

Inputs are the standard Boston housing features:

* `CRIM` per capita crime rate by town
* `ZN` proportion of residential land zoned for lots over 25,000 sq ft
* `INDUS` proportion of non retail business acres per town
* `CHAS` Charles River dummy variable (1 if tract bounds river, else 0)
* `NOX` nitric oxides concentration (parts per 10 million)
* `RM` average number of rooms per dwelling
* `AGE` proportion of owner occupied units built prior to 1940
* `DIS` weighted distances to five Boston employment centres
* `RAD` index of accessibility to radial highways
* `TAX` full value property tax rate per 10,000 dollars
* `PTRATIO` pupil teacher ratio by town
* `B` 1000(Bk minus 0.63)^2 where Bk is the proportion of Black residents by town
* `LSTAT` percent lower status of the population

Output:

* `prediction` predicted MEDV, in thousands of US dollars

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
* scikit learn for preprocessing and regression
* Gunicorn Web Server Gateway Interface (WSGI) server for production
* Docker for container builds
* GitHub Actions for Continuous Integration (CI)
* GitHub Container Registry (GHCR) for publishing images

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

* `artifacts/regmodel.pkl` contains the trained regression model
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
docker run --name harbor -p 8000:8000 -e PORT=8000 peakwhale-harbor
```

Open:

```text
http://localhost:8000
```

## Run from GHCR

On pushes to `main`, GitHub Actions builds and publishes an image to GHCR.

If the package is private, authenticate with a GitHub Personal Access Token (PAT) that has `read:packages` permission:

```bash
docker login ghcr.io -u YOUR_GITHUB_USERNAME
docker pull ghcr.io/peakwhale/harbor:latest
docker run --name harbor -p 8000:8000 -e PORT=8000 ghcr.io/peakwhale/harbor:latest
```

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
