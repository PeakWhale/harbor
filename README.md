<p><img src="harbor.svg" width="64" height="64" alt="Harbor logo"></p>

# PeakWhale™ Harbor

### Turn a Spreadsheet into a Prediction API (Local First, Open Source)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Flask](https://img.shields.io/badge/Flask-Web%20App-black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Docker](https://img.shields.io/badge/Docker-Containerization-informational)
![Gunicorn](https://img.shields.io/badge/Gunicorn-WSGI%20Server-success)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-CI-2088FF)
![GHCR](https://img.shields.io/badge/GHCR-Container%20Registry-6f42c1)
![MCP](https://img.shields.io/badge/MCP-AI%20Assistants-8a2be2)
![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-green)

PeakWhale™ Harbor reads a CSV file and learns to predict one number column.
Then it gives you:

* a web page with a form that is made from your columns
* a JSON API (Application Programming Interface)
* batch predictions: upload a CSV file, get a CSV file back
* a **range** with every prediction ("likely between X and Y"), not only one number
* honest test results, measured on rows the model never saw

Everything runs on your own computer. No cloud account is needed.

## Who Is It For

* Analysts and small teams who have a spreadsheet and want a quick, local prediction service.
* Developers who need a solid first model behind an API, with a clear schema.
* Learners who want to see the full path from a CSV file to a running, tested service.

Harbor is a starting point. It is not a tool for high-stakes decisions without expert review.

## Quick Start (3 Steps)

You need Python 3.10 or newer.

**1. Install**

```bash
git clone https://github.com/PeakWhale/harbor.git
cd harbor
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

On Windows (Command Prompt), create and start the environment like this instead:

```bat
py -m venv .venv
.venv\Scripts\activate
pip install -e .
```

The longer commands below are split over several lines with `\` for macOS and Linux.
On Windows, type each one on a single line without the `\`.

**2. Train the demo model** (California housing data, usually well under a minute)

```bash
harbor train data/california_housing.csv --target median_house_value \
  --descriptions data/california_housing.columns.json \
  --out models/california-housing
```

**3. Start the app**

```bash
harbor serve models/california-housing
```

Open <http://127.0.0.1:5050>. Change some values and click **Predict**.
(Use `--port` to pick another port.)

If training or predictions are very slow on a busy computer, run with
`OMP_NUM_THREADS=1`, for example `OMP_NUM_THREADS=1 harbor train ...`
(on Windows, run `set OMP_NUM_THREADS=1` first). Harbor already uses one thread
for predictions, and the Docker image sets this for you.

Prefer Docker? Run `docker build -t peakwhale-harbor .` and then
`docker run -p 127.0.0.1:8000:8000 peakwhale-harbor`. Open <http://localhost:8000>.

## Bring Your Own CSV

Your file needs:

* one header row with column names
* one row per example (one house, one order, one machine, and so on)
* one column with the number you want to predict (the **target**)

Then run:

```bash
harbor train my_data.csv --target price
harbor serve models/my-data
```

Harbor looks at every column and decides what to do with it:

| What the column holds | What Harbor does |
| --- | --- |
| Numbers | Uses it as a **numeric** input. Empty cells are fine. |
| A few different text values (like "north", "south") | Uses it as a **categorical** input. |
| A different text value in almost every row (codes, names, free text) | Skips it and tells you why. |
| Number IDs (see below) | Skips it and tells you why. |
| Dates, such as `2024-01-31`, `31/01/2024`, `1/5/2024 0:00`, `Jan 5, 2024` or `Jan 2024` | Skips it. Add columns like `year` or `month` instead. |
| Only empty cells, or the same value in every row | Skips it. |

Spreadsheet number formats are fine: `$1,234.50`, `2,092` and `12%` are read as
1234.5, 2092 and 12. Cells such as `-`, `?` or `#DIV/0!` in a number column count as
empty, both in training and in predictions. Numbers need a decimal point (`2.5`), not a
decimal comma (`2,5`).

Harbor finds number IDs only by their name (such as `id`, `customer_id`, `CustomerID`
or `order_no`) or when they simply count the rows (1, 2, 3, ...). An ID with another
name looks like a normal number to Harbor. **Check the printed list after training**,
and use `--drop` for any ID column that Harbor did not skip.

Harbor prints its choices. If a choice is wrong, change it:

```bash
harbor train my_data.csv --target price \
  --drop customer_name \
  --categorical zip_code \
  --numeric rating
```

Other useful options:

* `--out models/my-model` picks the output folder.
* `--coverage 0.8` asks for an 80% range instead of 90%.
* `--seed 7` changes the random split.
* `--title "My Model"` and `--descriptions columns.json` set the text on the web page.
  `columns.json` is a simple JSON object: `{"column name": "short description"}`.

Run `harbor train --help` to see all options.

Harbor also has a second, small example that is not about housing. It is made-up
data about machine repairs at a factory:

```bash
harbor train examples/equipment_repairs.csv --target repair_hours
harbor serve models/equipment-repairs
```

More commands:

```bash
harbor info models/equipment-repairs                      # inputs and test results
harbor predict models/equipment-repairs new_rows.csv --out predictions.csv
```

`new_rows.csv` is your own CSV file with the same input columns as the training file
(the target column is not needed). The `/template.csv` page of the web app gives you
one with the right columns. To try it now, use `examples/equipment_repairs.csv` itself.

Excel file? Save it as CSV first: File > Save As > **CSV UTF-8 (Comma delimited)**.
Harbor also reads older "CSV (Comma delimited)" files, files with `;` between the
columns (Excel in many European languages) and `.tsv` files. `harbor train` and
`harbor predict` say so when a file is not UTF-8 text.

## What Is in a Model Bundle

`harbor train` writes one folder, called a **bundle**:

```text
models/california-housing/
  model.skops     the trained models (skops format)
  schema.json     input fields, target, range settings, JSON Schema for the API
  metrics.json    test results on rows the model never saw
  metadata.json   when and how it was trained: options, row counts, library versions, checksums
```

The models are saved with [skops](https://skops.readthedocs.io), not with Python
pickle. skops refuses unknown object types, and Harbor only allows a short, fixed list
of extra types. Harbor also checks the file against the SHA-256 checksum in
`metadata.json`.

> **Safety:** loading a model file from an untrusted source is never fully safe,
> even with skops. Treat a bundle like a program. Only load bundles that you trained
> yourself or that come from someone you trust.

A bundle works best with the same library versions that trained it. Harbor warns
you when the scikit-learn version is different. If a bundle cannot be loaded at all,
Harbor names the versions that trained it and the ones you have. In both cases,
install the versions in `requirements.txt` or train again.

## API

| Method and path | What it does |
| --- | --- |
| `GET /` | The web page with the form |
| `GET /health` | Health check |
| `GET /schema` | Inputs, target, metrics, JSON Schema and an example request |
| `POST /predict_api` | Predict one record or a list of records (JSON) |
| `POST /predict_batch` | Upload a CSV file (form field `file`), get a CSV file back |
| `GET /template.csv` | A CSV file with the right columns and one example row |
| `POST /predict` | Form posts from the web page |

The examples below use the demo model on port 5050.

**One record**

```bash
curl -X POST http://localhost:5050/predict_api \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "longitude": -122.23,
      "latitude": 37.88,
      "housing_median_age": 41,
      "total_rooms": 880,
      "total_bedrooms": 129,
      "population": 322,
      "households": 126,
      "median_income": 8.3252,
      "ocean_proximity": "NEAR BAY"
    }
  }'
```

The answer looks like this (numbers shortened):

```json
{
  "prediction": 440721.9,
  "low": 309652.8,
  "high": 508046.1,
  "coverage": 0.9,
  "notes": [],
  "target": {"name": "median_house_value", "description": "Median house value in the district, ..."}
}
```

**Rules for input**

* Send every input field. Use `null` when a value is unknown. The model can handle
  empty values, and `notes` tells you about them.
* Extra fields are ignored.
* A category value must be one of the values seen in training (upper or lower case does not matter).
* A number outside the range seen in training is allowed, but `notes` warns you.

**Many records**: send a list, `{"data": [{...}, {...}]}`, up to 1,000 records.
The answer has a `predictions` list.

**Errors** return HTTP 400 and say what is wrong:

```json
{
  "error": "1 problem(s) found in the input.",
  "problems": [
    {
      "row": 0,
      "field": "ocean_proximity",
      "error": "unknown value 'BEACH'. Allowed values: '<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'"
    }
  ],
  "required": ["longitude", "latitude", "..."]
}
```

**CSV batch**

```bash
curl -o template.csv http://localhost:5050/template.csv
curl -F file=@template.csv http://localhost:5050/predict_batch -o predictions.csv
```

You get your file back with four new columns: `<target>_prediction`, `<target>_low`,
`<target>_high` and `notes`. If your file already has a column with one of these names,
the new column gets `_2` at the end, so none of your columns are replaced. Errors give
the CSV line number. Uploads are limited to 10 MB (set `HARBOR_MAX_UPLOAD_MB` to change
this), 100,000 rows and 1,000 columns.

## How the Ranges Work

A single number hides how sure the model is. So Harbor gives a range too.

1. Harbor splits your rows: 60% to **train**, 20% to **calibrate**, 20% to **test**.
2. It trains three gradient boosting models (scikit-learn
   `HistGradientBoostingRegressor`) on the train rows:
   one for the best guess, one for a low value (5th percentile) and one for a high
   value (95th percentile).
3. On the calibration rows, Harbor checks how often the true value falls inside the
   low-high range. Then it makes the range a little wider or narrower, so that about
   90% of true values fall inside. This method is called
   *conformalized quantile regression* (Romano, Patterson and Candès, 2019).
4. On the test rows, which no model has seen, Harbor measures how often the true value
   really was inside the range. That number is in `metrics.json` and on the web page.

What "90% range" means: over many predictions, about 9 out of 10 true values should be
inside their range. It is not a promise for one single row. It only holds when new rows
look like the training rows. The range always contains the best guess. If the target
is never negative in your data, the range does not go below zero.

The range is wider when a row is harder to predict. That is useful: a wide range
tells you to be careful.

## Demo Results (Measured)

These numbers come from the commands below, run with the versions in
`requirements.txt` (Python 3.11). The Docker build runs the same command with the same
versions. All numbers are measured on test rows that the model never saw.

**California housing** (20,640 rows, 4,128 test rows):

```bash
harbor train data/california_housing.csv --target median_house_value \
  --descriptions data/california_housing.columns.json \
  --out models/california-housing
```

```text
Hold-out test results (4,128 rows the model never saw):
  Typical error (MAE)                31,278
  Root mean squared error (RMSE)     46,501
  R² (1.0 is perfect)                0.835
  Simple guess error (median)        87,404
  90% range: true value inside       90.0% of test rows
  Average range width                142,540
```

The values are in US dollars of 1990. The "simple guess" always predicts the median
house value of the training rows. The model's typical error is about one third of that.

**Equipment repairs** (1,200 made-up rows, 240 test rows):

```bash
harbor train examples/equipment_repairs.csv --target repair_hours
```

```text
Hold-out test results (240 rows the model never saw):
  Typical error (MAE)                6.38
  Root mean squared error (RMSE)     8.66
  R² (1.0 is perfect)                0.661
  Simple guess error (median)        11.92
  90% range: true value inside       87.1% of test rows
  Average range width                23.56
```

With only 240 test rows, the measured share moves around by a few points. Here it is
87.1%, a bit below the 90% goal. Small data gives less exact ranges.

Other library versions or another `--seed` can change these numbers a little.

## Use Harbor from an AI Assistant (MCP)

Harbor can run as an MCP (Model Context Protocol) server. AI assistants that support
MCP can then ask your model for predictions. There are two tools:

* `describe_model`: the input fields, allowed values, target and test results
* `predict`: one prediction with its range and notes

Install the extra package and start the server:

```bash
pip install -e ".[mcp]"
harbor mcp models/california-housing
```

The server talks over standard input and output, so normally your assistant starts
it for you. Add it to your assistant's MCP settings (for example
`claude_desktop_config.json`). Use full paths:

```json
{
  "mcpServers": {
    "harbor-california-housing": {
      "command": "/full/path/to/harbor/.venv/bin/harbor",
      "args": ["mcp", "/full/path/to/harbor/models/california-housing"]
    }
  }
}
```

On Windows, the command is `C:\\full\\path\\to\\harbor\\.venv\\Scripts\\harbor.exe` (in JSON,
every `\` is written twice).

## Docker

The image trains the California demo model while it builds. Then it serves the model
with Gunicorn on port 8000.

```bash
docker build -t peakwhale-harbor .
docker run --name harbor -p 127.0.0.1:8000:8000 -e PORT=8000 peakwhale-harbor
```

`-p 127.0.0.1:8000:8000` makes the app reachable only from your own computer. The API
has no login. Before you open it to other computers, put it behind a reverse proxy that
asks for a login.

To serve your own bundle, mount it and set `HARBOR_BUNDLE`:

```bash
docker run -p 127.0.0.1:8000:8000 \
  -v "$PWD/models/my-model:/bundle:ro" \
  -e HARBOR_BUNDLE=/bundle \
  peakwhale-harbor
```

The image has the exact versions in `requirements.txt` (Python 3.11). Train the bundle
with the same versions (`pip install -r requirements.txt -e .` on Python 3.11), or the
image may not be able to load it. You can also train inside the image:

```bash
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/work" -w /work \
  --entrypoint harbor peakwhale-harbor \
  train my_data.csv --target price --out models/my-model
```

Settings: `PORT` (default 8000), `WORKERS` (Gunicorn workers, default 2),
`HARBOR_BUNDLE`, `HARBOR_MAX_UPLOAD_MB` and `OMP_NUM_THREADS` (default 1). The container
runs as a normal user, not root.

## Run from GHCR

On pushes to `main`, GitHub Actions runs the tests, builds the image, checks
`/health` and one prediction, and then publishes the image to GHCR.

If the package is private, log in with a GitHub Personal Access Token (PAT) that has
the `read:packages` permission:

```bash
docker login ghcr.io -u YOUR_GITHUB_USERNAME
docker pull ghcr.io/peakwhale/harbor:latest
docker run --name harbor -p 127.0.0.1:8000:8000 -e PORT=8000 ghcr.io/peakwhale/harbor:latest
```

## Honest Limits

* **Not financial or appraisal advice.** The demo data is from 1990. It does not tell
  you what a home is worth today.
* **Numbers only.** Harbor predicts a number (regression). Yes/no or category targets
  are not supported yet.
* **No dates or free text.** Harbor skips these columns. It knows the common date
  formats; a date written another way may be read as a category, so check the list.
* **New data can be different.** Test results and ranges hold for rows that look like
  the training rows. A number outside the training range gets a note, because the
  model cannot extrapolate.
* **Random split.** Harbor splits rows at random. If your rows are in time order (for
  example sales by month), the real future error can be larger than the test error.
* **Capped data.** In the California data, house values stop at 500,001. The model
  cannot tell you anything about more expensive areas. The high end of a range can go a
  little above the cap.
* **Small data, less exact.** With a few hundred rows, test results and ranges move around.
* **No login.** The API has no authentication. Do not put it on the public internet
  without a proxy that adds it.
* **Trust your bundles.** Only load model files you trust (see above).

## Data and Licenses

* California housing data: from the 1990 U.S. Census, via Pace and Barry (1997) and
  the Apache-2.0 [handson-ml2](https://github.com/ageron/handson-ml2) repository.
  Source, checksum and details are in [data/README.md](data/README.md).
* Equipment repairs data: made up by `scripts/make_example_data.py` (fixed seed).
* Earlier versions of Harbor used the Boston housing data. That data set is retired,
  because one of its columns was built from the share of Black residents in each
  town. Harbor no longer uses it.

## Development

```bash
pip install -r requirements.txt -e ".[dev]"
pytest
```

`requirements.txt` holds the exact versions for Docker and CI, and they need Python
3.11 or newer. On Python 3.10, install without it: `pip install -e ".[dev]"`.

The tests cover column type inference, train → save → load → predict, the API
endpoints, CSV batch files, bad input, the MCP server, and the rule that every range
contains its prediction.

## Repository Structure

```text
harbor/
  src/peakwhale_harbor/
    cli.py           the harbor command
    csvfile.py       read CSV files (separators, encodings)
    schema.py        column types and input checks
    train.py         training, ranges and metrics
    bundle.py        save and load bundles, predictions
    server.py        Flask web app and API
    mcp_server.py    MCP server for AI assistants
    templates/home.html
  data/              California housing demo data and its source notes
  examples/          made-up equipment repairs data
  scripts/           script that makes the example data
  tests/             pytest tests
  .github/workflows/main.yaml
  Dockerfile
  entrypoint.sh
  pyproject.toml
  requirements.txt   exact versions for Docker and CI
  LICENSE
```

## Part of the PeakWhale™ Ecosystem

* **PeakWhale™ Helm**: research assistant over your own data.
* **PeakWhale™ Orca**: real-time fraud detection.
* **PeakWhale™ Harbor**: turn a spreadsheet into a prediction API.
* **PeakWhale™ Trident**: fail-safe triage engine, AI plus editable rules; unclear cases go to a person.
* **PeakWhale™ Berth**: EV fleet charging planner.

## License

Apache License 2.0 © 2025-2026 Aditya Rao (Addy) Kaveti. See [LICENSE](LICENSE) and [NOTICE](NOTICE). If you share or build on Harbor, keep the NOTICE file. PeakWhale™ is Addy's project name; the license does not cover the name or logo.

## Author

Built by Addy
