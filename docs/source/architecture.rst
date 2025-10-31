Architecture
============

High-level package layout
-------------------------

The repository follows a simple layout:

- `src/mangetamain` — main Python package
  - `app/` — Streamlit pages and UI glue (runtime-only)
  - `core/` — core logic: dataset loaders, preprocessing, clustering, services
  - `preprocessing/` — utility preprocessing scripts
- `data/` — raw and processed datasets
- `docs/` — Sphinx documentation sources

Core modules to look at
-----------------------

- `core/dataset.py` — `RecipesDataset` and `InteractionsDataset` loaders and
  preprocessing helpers (parquet / csv loaders, cleaning utilities, and
  caching-friendly helpers such as `to_hashable`).
- `core/recipes_eda.py` — Recipes exploratory data analysis helpers (duplication
  detection, nutrition expansion, histograms, TF-IDF signatures, etc.).
- `core/interactions_eda.py` — Interactions/reviews EDA (text features,
  time-series rollups, cohorts, token analysis, user/review aggregates).
- `core/app_logging.py` — application logging utilities and configuration.
- `core/map_builder/map.py` — helper to build interactive maps used by the UI.
- `core/handlers/*` — small "handler" modules used to enrich recipes with
  country or season information (see `core/handlers/country_handler.py` and
  `core/handlers/seasonnality_handler.py`).
- `core/clustering/` — clustering helpers (TF-IDF for ingredients, nutrient
  clustering, time-tag clustering). Files include:
  - `clustering_recipes.py`
  - `clustering_ingredients.py`
  - `clustering_nutrivalues.py`
  - `clustering_time_tags.py`

Design notes
------------

- The app uses Streamlit for UI and keeps computation-heavy logic inside
  `core/` so it can be tested and documented separately from UI pages.
- Avoid import-time side-effects in library modules; move I/O into
  functions to keep imports cheap and safe for tools like Sphinx.

Deployment / runtime
--------------------

- The project includes a `Dockerfile` with a reproducible runtime image
  (base Python 3.11 slim). When running locally we recommend using Poetry
  to install dependencies; for production the Docker image provides an
  isolated runtime with `streamlit` started via the `main.py` entrypoint.

- The docker images is available on Docker Hub at
  `vlod08/mangetamain:latest`.

Notes for contributors
----------------------

- Keep core logic free of Streamlit UI constructs when possible so functions
  can be imported by Sphinx/autodoc and exercised by tests without a Streamlit
  runtime. Use the testing helpers in `tests/conftest.py` which already
  mock Streamlit cache decorators for the test environment.
