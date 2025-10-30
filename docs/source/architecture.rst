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

- `core/dataset.py` — `RecipesDataset` and `InteractionsDataset` helpers that
  load parquet/csv artifacts and provide cached accessors used by the UI.
- `core/recipes_preprocessing.py` — recipes cleaning and nutrition parsing.
- `core/interactions_preprocess.py` — reviews/interactions cleaning pipeline.
- `core/clustering_recipes.py` — TF-IDF + KMeans clustering utilities.
- `core/clustering_nutrivalues.py` — numeric nutriment similarity helpers.

Design notes
------------

- The app uses Streamlit for UI and keeps computation-heavy logic inside
  `core/` so it can be tested and documented separately from UI pages.
- Avoid import-time side-effects in library modules; move I/O into
  functions to keep imports cheap and safe for tools like Sphinx.
