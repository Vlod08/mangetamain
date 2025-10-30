Overview
========

What is Mangetamain
--------------------

**Mangetamain** is an analysis and visualization toolkit developed for the
Kit Big Data project. It provides data cleaning, exploratory analysis and
several interactive Streamlit pages to explore recipes and user interactions
from the Food.com dataset.

Quick structure
---------------

- `src/mangetamain/core` — algorithmic and preprocessing modules (clustering,
  dataset loaders, preprocessing)
- `src/mangetamain/app` — Streamlit application pages and UI glue
- `data/` — raw and processed datasets (not included in published packages)

When to read the API
---------------------

The API reference (under "Modules") contains automatically generated
documentation for the core Python modules. UI pages that rely on
Streamlit are intentionally excluded from automatic imports because they run
at import-time and depend on a running Streamlit environment.

Want to run the app?
--------------------

See the `Usage` page for a quick start and Docker instructions.
