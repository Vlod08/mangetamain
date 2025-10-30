Usage
=====

Quick start
-----------

After installing dependencies (see `Installation`), launch the Streamlit app:

```bash
poetry run streamlit run src/mangetamain/app/Home.py
```

The app runs by default on http://localhost:8501.

Running tests
-------------

Run the test suite with pytest (the repository includes a minimal smoke test):

```bash
poetry run pytest -q
```

Building the documentation
--------------------------

To build the HTML documentation locally, inside the `docs/` folder, run:

```bash
poetry run sphinx-apidoc -o ./source ../src/mangetamain ../src/mangetamain/app --force --separate --module-first
poetry run make html -C docs
```

Tips
----

- If the app fails to start because of missing data, mount your local `data/`
  folder into the container or ensure `data/processed` has the expected parquet files.
