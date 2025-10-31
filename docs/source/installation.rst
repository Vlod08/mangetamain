Installation
============

Requirements
------------

- Python 3.11 (the project targets >=3.11,<3.12)
- Poetry 2.2.0 (recommended) or Docker

Using Poetry (development / local runs)
---------------------------------------

1. Install Poetry (if not installed). The recommended installer is the
   official install script:

   ```bash
   curl -sSL https://install.python-poetry.org | python -
   # or install a pinned poetry version via pip for CI parity:
   python -m pip install --upgrade pip && pip install "poetry==2.2.0"
   ```

2. Install project dependencies:

   ```bash
   poetry install --no-interaction
   ```

4. Run the Streamlit app locally (the actual entrypoint is `main.py`):

   ```bash
   poetry run streamlit run src/mangetamain/main.py --server.port=8501
   ```

Using Docker (recommended for deployment)
-----------------------------------------

Build the image (from project root):

```bash
docker build -t YOUR_DH_USERNAME/mangetamain:latest .
```

Run the container:

```bash
docker run --rm -p 8501:8501 YOUR_DH_USERNAME/mangetamain:latest
```

Notes
-----

- If your local dataset files are large you can mount them into the container
  at runtime instead of including them in the image (see Dockerfile comments).
