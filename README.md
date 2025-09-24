# Mangetamain

Mangetamain is a web application developed as part of the Kit Big Data project at Telecom Paris.

## Project Structure

```
mangetamain/
├── src/
│   └── mangetamain/
│       └── __init__.py
│       └── main.py
├── tests/
│   └── __init__.py
├── pyproject.toml
├── poetry.lock
├── poetry.toml
├── .env
├── .gitignore
├── setup.sh
├── README.md
└── .venv/
```

## Getting Started

### Prerequisites
- Python 3.11 (see `.env` for the expected path)
- [Poetry](https://python-poetry.org/docs/#installation)

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Vlod08/mangetamain.git
   cd mangetamain
   ```
2. Configure your Python path in `.env`.
3. Run the setup script:
   ```bash
   ./setup.sh
   ```
   This will:
   - Check for Poetry and Python 3.11
   - Set up a virtual environment in the project root directory
   - Install all dependencies

### Activating the Environment
- To activate the virtual environment:
  ```bash
  poetry env activate
  # or
  source .venv/Scripts/activate
  ```
- To run commands inside the environment:
  ```bash
  poetry run python src/mangetamain/main.py
  ```

## Development
- Add dependencies with:
  ```bash
  poetry add <package>
  ```
- Run tests:
  ```bash
  poetry run pytest
  ```

---
