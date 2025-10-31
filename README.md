# 🍽️ Mangetamain

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Poetry](https://img.shields.io/badge/packaging-poetry-60A5FA.svg)](https://python-poetry.org/)
[![Streamlit](https://img.shields.io/badge/framework-streamlit-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

<table>
<tr>
<td>

## Project Overview

**Mangetamain** is a data-driven application designed to visualize, analyze, and interpret food-related information using:

- **Machine Learning**
- **Data Visualization**
- **Interactive Dashboards** (built with **Streamlit** and **Plotly**)

This project provides a unified platform for exploring datasets, generating predictions, and offering insightful analytics.

</td>
<td>

<img src="assets/mangetamain-logo.jpg" alt="App logo" width="300" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">

</td>
</tr>
</table>

<hr style="height:3px;border-width:0;color:gray;background-color:gray">

## Authors

All authors have contributed equally to this project, which was developed as part of the **Master Spécialisé en Intelligence Artificielle** ([Expert Data & MLops](https://www.telecom-paris.fr/fr/masteres-specialises/formation-big-data) & [IA multimodale et autonome](https://www.telecom-paris.fr/fr/masteres-specialises/formation-intelligence-artificielle)) at [Télécom Paris](https://www.telecom-paris.fr/).

| Name | Email |
|------|--------|
| **Bryan LY** | bryan29.ly@gmail.com |
| **Khalil OUNIS** | khalilounis10@gmail.com |
| **Mohammed ELAMINE** | elamine.mohammed.14@gmail.com |
| **Lokeshwaran VENGADABADY** | lokeshvengadabady@gmail.com |
| **Lina RHIATI HAZIME** | lina.rhiati2@gmail.com |

<hr style="height:3px;border-width:0;color:gray;background-color:gray">

## Tech Stack

- **Python 3.11**
- **Streamlit** – Interactive dashboarding
- **Plotly** – Advanced data visualization
- **Scikit-learn** – Machine learning toolkit
- **Seaborn** – Statistical data visualization
- **Sphinx** – Documentation engine

<hr style="height:3px;border-width:0;color:gray;background-color:gray">

## Project Structure

```
mangetamain/
├── data/                       # Dataset storage
│   ├── processed/              # Processed data (pickle files)
│   └── raw/                    # Original downloaded data (pickle files)
├── src/
│   ├── mangetamain/
│   │   ├── app/
│   │   │   ├── app_utils/
│   │   │   ├── pages/
│   │   │   └── __init__.py         
│   │   ├── core/               
│   │   ├── preprocessing/      # Data preprocessing
│   │   ├── main.py             # Streamlit application
│   │   └── __init__.py
│   └── scripts/                # Utility scripts
├── tests/
│   └── __init__.py
├── docs/                       # Documentation
├── image/
├── pyproject.toml              # Project dependencies
├── poetry.lock
├── poetry.toml
├── .gitignore
├── setup.sh                    # Setup script
└── README.md
```

<hr style="height:3px;border-width:0;color:gray;background-color:gray">

## Project Setup Guide — mangetamain

This guide explains how to install and use Poetry to manage the dependencies, environments, and development workflow of this project.

### 1. Install Poetry

You must have **Python ≥ 3.11** and **< 3.12** installed.

Check your Python version:
```
python3 --version
```

If it’s correct, install [Poetry 2.2.0](https://python-poetry.org/docs/#installation) globally:
```
curl -sSL https://install.python-poetry.org | python3 - --version 2.2.0
```

Then make sure Poetry is available in your path:
```
poetry --version
```

If not found, add it manually (depending on your system):
```
export PATH="$HOME/.local/bin:$PATH"
```

### 2. Clone the Project
```
git clone https://github.com/Vlod08/mangetamain.git
cd mangetamain
```

### 3. Run the setup script:
```bash
./setup.sh
```

This will:
- Check for Python 3.11
- Check for Poetry 2.2.0
- Set up a virtual environment

### 4. Install Dependencies

All project dependencies are declared in `pyproject.toml`.

To install everything (main + dev dependencies):
```
poetry install
```

If you only want runtime dependencies (without dev tools like Sphinx or pytest):
```
poetry install --without dev
```

### 5. Use the Virtual Environment

Poetry automatically creates a virtual environment.

To get basic information about the currently activated virtual environment, you can use the `env info` command:
```
poetry env info
```

The `poetry env activate` command prints the activate command of the virtual environment to the console. You can run the output command manually or feed it to the eval command of your shell to activate the environment.
```
eval $(poetry env activate)
```

Now, any Python or Streamlit commands you run will use this environment.

*Alternatively*, you can run commands without entering the shell:
```
poetry run python src/mangetamain/app.py
```

### 6. Running Tests

Tests are located in the `tests/ directory`.
You can run them with:
```
poetry run pytest
```

If you have already activated the environment in your current shell:
```
pytest
```

### 7. Documentation (optional)

If you want to build the Sphinx documentation locally:
```
poetry run sphinx-build -b html docs/ docs/_build/
```

### 8. Adding or Updating Dependencies

To add a new library:
```
poetry add numpy
```

To add a development dependency (like black or pytest):
```
poetry add --group dev black
```

To update dependencies:
```
poetry update
```

### 9. Export Requirements (for non-Poetry users)

If someone prefers using plain pip, they can generate a requirements.txt:
```
poetry export -f requirements.txt --output requirements.txt
```

Then install using:
```
pip install -r requirements.txt
```

### 10. Troubleshooting

Poetry environment not found?
```
poetry env info
poetry env remove python
poetry install
```

To check all available environments:
```
poetry env list
```

To clear all Poetry caches (rarely needed):
```
poetry cache clear pypi --all
```

<hr style="height:3px;border-width:0;color:gray;background-color:gray">

## Best Practices for Contributors

* Always run `poetry install` after pulling new changes.

* Never manually edit `poetry.lock`.

* Use `poetry add` to install new dependencies.

* Commit both `pyproject.toml` and `poetry.lock`.

* Before pushing, test locally with:
```
poetry run pytest
```

<hr style="height:3px;border-width:0;color:gray;background-color:gray">

## Dataset

In this project, we use the Food.com dataset for recipe recommendations.

For more information, visit [Food.com Dataset on Kaggle](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions).

<hr style="height:3px;border-width:0;color:gray;background-color:gray">

## Getting Started

Launch the Streamlit application:
  ```bash
  poetry run streamlit run src/mangetamain/main.py
  ```

Set up and launch via Docker:
  1) Download the app image locally:
  ```bash
  docker pull vlod08/mangetamain
  ```
  2) Run container and expose port 8501:
  ```bash
  docker run -p 8501:8501 vlod08/mangetamain 
  ```
  3) You can check the app by following the link:
  http://localhost:8501/ 
<hr style="height:3px;border-width:0;color:gray;background-color:gray">

## Configuration

The project uses several configuration files:
- `pyproject.toml`: Project metadata and dependencies
- `poetry.toml`: Poetry-specific configuration

<hr style="height:3px;border-width:0;color:gray;background-color:gray">

## License

This project is licensed under the MIT License – see the [LICENSE](./LICENSE) file for details.

Feel free to use, modify, and distribute under the same terms.