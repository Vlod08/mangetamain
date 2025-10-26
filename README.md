# Mangetamain - Food.com Recipe Recommendation System

**Mangetamain** is a *web application* developed as part of the **Kit Big Data** project at [Telecom Paris](https://www.telecom-paris.fr/).

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

## Setup Instructions

### Prerequisites
- Python 3.11 
- [Poetry 2.2.0](https://python-poetry.org/docs/#installation) for dependency management

### Installation & Setup

1. **Install Poetry:** (if not already installed)

  Install Poetry using the official installation script:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 - --version 2.2.0
   ```

  Add Poetry to your `PATH`:
  * For Unix/MacOS, add the following line to your shell profile (e.g., `~/.bashrc`, `~/.zshrc`):
   ```bash
   export PATH="$HOME/.local/bin:$PATH"
   ```
  * For Windows, add `%USERPROFILE%\.local\bin` to your system `PATH` environment variable.
   ```bash
   export PATH="%APPDATA%\pypoetry\venv\Scripts\poetry:$PATH"
   ```
  
  Restart your terminal or run `source ~/.bashrc` (or equivalent for your shell) to apply the changes.

2. **Clone the repository:**
   ```bash
   git clone https://github.com/Vlod08/mangetamain.git
   cd mangetamain
   ```

3. **Run the setup script:**
   ```bash
   ./setup.sh
   ```

  This will:
  - Check for Python 3.11
  - Check for Poetry 2.2.0 and install if not already installed
  - Set up a virtual environment
  - Install all dependencies

## Dataset

In this project, we use the Food.com dataset for recipe recommendations. For more information, visit [Food.com Dataset on Kaggle](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions).


## Getting Started

- **Launch the Streamlit application:**
  ```bash
  poetry run streamlit run src/mangetamain/main.py
  ```
  
- **Alternative activation methods:**
  Activate the virtual environment by running:
  ```bash
  poetry shell
  ```

  or  
  ```bash 
  # (Windows)
  source .venv/Scripts/activate
  ```

  or
  ```bash
  # (Unix/MacOS)
  source .venv/bin/activate
  ```

  Then run:
   ```bash
  streamlit run src/mangetamain/main.py
  ```

## Configuration

The project uses several configuration files:
- `pyproject.toml`: Project metadata and dependencies
- `poetry.toml`: Poetry-specific configuration

--- 