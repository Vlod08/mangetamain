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

Follow these steps to install and set up the project locally:

```bash
git clone https://github.com/Vlod08/mangetamain.git
cd mangetamain
./setup.sh
poetry install
```

> 💡 For detailed instructions (including environment setup, dependencies, and troubleshooting), please refer to the [Installation Guide](./guides/INSTALLATION.md).

<hr style="height:3px;border-width:0;color:gray;background-color:gray">

## Getting Started

### Run the streamlit application

Launch the app directly with Poetry:
  ```bash
  poetry run streamlit run src/mangetamain/main.py
  ```

### Run with Docker

Alternatively, you can run the application in a Docker container:

  1) Pull the latest image:
  ```bash
  docker pull vlod08/mangetamain
  ```
  2) Start the container and expose port 8501:
  ```bash
  docker run -p 8501:8501 vlod08/mangetamain 
  ```
  3) Access the app locally:
  http://localhost:8501/ 

<hr style="height:3px;border-width:0;color:gray;background-color:gray">

## Dataset

In this project, we use the Food.com dataset for recipe recommendations.

For more information, visit [Food.com Dataset on Kaggle](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions).

<hr style="height:3px;border-width:0;color:gray;background-color:gray">

## Configuration

The project uses several configuration files:
- `pyproject.toml`: Project metadata and dependencies
- `poetry.toml`: Poetry-specific configuration

<hr style="height:3px;border-width:0;color:gray;background-color:gray">

## License

This project is licensed under the MIT License – see the [LICENSE](./LICENSE) file for details.

Feel free to use, modify, and distribute under the same terms.