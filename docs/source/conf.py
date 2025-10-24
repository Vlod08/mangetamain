# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'mangetamain'
copyright = '2025, Lina Hazime, Bryan LY, Khalil OUNIS, Mohammed ELAMINE, Lokeshwaran VENGADABADY'
author = 'Lina Hazime, Bryan LY, Khalil OUNIS, Mohammed ELAMINE, Lokeshwaran VENGADABADY'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = []

# If some heavy runtime-only dependencies (like streamlit) are not installed
# in the docs build environment, mock them so autodoc can import modules.
autodoc_mock_imports = ['streamlit']

# Enable autosummary so Sphinx can generate short API pages when autosummary
# directives are used or when running sphinx-apidoc + autosummary generation.
autosummary_generate = True

# Napoleon settings (use Google/NumPy style docstrings)
napoleon_google_docstring = False
napoleon_numpy_docstring = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_css_files = ['image.css']