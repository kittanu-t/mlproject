#!/usr/bin/env bash
set -euo pipefail

# Usage: ./init_mlops_project.sh  (from repo root)

# Create directories
mkdir -p .github/workflows
mkdir -p mlops_pipeline/scripts
mkdir -p mlruns
mkdir -p processed_data

# Create placeholder workflow
cat > .github/workflows/main.yml <<'YAML'
name: CI

on:
  push:
  pull_request:

jobs:
  lint-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install deps
        run: |
          python -m pip install --upgrade pip
          pip install -r mlops_pipeline/requirements.txt
      - name: Lint
        run: flake8 .
      - name: Test
        run: pytest -q
YAML

# Create pipeline script stubs
for f in 01_data_validation.py 02_data_preprocessing.py 03_train_evaluate_register.py 04_transition_model.py 05_api_service.py; do
  cat > "mlops_pipeline/scripts/$f" <<'PY'
#!/usr/bin/env python
"""
Stub file. Replace with implementation.
"""
if __name__ == "__main__":
    print("TODO: implement")
PY
  chmod +x "mlops_pipeline/scripts/$f"
done

# Create requirements (you’ll overwrite with the content below if needed)
cat > mlops_pipeline/requirements.txt <<'REQ'
mlflow
scikit-learn
pandas
nltk
flask
gunicorn
flake8
pytest
REQ

# Create a standard Python .gitignore
cat > .gitignore <<'GIT'
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
Pipfile.lock

# poetry
poetry.lock

# PEP 582; used by e.g. github.com/David-OConnor/pyflow
__pypackages__/

# Celery
celerybeat-schedule
celerybeat.pid

# SageMath
*.sage.py

# Environments
.spyproject
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# pytype
.pytype/

# VSCode / IDE
.vscode/
.idea/

# ML & data artifacts
mlruns/
processed_data/
*.model
*.pkl
*.joblib

# OS files
.DS_Store
Thumbs.db
GIT

echo "✅ Project skeleton created."
