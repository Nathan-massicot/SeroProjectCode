# 🧠 SeroProjectCode – Data Analysis Pipeline

Projet d’analyse de données pour une application de prévention santé (planification et suivi d’utilisateurs, aide à la dépression).  
Les données sont fournies sous forme de fichiers CSV (non versionnés dans ce repo).

---

# 📦 Initialisation du projet

### 1. Prérequis
- Python **3.13+**
- [Poetry 2.x](https://python-poetry.org/docs/#installation) (idéalement via `UV`)

Vérifier l’installation :
```bash
python3 --version
poetry --version

#Clone repo 

git clone <URL_DU_REPO>.git
cd <PathYourProject>

#Config env with poetry
uv .venv --python 3.12 
uv activate sources/.venv
uv sync 

#Project structure 

SeroProjectCode/
│── data/                # Données brutes (non versionnées)
│── data_sample/         # Exemples anonymisés
│── notebooks/           # Notebooks exploratoires
│── src/                 # Code source (pipeline, analyse)
│── tests/               # Tests unitaires
│── README.md            # Ce document
│── pyproject.toml       # Définition de l'environnement Poetry
│── .gitignore           # Exclusions (data/, venv/, etc.)

## Regression validation report

Automated regression robustness checks are available via:

```bash
uv run python src/validate_sentiment_regression.py
```

Useful options:

```bash
# Faster run on a subset
uv run python src/validate_sentiment_regression.py --max-submissions 200 --perm-iterations 50 --bootstrap-iterations 100

# If you already have a scored table (distance + sentiment_score)
uv run python src/validate_sentiment_regression.py --scored-csv path/to/scored.csv
```

Outputs:
- `reports/sentiment_validation_report.md`
- `reports/sentiment_validation_report.json`
