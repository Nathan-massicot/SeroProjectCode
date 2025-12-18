# 🧠 SeroProjectCode – Data Analysis Pipeline

Projet d’analyse de données pour une application de prévention santé (planification et suivi d’utilisateurs, aide à la dépression).  
Les données sont fournies sous forme de fichiers CSV (non versionnés dans ce repo).

---

# 📦 Initialisation du projet

### 1. Prérequis
- Python **3.12+**
- [Poetry 2.x](https://python-poetry.org/docs/#installation) (idéalement via `UV`)

Vérifier l’installation :
```bash
python3 --version
poetry --version

#Clone repo 

git clone <URL_DU_REPO>.git
cd SeroProjectCode

#Config env with poetry

poetry config virtualenvs.in-project true
poetry env use python3.12
poetry install


#Project structure 

SeroProjectCode/
│── data/                # Données brutes (non versionnées)
│── data_sample/         # Exemples anonymisés
│── notebooks/           # Notebooks exploratoires
│── src/                 # Code source (pipeline, analyse)
│── tests/               # Tests unitaires
│── README.md            # Ce document
│── pyproject.toml       # Définition de l'environnement Poetry
│── poetry.lock          # Versions figées des dépendances
│── .gitignore           # Exclusions (data/, venv/, etc.)