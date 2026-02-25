# Guide d'utilisation de Poetry

Ce projet utilise [Poetry](https://python-poetry.org/) pour la gestion des dépendances et de l'environnement virtuel, garantissant une reproductibilité totale.

## 1. Installation de Poetry
Si Poetry n'est pas encore installé sur votre machine, vous pouvez l'installer via la commande suivante :
`curl -sSL https://install.python-poetry.org | python3 -`
OU 
`pip install poetry`
Assurez-vous que Poetry est correctement installé en vérifiant la version :
`poetry --version`

## 2. Installation des dépendances du projet
À la racine du projet (là où se trouve le fichier `pyproject.toml`), exécutez simplement :
`poetry install`
Cette commande va lire le fichier `poetry.lock` et installer les versions exactes de toutes les bibliothèques nécessaires (PyTorch, Transformers, ffmpeg, etc.).

## 3. Exécuter le code ou le Notebook
Pour exécuter une commande Python dans l'environnement virtuel créé par Poetry :
`poetry run python src/mon_script.py`

Pour lancer Jupyter Notebook et tester l'exécution de bout en bout (`test.ipynb`) :
`poetry run jupyter notebook`

*Note : Alternativement, vous pouvez activer le shell de l'environnement virtuel avec `poetry shell`.*