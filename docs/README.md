# Titanic Survival Prediction

## Description du projet

Le projet **Titanic Survival Prediction** vise à prédire les chances de survie des passagers du Titanic en utilisant des données historiques. Il repose sur la refactorisation d'un notebook existant disponible sur Kaggle en scripts Python modulaires et réutilisables.

L'objectif est d'appliquer les bonnes pratiques d'ingénierie logicielle, y compris la structuration du code, l'écriture de tests unitaires, l'utilisation de Git/GitHub pour la collaboration et la mise en place d'un pipeline CI/CD.

## Structure du projet
Le projet est organisé en plusieurs modules Python :

- `src/data_preprocessing.py` : Chargement, nettoyage et transformation des données.
- `src/model_training.py` : Entraînment et sauvegarde du modèle de prédiction.
- `src/model_evaluation.py` : Évaluation des performances du modèle.
- `tests/` : Contient les tests unitaires pour vérifier le bon fonctionnement du code.
- `docs/` : Documentation du projet.
- `requirements.txt` : Liste des dépendances du projet.
- `.github/workflows/` : Configuration du pipeline CI/CD avec GitHub Actions.

## Prérequis
Avant d'exécuter le projet, assurez-vous d'avoir installé :

- **Python 3.8+**
- **Git**
- **Poetry** (ou pip pour la gestion des dépendances)

## Installation

1. **Cloner le dépôt GitHub**
   ```bash
   git clone https://github.com/LoryndaL/Projet-Developement-logiciel
   cd titanic-survival-prediction
   ```

2. **Créer un environnement virtuel et installer les dépendances**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sur Windows : venv\Scripts\activate
   pip install -r requirements.txt
   ```

   *Ou avec Poetry :*
   ```bash
   poetry install
   ```

3. **Exécuter le pipeline complet**
   ```bash
   python src/data_preprocessing.py
   python src/model_training.py
   python src/model_evaluation.py
   ```

## Tests
Les tests unitaires sont définis dans le dossier `tests/`. Pour les exécuter :

```bash
pytest tests/
```

## Gestion de version et collaboration
Nous utilisons **Git** et **GitHub** avec la stratégie de branches suivante :
- `main` : Branche stable avec les versions validées du projet.
- `develop` : Branche de développement intégrant les nouvelles fonctionnalités.
- `feature/xxx` : Branches individuelles pour chaque fonctionnalité.

## CI/CD avec GitHub Actions
Nous avons mis en place une intégration et déploiement continus (CI/CD) pour assurer la qualité du code.

- **Linting** : `flake8` et `black` pour vérifier et formater le code.
- **Tests** : `pytest` exécuté automatiquement sur chaque commit.


## Contributions
L'équipe du projet :
- **EL YAOUTI AYA** (Responsable Data Processing)
- **LOUFOUA LORYNDA** (Responsable Modélisation)
- **N'DIAYE MARIAM** (Responsable CI/CD)
- **SOUMAHORO MAXIMILIEN** (Responsable Documentation)


**Date limite : Lundi 3 Février 2025**

---
Projet réalisé dans le cadre du BUT VCOD (IUT Paris Cité) 2024-2025.

