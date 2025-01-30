# Projet-Developement-logiciel

## Description du projet
Le projet Titanic Survival Prediction consiste à prédire les chances de survie des passagers du Titanic en utilisant des données historiques. Les données sont traitées et un modèle de machine learning (ML) est construit pour effectuer les prédictions. Le projet initial est disponible sur Kaggle sous forme de notebook, et notre mission consiste à refactoriser ce code en scripts Python modulaires et réutilisables en suivant les bonnes pratiques d’ingénierie logicielle.

## Étapes du Projet
### 1. Refactorisation du Code
Le code a été refactorisé en plusieurs modules Python :
- **data_preprocessing.py** : Chargement, nettoyage et transformation des données.
- **model_training.py** : Entraînement du modèle de prédiction et sauvegarde.
- **model_evaluation.py** : Évaluation du modèle et analyse des performances.

### 2. Tests Unitaires
Des tests unitaires ont été ajoutés pour assurer la qualité du code. Ces tests sont regroupés dans le dossier `tests/`. Nous utilisons **pytest** pour les tests.

### 3. Documentation
Le code est bien documenté à l'aide de **docstrings** dans chaque fonction et module. Le fichier `README.md` fournit des informations générales sur le projet, les étapes de l'exécution, ainsi que des instructions d'installation et d'utilisation.

### 4. Gestion de Version avec Git et GitHub
Nous utilisons **Git** et **GitHub** pour le suivi des versions et la collaboration en équipe. Le flux de travail Git suit une stratégie de branches avec `main`, `develop`, et des branches de fonctionnalités.
