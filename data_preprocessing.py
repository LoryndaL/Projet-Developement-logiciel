import pandas as pd


# Chargement des données depuis les chemins absolus
train_data = pd.read_csv("/Volumes/PHILIPS UFD/BUT3/Ing_logiciel/PROJET/Projet_Titanic/train.csv")
test_data = pd.read_csv("/Volumes/PHILIPS UFD/BUT3/Ing_logiciel/PROJET/Projet_Titanic/test.csv")


def preprocess_data():
    """
    Prépare et traite les données pour l'entraînement du modèle

    Sélectionne les caractéristiques pertinentes, convertit les variables catégorielles en numériques
    et renvoie les ensembles de données prêts pour l'entraînement et l'évaluation.

    Returns:
        tuple: (X, X_test, y, passenger_ids)
        - X : Variables du jeu d'entraînement après transformation.
        - X_test : Variables du jeu de test après transformation.
        - train_data["Survived"] : Variable cible du jeu d'entraînement (Survived).
        - test_data["PassengerId"] : Identifiants des passagers du jeu de test.
    """
    # Sélection des colonnes pertinentes pour l'entraînement du modèle
    features = ["Pclass", "Sex", "SibSp", "Parch"]

    # Traitement des données, par exemple conversion des variables catégorielles en numériques
    X = pd.get_dummies(train_data[features])
    X_test = pd.get_dummies(test_data[features])

    return X, X_test, train_data["Survived"], test_data["PassengerId"]


if __name__ == "__main__":
    X, X_test, y, passenger_ids = preprocess_data()
    print(X.head())