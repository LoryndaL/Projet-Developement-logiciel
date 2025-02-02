import pandas as pd
import joblib
import os
from src.data_preprocessing import preprocess_data


# Définir les chemins relatifs des fichiers nécessaires
base_path = os.path.dirname(
    os.path.abspath(__file__)
)  # Répertoire du fichier actuel (model_evaluation.py)
model_path = os.path.join(
    base_path, "../titanic_model.pkl"
)  # Chemin relatif vers le modèle
submission_path = os.path.join(
    base_path, "../submission.csv"
)  # Chemin relatif pour sauvegarder les prédictions

# Chargement des données prétraitées
X, X_test, y, passenger_ids = preprocess_data()


# Chargement du modèle sauvegardé
def load_model(model_path):
    """
    Charge le modèle préalablement sauvegardé depuis un chemin spécifié.

    Args:
        model_path (str): Le chemin vers le fichier du modèle sauvegardé

    Returns:
        model: Le modèle chargé.
    """
    return joblib.load(model_path)


# Faire des prédictions
def make_predictions(model, X_test):
    """
    Effectue des prédictions sur les données de test à l'aide du modèle fourni.

    Args:
        model: Le modèle entraîné utilisé pour prédire.
        X_test (pd.DataFrame): Données de test sur lesquelles faire des prédictions.

    Returns:
        X_test : Un tableau des prédictions effectuées sur les données de test.
    """
    return model.predict(X_test)


# Sauvegarder les prédictions dans un fichier
def save_predictions(predictions, passenger_ids):
    """
    Sauvegarde les prédictions dans un fichier CSV.

    Args:
        predictions (Array of int64): Les prédictions faites pour chaque passager.
        passenger_ids (pd.Series): Les identifiants des passagers correspondants aux prédictions.

    Enregistre les résultats dans un fichier CSV nommé 'submission.csv' dans le répertoire spécifié.
    """
    output = pd.DataFrame({"PassengerId": passenger_ids, "Survived": predictions})
    output.to_csv(submission_path, index=False)  # Utilisation du chemin relatif
    print(f"Predictions saved to '{submission_path}'")


if __name__ == "__main__":
    model = load_model(
        model_path
    )  # Utilisation du chemin relatif pour charger le modèle
    predictions = make_predictions(model, X_test)
    save_predictions(predictions, passenger_ids)
