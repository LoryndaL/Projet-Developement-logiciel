import os
import sys
import pandas as pd
import joblib

# Ajouter le répertoire src au chemin de recherche des modules Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Chargement des données prétraitées
from src.data_preprocessing import preprocess_data  # noqa: E402

# Définir les chemins relatifs des fichiers nécessaires
base_path = os.path.dirname(os.path.abspath(__file__))  # Répertoire du fichier actuel
model_path = os.path.join(base_path, "../models/titanic_model.pkl")  # Modèle sauvegardé
submission_path = os.path.join(
    base_path, "../data/submission.csv"
)  # Fichier de soumission

# Chargement des données prétraitées
X, X_test, y, passenger_ids = preprocess_data()


def load_model(model_path):
    """
    Charge le modèle préalablement sauvegardé depuis un chemin spécifié.

    Args:
        model_path (str): Le chemin vers le fichier du modèle sauvegardé.

    Returns:
        model: Le modèle chargé.
    """
    return joblib.load(model_path)


def make_predictions(model, X_test):
    """
    Effectue des prédictions sur les données de test à l'aide du modèle fourni.

    Args:
        model: Le modèle entraîné utilisé pour prédire.
        X_test (pd.DataFrame): Données de test sur lesquelles faire des prédictions.

    Returns:
        np.ndarray: Un tableau des prédictions effectuées sur les données de test.
    """
    return model.predict(X_test)


def save_predictions(predictions, passenger_ids):
    """
    Sauvegarde les prédictions dans un fichier CSV.

    Args:
        predictions (np.ndarray): Les prédictions faites pour chaque passager.
        passenger_ids (pd.Series): Les identifiants des passagers correspondants aux prédictions.

    Enregistre les résultats dans un fichier CSV nommé 'submission.csv'.
    """
    output = pd.DataFrame({"PassengerId": passenger_ids, "Survived": predictions})
    output.to_csv(submission_path, index=False)
    print(f"Predictions saved to '{submission_path}'")


if __name__ == "__main__":
    model = load_model(model_path)  # Chargement du modèle
    predictions = make_predictions(model, X_test)  # Prédictions
    save_predictions(predictions, passenger_ids)  # Sauvegarde des résultats
