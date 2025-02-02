import os
import sys
import joblib
from sklearn.ensemble import RandomForestClassifier

# Ajouter le répertoire src au chemin de recherche des modules Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Chargement des données prétraitées
from src.data_preprocessing import preprocess_data  # noqa: E402

# Définir le chemin relatif pour sauvegarder le modèle
base_path = os.path.dirname(os.path.abspath(__file__))  # Répertoire du fichier actuel
model_path = os.path.join(base_path, "../models/titanic_model.pkl")  # Chemin vers le modèle

# Chargement des données et séparation en features et labels
X, X_test, y, passenger_ids = preprocess_data()


def train_model(X, y):
    """
    Entraîne un modèle RandomForestClassifier sur les données fournies.

    Args:
        X (pd.DataFrame): Les variables du jeu d'entraînement.
        y (pd.Series): La variable cible (Survived).

    Returns:
        RandomForestClassifier: Modèle entraîné.
    """
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    model.fit(X, y)
    return model


def save_model(model, filename):
    """
    Sauvegarde le modèle entraîné dans un fichier.

    Args:
        model (RandomForestClassifier): Le modèle entraîné.
        filename (str): Le chemin du fichier où sauvegarder le modèle.
    """
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")


if __name__ == "__main__":
    model = train_model(X, y)  # Entraînement du modèle
    save_model(model, model_path)  # Sauvegarde du modèle
