import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib  # Pour sauvegarder le modèle

# Chargement des données prétraitées (assurez-vous que le chemin de data_preprocessing est correct)
from data_preprocessing import preprocess_data

# Chargement des données et séparation en features et labels
X, X_test, y, passenger_ids = preprocess_data()

# Entraînement du modèle RandomForest
def train_model(X, y):
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    model.fit(X, y)
    return model

# Sauvegarde du modèle entraîné
def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")

if __name__ == "__main__":
    model = train_model(X, y)
    save_model(model, '/Volumes/PHILIPS UFD/BUT3/Ing_logiciel/PROJET/Projet_Titanic/titanic_model.pkl')
