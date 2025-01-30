import pandas as pd
import joblib
from data_preprocessing import preprocess_data

# Chargement des données prétraitées
X, X_test, y, passenger_ids = preprocess_data()

# Chargement du modèle sauvegardé
def load_model(model_path):
    return joblib.load(model_path)

# Faire des prédictions
def make_predictions(model, X_test):
    return model.predict(X_test)

# Sauvegarder les prédictions dans un fichier
def save_predictions(predictions, passenger_ids):
    output = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': predictions})
    output.to_csv('/Volumes/PHILIPS UFD/BUT3/Ing_logiciel/PROJET/Projet_Titanic/submission.csv', index=False)
    print("Predictions saved to 'submission.csv'")

if __name__ == "__main__":
    model = load_model('/Volumes/PHILIPS UFD/BUT3/Ing_logiciel/PROJET/Projet_Titanic/titanic_model.pkl')
    predictions = make_predictions(model, X_test)
    save_predictions(predictions, passenger_ids)
