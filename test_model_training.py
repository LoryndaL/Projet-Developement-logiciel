import pytest
import joblib
import os
from model_training import train_model, save_model
from data_preprocessing import preprocess_data

def test_train_model():
    """Test l'entraînement du modèle."""
    X, X_test, y, passenger_ids = preprocess_data()
    model = train_model(X, y)
    
    assert model is not None, "Le modèle doit être entraîné"
    assert hasattr(model, 'predict'), "Le modèle doit avoir une méthode 'predict'"
    assert model.n_estimators == 100, "Le modèle doit avoir 100 arbres"
    assert model.max_depth == 5, "Le modèle doit avoir une profondeur max de 5"

def test_save_model():
    """Test la sauvegarde et le chargement du modèle."""
    X, X_test, y, passenger_ids = preprocess_data()
    model = train_model(X, y)
    
    model_path = 'titanic_model_test.pkl'
    save_model(model, model_path)
    
    assert os.path.exists(model_path), "Le fichier du modèle doit être sauvegardé"
    
    # Vérifier le rechargement du modèle
    loaded_model = joblib.load(model_path)
    assert hasattr(loaded_model, 'predict'), "Le modèle chargé doit avoir une méthode 'predict'"

    # Supprimer le fichier après le test
    os.remove(model_path)

def test_model_prediction():
    """Test que le modèle génère des prédictions valides."""
    X, X_test, y, passenger_ids = preprocess_data()
    model = train_model(X, y)
    
    predictions = model.predict(X_test)
    
    assert len(predictions) == len(X_test), "Le modèle doit générer une prédiction pour chaque échantillon"
    assert set(predictions).issubset({0, 1}), "Les prédictions doivent être binaires (0 ou 1)"
