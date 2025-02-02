import pytest
import pandas as pd
import sys
import os

# Ajouter le dossier src au PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Importer le module après avoir ajouté src au chemin
from data_preprocessing import preprocess_data

def test_preprocess_data():
    # Appel de la fonction de prétraitement
    X, X_test, y, passenger_ids = preprocess_data()
    
    # Vérifier que les données sont bien chargées
    assert isinstance(X, pd.DataFrame), "X doit être un DataFrame pandas"
    assert isinstance(X_test, pd.DataFrame), "X_test doit être un DataFrame pandas"
    assert isinstance(y, pd.Series), "y doit être une Series pandas"
    assert isinstance(passenger_ids, pd.Series), "passenger_ids doit être une Series pandas"
    
    # Vérifier la présence des colonnes dans X
    expected_columns = ['Pclass', 'SibSp', 'Parch', 'Sex_female', 'Sex_male']
    for column in expected_columns:
        assert column in X.columns, f"Colonne {column} pas trouvée dans X"

    # Vérifier la taille des données pour s'assurer qu'elles ne sont pas vides
    assert len(X) > 0, "X ne doit pas être vide"
    assert len(X_test) > 0, "X_test ne doit pas être vide"
    assert len(y) > 0, "y ne doit pas être vide"
    assert len(passenger_ids) > 0, "passenger_ids ne doit pas être vide"
