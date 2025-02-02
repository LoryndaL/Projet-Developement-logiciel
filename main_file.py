
import os
import sys
import pandas as pd
from src.data_preprocessing import preprocess_data
from src.model_training import train_model, save_model
from src.model_evaluation import load_model, make_predictions, save_predictions

def main():
    """Menu interactif pour exécuter le projet Titanic."""
    # Ajouter le répertoire src au chemin de recherche des modules Python
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

    # Chargement des données
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    # Prétraitement des données
    X_train, X_test, y_train, passenger_ids = preprocess_data()

    while True:
        print("\nProjet Titanic - Choisissez une action :")
        print("1️ Prétraiter les données")
        print("2️ Entraîner le modèle")
        print("3️ Évaluer le modèle")
        print("4️ Quitter")

        choice = input("Entrez le numéro de l'action : ")

        if choice == "1":
            print("Données prétraitées avec succès !")
        elif choice == "2":
            model = train_model(X_train, y_train)
            save_model(model, "models/titanic_model.pkl")
            print("Modèle entraîné et sauvegardé avec succès !")
        elif choice == "3":
            if not os.path.exists("models/titanic_model.pkl"):
                print("Modèle non trouvé. Entraînez-le d'abord.")
            else:
                model = load_model("models/titanic_model.pkl")
                predictions = make_predictions(model, X_test)
                save_predictions(predictions, passenger_ids)
                print("Prédictions effectuées et sauvegardées avec succès !")
        elif choice == "4":
            print("Fin du programme.")
            break
        else:
            print("Option invalide. Réessayez.")

if __name__ == "__main__":
    main()
