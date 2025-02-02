"""
Script principal pour exécuter le projet Titanic.
L'utilisateur peut choisir l'étape à exécuter.
"""

import os
from data_preprocessing import load_data, preprocess_data
from model_training import train_model, save_model
from model_evaluation import load_model, make_predictions, save_predictions


def main():
    """Menu interactif pour exécuter le projet Titanic."""
    train_df, test_df = load_data("data/train.csv", "data/test.csv")
    X_train, X_test, y_train, passenger_ids = preprocess_data(train_df, test_df)

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
        elif choice == "3":
            if not os.path.exists("models/titanic_model.pkl"):
                print(" Modèle non trouvé. Entraînez-le d'abord.")
            else:
                model = load_model("models/titanic_model.pkl")
                predictions = make_predictions(model, X_test)
                save_predictions(predictions, passenger_ids)
        elif choice == "4":
            print(" Fin du programme.")
            break
        else:
            print("Option invalide. Réessayez.")


if __name__ == "__main__":
    main()
