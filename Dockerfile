# 1. Utiliser une image de base officielle Python
FROM python:3.9-slim

# 2. Définir le répertoire de travail dans le conteneur
WORKDIR /app

# 3. Copier les fichiers nécessaires depuis ton projet dans le conteneur
COPY . /app

# 4. Installer les dépendances de ton projet
RUN pip install --no-cache-dir -r requirements.txt

# 5. Exposer un port (optionnel, pour les applications web)
EXPOSE 5000

# 6. Commande pour lancer ton application (par exemple un fichier app.py)
CMD ["python", "app.py"]