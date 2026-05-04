# Titanic Survival Prediction

## Introduction
Ce projet présente une solution complète pour la compétition Kaggle "Titanic - Machine Learning from Disaster". L'objectif est de prédire la survie des passagers du Titanic en utilisant des techniques de Machine Learning, en se basant sur un ensemble de données historiques. Le pipeline de développement suit une approche rigoureuse, de l'analyse des données à l'optimisation des modèles, pour garantir une solution de qualité production.

## Score Obtenu
Notre modèle a atteint un score de **0.76555** sur le classement public de Kaggle, démontrant une bonne capacité de prédiction.

![Kaggle Submission Score](https://private-us-east-1.manuscdn.com/sessionFile/n4Jq9w5lIHuIoGmNkwrtpV/sandbox/UNOz0YXOgAZyDZC5u6EOgu-images_1777808096368_na1fn_L2hvbWUvdWJ1bnR1L3RpdGFuaWNfcHJvamVjdC9rYWdnbGVfc3VibWlzc2lvbl9zY29yZQ.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvbjRKcTl3NWxJSHVJb0dtTmt3cnRwVi9zYW5kYm94L1VOT3owWVhPZ0FaeURaQzV1NkVPZ3UtaW1hZ2VzXzE3Nzc4MDgwOTYzNjhfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzUnBkR0Z1YVdOZmNISnZhbVZqZEM5cllXZG5iR1ZmYzNWaWJXbHpjMmx2Ymw5elkyOXlaUS5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3OTg3NjE2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=MCXT93yZnRFSuIaEhgYiEIRA0DkeAN56I2xuLK0~TOa9fHoMaRegXQo-~-R~7xjSizKMn66as2HlNP4XKuX4b3qosKrU5QQeRAfUnjs2WONSJyOj~pKzaNdz~IDRJWT5MsS3ChWk8HWlQJXru9sGSDlSZje-N14nhh5dvu1SvggYeL87iQC94V11rTvRqL48f1tDSJVfuX3cLuNXaAYy67W~l1xsRd1J7DcfD0J1Adp-vRcT6Er4dz0siuTHE674g~V0eXH0hohbDXR3Wn~u7uP7J9vDy3saNh2~z4gqCIkSfhiOTIGmGaXIZZ~tYNRuedfY5VLFp7QRWlewYQlbTg__)
<img width="1000" height="800" alt="correlation_heatmap" src="https://github.com/user-attachments/assets/6bf3cb4e-d90c-4f5e-9fd4-fa9df1f8fea5" />
<img width="1500" height="1000" alt="eda_insights" src="https://github.com/user-attachments/assets/7afe2468-7d77-4cf5-9d3d-5b05e9a08d5b" />



## Structure du Projet
Le dépôt contient les fichiers suivants :
- `train.csv` et `test.csv` : Les ensembles de données originaux de la compétition.
- `titanic_full_solution.py` : Le script Python principal qui exécute l'intégralité du pipeline (chargement, prétraitement, ingénierie des features, entraînement, évaluation et prédiction).
- `submission.csv` : Le fichier de soumission généré, prêt à être téléchargé sur Kaggle.
- `eda_insights.png` : Visualisations clés de l'Analyse Exploratoire des Données (EDA).
- `correlation_heatmap.png` : Carte de chaleur des corrélations entre les features numériques.
- `best_model.pkl` : Le modèle entraîné et sérialisé.

## Méthodologie
Le projet a été développé en suivant un pipeline structuré :

### 1. Compréhension des Données
Chargement et inspection des fichiers `train.csv` et `test.csv`. Analyse des schémas, types de données, valeurs manquantes et distributions. Identification de la variable cible (`Survived`).

### 2. Nettoyage des Données
- **Valeurs manquantes** : Imputation de l'âge (`Age`) par la médiane basée sur le titre du passager. Imputation de l'embarquement (`Embarked`) par le mode et du tarif (`Fare`) par la médiane. La colonne `Cabin` a été jugée trop éparse et non utilisée.
- **Colonnes non pertinentes** : `PassengerId`, `Name`, `Ticket` et `Cabin` ont été gérées ou écartées après extraction des informations utiles.

### 3. Ingénierie des Features
- **Extraction du Titre** : Création d'une nouvelle feature `Title` à partir du nom (`Name`) (ex: Mr, Mrs, Miss). Les titres rares ont été regroupés.
- **Taille de la Famille** : Création de `FamilySize` en combinant `SibSp` (nombre de frères et sœurs/conjoints à bord) et `Parch` (nombre de parents/enfants à bord) + 1.
- **Voyage Seul** : Création de `IsAlone` pour indiquer si le passager voyageait seul.
- **Catégories d'Âge** : Binning de l'âge en catégories (Enfant, Adolescent, Adulte, Senior).
- **Encodage** : `Sex` encodé en binaire (0/1). `Embarked` et `Title` sont gérés par One-Hot Encoding au sein du pipeline de prétraitement.
- **Normalisation** : Les features numériques (`Age`, `Fare`, `FamilySize`) ont été standardisées.

### 4. Analyse Exploratoire des Données (EDA)
Analyse des taux de survie par sexe, classe de passager (`Pclass`), groupes d'âge et taille de la famille. Des visualisations (diagrammes à barres, cartes de chaleur) ont été générées pour extraire des insights clés.

### 5. Construction et Évaluation des Modèles
Plusieurs modèles ont été entraînés et comparés en utilisant la validation croisée (k-fold) :
- Régression Logistique
- Forêt Aléatoire (Random Forest)
- Gradient Boosting

Le modèle final est un **Ensemble de Vote (VotingClassifier)** combinant ces trois modèles, ce qui a permis d'améliorer la robustesse et la performance.

### 6. Optimisation des Modèles
Les hyperparamètres du modèle Random Forest ont été optimisés à l'aide de `GridSearchCV` pour trouver la meilleure combinaison de paramètres, évitant ainsi le surapprentissage.

### 7. Prédictions Finales
Le modèle ensemble final a été utilisé pour générer des prédictions sur l'ensemble de données `test.csv`, produisant le fichier `submission.csv` au format requis par Kaggle.

## Comment Exécuter le Code
1.  **Cloner le dépôt** :
    ```bash
    git clone https://github.com/flama-wolf/Titanic-Machine-learning.git
    cd Titanic-Machine-learning
    ```
2.  **Installer les dépendances** :
    ```bash
    pip install pandas scikit-learn matplotlib seaborn joblib
    ```
3.  **Exécuter le script principal** :
    ```bash
    python titanic_full_solution.py
    ```
    Cela générera le fichier `submission.csv` ainsi que les visualisations EDA.

## Améliorations Possibles
- **Feature Engineering Avancé** : Explorer le regroupement des tickets, l'analyse des cabines (si suffisamment de données), ou des features basées sur la localisation.
- **Modèles Avancés** : Intégrer des modèles comme XGBoost ou LightGBM avec une optimisation bayésienne pour un réglage plus fin des hyperparamètres.
- **Analyse d'Erreurs** : Examiner les erreurs de prédiction pour identifier les cas difficiles et améliorer le modèle spécifiquement sur ces segments.
- **Stacking** : Utiliser une approche de stacking plus complexe pour combiner les prédictions des modèles de base.
