# 🤖 Teachable Machine

## Application ML Interactive Sans Code

Une application web professionnelle permettant de créer, entraîner et évaluer des modèles de Machine Learning sans écrire une seule ligne de code. Interface intuitive en français avec des résultats professionnels.

---

## 📋 Table des matières

- [Fonctionnalités](#-fonctionnalités)
- [Installation](#-installation)
- [Types de données supportés](#-types-de-données-supportés)
- [Workflow en 4 étapes](#-workflow-en-4-étapes)
- [Algorithmes disponibles](#-algorithmes-disponibles)
- [Exemples d'utilisation](#-exemples-dutilisation)
- [Architecture technique](#-architecture-technique)
- [Auteur](#-auteur)

---

## ✨ Fonctionnalités

### 🎯 Principales

- **Interface No-Code** : Aucune compétence en programmation requise
- **Workflow guidé** : Processus en 4 étapes simples et intuitives
- **Données Tabulaires** : Support CSV et Excel (Classification & Régression)
- **Images** : Classification avec CNN personnalisable
- **Multi-Modèles** : Entraînement parallèle de plusieurs algorithmes
- **Visualisations avancées** : Graphiques interactifs Plotly
- **Export professionnel** : CSV, JSON et rapports détaillés

### 📊 Types de problèmes

- ✅ **Classification** (données tabulaires)
- ✅ **Régression** (données tabulaires)
- ✅ **Classification d'images** (CNN)

---

## 🚀 Installation

### Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)

### Installation des dépendances

```bash
pip install streamlit pandas numpy plotly scikit-learn tensorflow pillow openpyxl
```

### Fichier requirements.txt

```txt
streamlit
pandas
numpy
plotly
scikit-learn
tensorflow
pillow
openpyxl
```

---

## 🎮 Démarrage rapide

### Lancer l'application

```bash
streamlit run teachable_machine.py
```

L'application s'ouvrira automatiquement dans votre navigateur à l'adresse :
```
http://localhost:8501
```

### Première utilisation

1. Choisissez votre type de données (Tabulaires ou Images)
2. Uploadez votre fichier
3. Configurez et lancez le preprocessing
4. Sélectionnez et entraînez vos modèles
5. Analysez les résultats et exportez

---

## 📊 Types de données supportés

### 1. Données Tabulaires

#### Formats acceptés
- **CSV** (`.csv`)
- **Excel** (`.xlsx`, `.xls`)

#### Types de problèmes
- **Classification** : Prédire des catégories (Spam/Non-spam, Type de fleur, Risque médical)
- **Régression** : Prédire des valeurs numériques (Prix immobilier, Température, Score)

#### Preprocessing automatique
- 🔧 **StandardScaler** pour les features numériques
- 🎯 **OneHotEncoder** pour les features catégorielles
- 📊 **LabelEncoder** pour la target (classification)
- 📈 **Split 80/20** avec stratification (classification)

### 2. Images

#### Formats acceptés
- JPG/JPEG, PNG, BMP

#### Méthodes d'upload
**Archive ZIP (recommandé)** :
```
dataset.zip
├── classe1/
│   ├── image1.jpg
│   └── image2.jpg
├── classe2/
│   └── ...
└── classe3/
    └── ...
```

#### Preprocessing automatique
- 📐 **Redimensionnement** : 64x64, 128x128, 224x224, ou 256x256 pixels
- 🌈 **Normalisation** : Pixels de [0, 255] vers [0, 1]
- 🎯 **Encodage** : One-hot encoding des labels
- 📈 **Split 80/20** : Train/Test

---

## 🔄 Workflow en 4 étapes

### Étape 1️⃣ : Upload des données

**Données Tabulaires :**
- Uploadez votre fichier CSV ou Excel
- Visualisez l'aperçu des données
- Statistiques descriptives automatiques
- Analyse des types de données
- Sélection de la colonne cible
- Détection automatique du type de problème

**Images :**
- Uploadez une archive ZIP
- Configuration des classes
- Aperçu des images
- Analyse de la distribution

### Étape 2️⃣ : Configuration et Preprocessing

- Application automatique des transformations
- Validation des données
- Split train/test
- Résumé détaillé du preprocessing

### Étape 3️⃣ : Entraînement des Modèles

**Données Tabulaires :**
- Sélection des modèles à entraîner (jusqu'à 12)
- Entraînement parallèle
- Barre de progression en temps réel

**Images (CNN) :**
- Configuration : époques (5-100), batch size (8-128), learning rate
- Visualisation des courbes d'apprentissage

### Étape 4️⃣ : Résultats et Analyse

- Graphiques comparatifs interactifs
- Détection du meilleur modèle
- Métriques complètes
- Visualisations des prédictions
- Export CSV/JSON

---

## 🤖 Algorithmes disponibles

### Classification (12 modèles)

- 📊 **Logistic Regression** | **SGD Classifier**
- 🌳 **Decision Tree** | **Random Forest** | **Extra Trees** | **Gradient Boosting** | **AdaBoost**
- 🔍 **SVM (Linear)** | **SVM (RBF)**
- 📈 **Gaussian NB** | **KNN**
- 🧠 **MLP Classifier**

### Régression (12 modèles)

- 📊 **Linear Regression** | **Ridge** | **Lasso** | **SGD Regressor**
- 🌳 **Decision Tree** | **Random Forest** | **Extra Trees** | **Gradient Boosting**
- 🔍 **SVR (Linear)** | **SVR (RBF)**
- 📈 **KNN Regressor**
- 🧠 **MLP Regressor**

### Classification d'Images

- 🧠 **CNN personnalisé** :
  - Conv2D (32) + MaxPooling
  - Conv2D (64) + MaxPooling
  - Conv2D (64)
  - Dense (64) + Dropout (0.5)
  - Softmax

---

## 📚 Exemples d'utilisation

### Exemple 1 : Classification de maladies cardiaques

**Dataset** : `dataset_classification_cardiaque.csv`
- 500 patients, 9 features
- Target : maladie_cardiaque (Risque/Sain)
- Résultats attendus : Accuracy ~85%

### Exemple 2 : Prédiction de prix immobiliers

**Dataset** : `dataset_regression_immobilier.csv`
- 400 biens, 9 features
- Target : prix_euros
- Résultats attendus : R² ~0.92

### Exemple 3 : Classification de formes

**Dataset** : `dataset_images_formes.zip`
- 90 images (cercle, carré, triangle)
- Résultats attendus : Accuracy ~95%

### Exemple 4 : Classification d'animaux

**Dataset** : `dataset_images_animaux.zip`
- 75 images (chat, chien, oiseau)
- Résultats attendus : Accuracy ~88%

---

## 🏗️ Architecture technique

### Stack technologique

**Backend** :
- 🐍 Python 3.8+
- 📚 Scikit-learn (ML classique)
- 🧠 TensorFlow/Keras (Deep Learning)
- 📊 Pandas/Numpy (Data processing)

**Frontend** :
- 🎨 Streamlit (Framework web)
- 📈 Plotly (Visualisations)
- 🖼️ PIL (Traitement d'images)

### Structure

```
teachable-machine/
├── teachable_machine.py    # Application principale
├── README.md                # Documentation
├── requirements.txt         # Dépendances
└── datasets/                # Datasets d'exemple
    ├── dataset_classification_cardiaque.csv
    ├── dataset_regression_immobilier.csv
    ├── dataset_images_formes.zip
    └── dataset_images_animaux.zip
```

---

## 📊 Métriques d'évaluation

### Classification
- **Accuracy** : Précision globale
- **Precision** : Prédictions positives correctes
- **Recall** : Vrais positifs détectés
- **F1-Score** : Moyenne harmonique

### Régression
- **R² Score** : Coefficient de détermination
- **MSE** : Mean Squared Error
- **MAE** : Mean Absolute Error
- **RMSE** : Root Mean Squared Error

### CNN
- **Accuracy** / **Loss** : Métriques principales
- **Val Accuracy** / **Val Loss** : Sur validation
- **Courbes d'apprentissage** : Évolution

---

## 💡 Recommandations

### Pour améliorer les performances

**Données Tabulaires** :
- Optimisation des hyperparamètres (GridSearch)
- Feature engineering
- Équilibrage des classes
- Augmentation des données

**Images** :
- Tester différentes tailles
- Augmentation de données (rotation, flip, zoom)
- Transfer learning (VGG, ResNet)
- Plus d'époques et de données


---
### Idées d'amélioration

- [ ] XGBoost, LightGBM
- [ ] Transfer learning
- [ ] Sauvegarde de modèles
- [ ] Augmentation de données
- [ ] Optimisation auto hyperparamètres
- [ ] API REST


---

## 👨‍💻 Auteur

**Projet ISMAILI AMINA Elève Ingénieure IA & Data Science**
- 🎓 5ème année
- 📅 2024

---



</div>
