#  Teachable Machine 

Une application web interactive pour entraîner des modèles de Machine Learning avec une interface dark mode moderne.

##  Fonctionnalités

###  Types de Données Supportés
- **Données tabulaires** (CSV, Excel)
- **Images** (JPG, PNG, BMP, TIFF)
- **Capture caméra** (simulation et upload)

###  Types de Problèmes
- **Classification** (Catégorielle)
- **Régression** (Numérique)

###  Algorithmes Implémentés

#### Classification
- **Linear Models** : Logistic Regression, SGD Classifier, Perceptron
- **Tree Based** : Decision Tree, Random Forest, XGBoost, LightGBM
- **SVM** : SVC (Linear, RBF, Poly), Nu-SVC
- **Naive Bayes** : Gaussian, Multinomial, Bernoulli
- **Neighbors** : KNN, Radius Neighbors
- **Neural Networks** : MLP, Simple NN, Deep NN, CNN

#### Régression
- **Linear Models** : Linear Regression, Ridge, Lasso, Elastic Net
- **Tree Based** : Decision Tree, Random Forest, XGBoost, LightGBM
- **SVM** : SVR (Linear, RBF, Poly)
- **Neighbors** : KNN Regressor, Radius Neighbors
- **Neural Networks** : MLP Regressor, Simple NN, Deep NN

##  Technologies Utilisées

- **Streamlit** - Interface web interactive
- **Pandas** - Manipulation des données
- **Plotly** - Visualisations interactives
- **Scikit-learn** - Algorithmes de Machine Learning
- **XGBoost & LightGBM** - Algorithmes boosting
- **Pillow** - Traitement d'images

##  Processus d'Utilisation

### Étape 1: Upload des Données
- **Choisissez votre type de données** (tabular, images, caméra)
- **Uploader un fichier** ou utiliser les données de démonstration
- **Sélectionnez la colonne cible** et le type de problème

### Étape 2: Configuration
- **Aperçu des données**
- **Vérification des paramètres**
- **Préparation pour l'entraînement**

### Étape 3: Entraînement
- **Sélection des algorithmes**
- **Entraînement individuel ou multiple**
- **Barres de progression en temps réel**

### Étape 4: Résultats
- **Comparaison des performances**
- **Métriques détaillées par modèle**
- **Visualisations interactives**
- **Architecture des réseaux de neurones**

##  Démarrage Rapide

### Utilisation avec GitHub Codespaces

1. **Cliquez sur le bouton `Code`** puis sélectionnez `Codespaces`
2. **Créez un nouveau codespace** (l'installation est automatique)
3. **L'application se lance automatiquement** sur le port 8501
4. **Ouvrez l'onglet "Ports"** et cliquez sur l'icône 🌐 pour prévisualiser

### Installation Locale

```bash
# Cloner le repository
git clone https://github.com/votre-username/teachable-machine.git
cd teachable-machine

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
streamlit run Teachable_machine.py
