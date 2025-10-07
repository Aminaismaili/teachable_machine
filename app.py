import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import pickle
import os
import zipfile
from PIL import Image
# Retirer l'import cv2 - utiliser PIL seulement

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, mean_squared_error,
    mean_absolute_error, r2_score
)

# Classification algorithms
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Regression algorithms
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

# Deep Learning
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    st.warning("⚠️ TensorFlow n'est pas installé. Le Deep Learning est désactivé.")

# Configuration de la page
st.set_page_config(
    page_title="Teachable Machine - Images & ML",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
.stApp {
    background: linear-gradient(to bottom right, #f8f9fa, #e9ecef);
}
h1 {
    color: #667eea;
    font-weight: bold;
}
.stButton>button {
    width: 100%;
    background: linear-gradient(to right, #667eea, #764ba2);
    color: white;
    font-weight: bold;
    border-radius: 10px;
    padding: 15px;
    border: none;
}
.stButton>button:hover {
    transform: scale(1.02);
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
}
.image-preview {
    border: 2px dashed #ddd;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# ==================== CLASSES ====================

class DataProcessor:
    """Classe pour le preprocessing des données"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self, file):
        """Charge les données depuis un fichier"""
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.name.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file)
            else:
                return None, "Format non supporté"
            return df, None
        except Exception as e:
            return None, str(e)
    
    def preprocess_data(self, df, target_col, test_size=0.2, random_state=42):
        """Prétraitement des données tabulaires"""
        # Séparation features et target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Encodage des variables catégorielles dans X
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        # Encodage de y si catégoriel
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
            self.label_encoders[target_col] = le
        
        # Split des données
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Normalisation
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

class ImageProcessor:
    """Classe pour le traitement des images avec PIL seulement"""
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.class_names = []
        
    def load_images_from_zip(self, zip_file, img_size=(128, 128)):
        """Charge les images depuis un fichier zip avec PIL"""
        images = []
        labels = []
        
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            # Créer un dossier temporaire
            temp_dir = "temp_images"
            zip_ref.extractall(temp_dir)
            
            # Parcourir les dossiers (chaque dossier = une classe)
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        # Le nom du dossier parent est le label
                        label = os.path.basename(root)
                        if label not in self.class_names and label != temp_dir:
                            self.class_names.append(label)
                        
                        # Charger et prétraiter l'image avec PIL
                        img_path = os.path.join(root, file)
                        img = self.preprocess_image(img_path, img_size)
                        
                        if img is not None:
                            images.append(img)
                            labels.append(label)
            
            # Nettoyer le dossier temporaire
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        
        if len(images) == 0:
            st.error("❌ Aucune image trouvée. Vérifiez la structure de votre ZIP.")
            return np.array([]), np.array([])
            
        return np.array(images), np.array(labels)
    
    def preprocess_image(self, img_path, target_size):
        """Prétraite une image unique avec PIL seulement"""
        try:
            # Charger l'image avec PIL
            img = Image.open(img_path)
            
            # Convertir en RGB si nécessaire
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            # Redimensionner
            img = img.resize(target_size)
            
            # Convertir en array numpy et normaliser
            img_array = np.array(img) / 255.0
            
            return img_array
            
        except Exception as e:
            st.warning(f"⚠️ Erreur avec l'image {os.path.basename(img_path)}: {str(e)}")
            return None
    
    def encode_labels(self, labels):
        """Encode les labels textuels en numérique"""
        return self.label_encoder.fit_transform(labels)

class ClassicalMLTrainer:
    """Classe pour entraîner les modèles ML classiques"""
    
    def __init__(self, problem_type='classification'):
        self.problem_type = problem_type
        self.models = {}
        self.results = {}
        
    def get_models(self):
        """Retourne les modèles disponibles selon le type de problème"""
        if self.problem_type == 'classification':
            return {
                'Random Forest': RandomForestClassifier(random_state=42),
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                'SVM': SVC(random_state=42, probability=True),
                'KNN': KNeighborsClassifier(),
                'Decision Tree': DecisionTreeClassifier(random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(random_state=42),
                'AdaBoost': AdaBoostClassifier(random_state=42),
                'Naive Bayes': GaussianNB()
            }
        else:  # regression
            return {
                'Random Forest': RandomForestRegressor(random_state=42),
                'Linear Regression': LinearRegression(),
                'Ridge': Ridge(random_state=42),
                'Lasso': Lasso(random_state=42),
                'ElasticNet': ElasticNet(random_state=42),
                'SVR': SVR(),
                'KNN': KNeighborsRegressor(),
                'Decision Tree': DecisionTreeRegressor(random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(random_state=42)
            }
    
    def train_models(self, X_train, X_test, y_train, y_test, selected_models):
        """Entraîne les modèles sélectionnés"""
        # Si ce sont des images, flatten pour les modèles classiques
        if len(X_train.shape) > 2:
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
        else:
            X_train_flat = X_train
            X_test_flat = X_test
            
        available_models = self.get_models()
        
        for model_name in selected_models:
            if model_name in available_models:
                with st.spinner(f'🔄 Entraînement de {model_name}...'):
                    model = available_models[model_name]
                    model.fit(X_train_flat, y_train)
                    y_pred = model.predict(X_test_flat)
                    
                    self.models[model_name] = model
                    self.results[model_name] = {
                        'predictions': y_pred,
                        'metrics': self._calculate_metrics(y_test, y_pred)
                    }
        
        return self.results
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calcule les métriques selon le type de problème"""
        if self.problem_type == 'classification':
            return {
                'Accuracy': accuracy_score(y_true, y_pred),
                'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'F1-Score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
            }
        else:  # regression
            return {
                'R² Score': r2_score(y_true, y_pred),
                'MSE': mean_squared_error(y_true, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
                'MAE': mean_absolute_error(y_true, y_pred)
            }

class DeepLearningTrainer:
    """Classe pour créer et entraîner des réseaux de neurones"""
    
    def __init__(self, problem_type='classification', input_shape=None):
        self.problem_type = problem_type
        self.input_shape = input_shape
        self.model = None
        self.history = None
        
    def build_cnn_model(self, num_classes):
        """Construit un modèle CNN pour les images"""
        model = models.Sequential([
            # Première couche de convolution
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            
            # Deuxième couche de convolution
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            # Troisième couche de convolution
            layers.Conv2D(64, (3, 3), activation='relu'),
            
            # Couches fully connected
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            
            # Couche de sortie
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        self.model = model
        return model
    
    def build_simple_nn(self, input_dim, num_classes):
        """Construit un réseau de neurones simple pour données tabulaires"""
        model = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Entraîne le modèle"""
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """Évalue le modèle"""
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        return {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'F1-Score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }, y_pred

# ==================== INTERFACE PRINCIPALE ====================

def main():
    # Header
    st.markdown("""
        <h1 style='text-align: center; padding: 20px;'>
            🤖 Teachable Machine - Images & ML
        </h1>
        <p style='text-align: center; color: #666; font-size: 18px;'>
            Entraînez vos modèles sur des images ou données tabulaires
        </p>
    """, unsafe_allow_html=True)
    
    # Initialisation des sessions
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'trained' not in st.session_state:
        st.session_state.trained = False
    if 'data_type' not in st.session_state:
        st.session_state.data_type = "tabular"  # "tabular" or "image"
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🤖 Navigation")
        
        step = st.radio(
            "Étapes:",
            ["1️⃣ Upload Data", "2️⃣ Configuration", "3️⃣ Entraînement", "4️⃣ Résultats"],
            label_visibility="collapsed"
        )
    
    # ==================== ÉTAPE 1: UPLOAD DATA ====================
    if step == "1️⃣ Upload Data":
        st.header("📁 Upload vos données")
        
        # Sélection du type de données
        data_type = st.radio(
            "Type de données:",
            ["📊 Données Tabulaire (CSV/Excel)", "🖼️ Images (ZIP avec dossiers par classe)"],
            horizontal=True
        )
        
        st.session_state.data_type = "tabular" if "Tabulaire" in data_type else "image"
        
        if st.session_state.data_type == "tabular":
            uploaded_file = st.file_uploader(
                "Choisissez un fichier CSV ou Excel",
                type=['csv', 'xlsx', 'xls'],
                help="Formats supportés: CSV, Excel"
            )
        else:
            uploaded_file = st.file_uploader(
                "Choisissez un fichier ZIP contenant vos images",
                type=['zip'],
                help="Le ZIP doit contenir des dossiers (un par classe) avec les images"
            )
            st.info("""
            💡 **Structure recommandée:** 
            ```
            mon_dataset.zip
            ├── classe1/
            │   ├── image1.jpg
            │   ├── image2.jpg
            ├── classe2/
            │   ├── image1.jpg
            │   ├── image2.jpg
            ```
            """)
        
        if uploaded_file:
            if st.session_state.data_type == "tabular":
                processor = DataProcessor()
                df, error = processor.load_data(uploaded_file)
                
                if error:
                    st.error(f"❌ Erreur: {error}")
                else:
                    st.success("✅ Données tabulaires chargées avec succès!")
                    st.session_state.df = df
                    st.session_state.data_loaded = True
                    st.session_state.processor = processor
                    
                    # Aperçu des données
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📊 Lignes", df.shape[0])
                    with col2:
                        st.metric("📋 Colonnes", df.shape[1])
                    with col3:
                        st.metric("🔢 Types numériques", df.select_dtypes(include=[np.number]).shape[1])
                    
                    st.subheader("📋 Aperçu des données")
                    st.dataframe(df.head(10), use_container_width=True)
            
            else:  # Image data
                with st.spinner("🔄 Traitement des images..."):
                    image_processor = ImageProcessor()
                    images, labels = image_processor.load_images_from_zip(uploaded_file, img_size=(128, 128))
                    
                    if len(images) > 0:
                        st.success(f"✅ {len(images)} images chargées avec succès!")
                        st.session_state.images = images
                        st.session_state.labels = labels
                        st.session_state.image_processor = image_processor
                        st.session_state.data_loaded = True
                        
                        # Aperçu des images
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("🖼️ Images", len(images))
                        with col2:
                            st.metric("🎯 Classes", len(image_processor.class_names))
                        with col3:
                            st.metric("📐 Taille", f"{images[0].shape[0]}x{images[0].shape[1]}")
                        
                        st.subheader("📊 Distribution des classes")
                        label_counts = pd.Series(labels).value_counts()
                        fig, ax = plt.subplots(figsize=(10, 4))
                        label_counts.plot(kind='bar', ax=ax, color='skyblue')
                        ax.set_title('Nombre d\'images par classe')
                        ax.set_xlabel('Classes')
                        ax.set_ylabel('Nombre d\'images')
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                        
                        # Aperçu de quelques images
                        st.subheader("👀 Aperçu des images")
                        n_preview = min(6, len(images))
                        cols = st.columns(3)
                        for i in range(n_preview):
                            with cols[i % 3]:
                                fig, ax = plt.subplots(figsize=(3, 3))
                                ax.imshow(images[i])
                                ax.set_title(f'Classe: {labels[i]}')
                                ax.axis('off')
                                st.pyplot(fig)
                    else:
                        st.error("❌ Aucune image valide trouvée dans le fichier ZIP")
    
    # ==================== ÉTAPE 2: CONFIGURATION ====================
    elif step == "2️⃣ Configuration":
        if not st.session_state.data_loaded:
            st.warning("⚠️ Veuillez d'abord charger des données!")
            return
        
        st.header("⚙️ Configuration du modèle")
        
        if st.session_state.data_type == "tabular":
            df = st.session_state.df
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Type de problème")
                problem_type = st.radio(
                    "Sélectionnez:",
                    ["Classification", "Régression"],
                    horizontal=True
                )
                st.session_state.problem_type = problem_type.lower()
            
            with col2:
                st.subheader("🧠 Type de modèle")
                model_type = st.radio(
                    "Sélectionnez:",
                    ["ML Classique", "Deep Learning"] if TENSORFLOW_AVAILABLE else ["ML Classique"],
                    horizontal=True
                )
                st.session_state.model_type = model_type
            
            st.subheader("🎯 Variable cible")
            target_col = st.selectbox("Sélectionnez la colonne cible:", df.columns)
            st.session_state.target_col = target_col
        
        else:  # Image data
            st.subheader("🎯 Type de problème")
            st.info("🖼️ Classification d'images sélectionnée automatiquement")
            st.session_state.problem_type = "classification"
            
            st.subheader("🧠 Type de modèle")
            if TENSORFLOW_AVAILABLE:
                model_type = st.radio(
                    "Sélectionnez:",
                    ["ML Classique", "Deep Learning (CNN)"],
                    horizontal=True
                )
                st.session_state.model_type = model_type
            else:
                st.warning("⚠️ TensorFlow non disponible - ML Classique uniquement")
                st.session_state.model_type = "ML Classique"
            
            st.subheader("📊 Informations sur les données")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Classes", len(st.session_state.image_processor.class_names))
            with col2:
                st.metric("Images total", len(st.session_state.images))
            with col3:
                st.metric("Taille images", f"{st.session_state.images[0].shape[:2]}")
        
        st.subheader("✂️ Split des données")
        col1, col2 = st.columns([3, 1])
        with col1:
            test_size = st.slider(
                "Taille du test set (%)",
                min_value=10,
                max_value=40,
                value=20,
                step=5
            )
        with col2:
            st.metric("Train", f"{100-test_size}%")
            st.metric("Test", f"{test_size}%")
        
        st.session_state.test_size = test_size / 100
        
        if st.button("✅ Valider la configuration", type="primary"):
            st.success("✅ Configuration enregistrée! Passez à l'entraînement.")
    
    # ==================== ÉTAPE 3: ENTRAÎNEMENT ====================
    elif step == "3️⃣ Entraînement":
        if not st.session_state.data_loaded:
            st.warning("⚠️ Veuillez d'abord charger des données!")
            return
        
        if 'problem_type' not in st.session_state:
            st.warning("⚠️ Veuillez d'abord configurer votre modèle!")
            return
        
        st.header("🚀 Entraînement du modèle")
        
        # Préparation des données
        if 'X_train' not in st.session_state:
            with st.spinner("🔄 Préparation des données..."):
                if st.session_state.data_type == "tabular":
                    processor = st.session_state.processor
                    X_train, X_test, y_train, y_test, feature_names = processor.preprocess_data(
                        st.session_state.df,
                        st.session_state.target_col,
                        st.session_state.test_size
                    )
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.feature_names = feature_names
                
                else:  # Image data
                    images = st.session_state.images
                    labels = st.session_state.labels
                    
                    # Encoder les labels
                    y_encoded = st.session_state.image_processor.encode_labels(labels)
                    
                    # Split des données
                    X_train, X_test, y_train, y_test = train_test_split(
                        images, y_encoded, 
                        test_size=st.session_state.test_size, 
                        random_state=42,
                        stratify=y_encoded
                    )
                    
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.feature_names = None
            
            st.success("✅ Données préparées!")
        
        # ML CLASSIQUE
        if st.session_state.model_type == "ML Classique":
            st.subheader("🤖 Algorithmes ML Classiques")
            
            trainer = ClassicalMLTrainer(st.session_state.problem_type)
            available_models = list(trainer.get_models().keys())
            
            selected_models = st.multiselect(
                "Sélectionnez les algorithmes à entraîner:",
                available_models,
                default=available_models[:3]
            )
            
            if st.button("🚀 Lancer l'entraînement", type="primary"):
                if selected_models:
                    results = trainer.train_models(
                        st.session_state.X_train,
                        st.session_state.X_test,
                        st.session_state.y_train,
                        st.session_state.y_test,
                        selected_models
                    )
                    st.session_state.ml_results = results
                    st.session_state.ml_trainer = trainer
                    st.session_state.trained = True
                    st.success("✅ Entraînement terminé!")
                    st.balloons()
                else:
                    st.warning("⚠️ Veuillez sélectionner au moins un algorithme!")
        
        # DEEP LEARNING
        elif st.session_state.model_type.startswith("Deep Learning"):
            if not TENSORFLOW_AVAILABLE:
                st.error("❌ TensorFlow n'est pas installé. Veuillez installer TensorFlow pour utiliser le Deep Learning.")
                return
            
            st.subheader("🧠 Deep Learning")
            
            if st.session_state.data_type == "image":
                st.info("🖼️ Architecture CNN pour images sélectionnée")
                
                epochs = st.slider("Epochs", 10, 100, 30)
                batch_size = st.selectbox("Batch size", [16, 32, 64], index=1)
                
                if st.button("🚀 Lancer l'entraînement CNN", type="primary"):
                    with st.spinner("🔄 Entraînement du CNN en cours..."):
                        # Créer le modèle CNN
                        input_shape = st.session_state.X_train[0].shape
                        num_classes = len(np.unique(st.session_state.y_train))
                        
                        dl_trainer = DeepLearningTrainer(
                            problem_type=st.session_state.problem_type,
                            input_shape=input_shape
                        )
                        
                        model = dl_trainer.build_cnn_model(num_classes)
                        
                        # Afficher l'architecture
                        st.subheader("🏗️ Architecture du CNN")
                        model_summary = []
                        model.summary(print_fn=lambda x: model_summary.append(x))
                        st.text("\n".join(model_summary))
                        
                        # Entraînement
                        history = dl_trainer.train(
                            st.session_state.X_train,
                            st.session_state.y_train,
                            st.session_state.X_test,
                            st.session_state.y_test,
                            epochs=epochs,
                            batch_size=batch_size
                        )
                        
                        st.session_state.dl_trainer = dl_trainer
                        st.session_state.dl_history = history
                        st.session_state.trained = True
                        
                    st.success("✅ Entraînement CNN terminé!")
                    st.balloons()
            
            else:  # Tabular data with DL
                st.info("📊 Réseau de neurones simple pour données tabulaires")
                
                epochs = st.slider("Epochs", 10, 200, 50)
                batch_size = st.selectbox("Batch size", [16, 32, 64, 128], index=1)
                
                if st.button("🚀 Lancer l'entraînement DL", type="primary"):
                    with st.spinner("🔄 Entraînement en cours..."):
                        input_dim = st.session_state.X_train.shape[1]
                        num_classes = len(np.unique(st.session_state.y_train))
                        
                        dl_trainer = DeepLearningTrainer(st.session_state.problem_type)
                        model = dl_trainer.build_simple_nn(input_dim, num_classes)
                        
                        # Entraînement
                        history = dl_trainer.train(
                            st.session_state.X_train,
                            st.session_state.y_train,
                            st.session_state.X_test,
                            st.session_state.y_test,
                            epochs=epochs,
                            batch_size=batch_size
                        )
                        
                        st.session_state.dl_trainer = dl_trainer
                        st.session_state.dl_history = history
                        st.session_state.trained = True
                    
                    st.success("✅ Entraînement DL terminé!")
                    st.balloons()
    
    # ==================== ÉTAPE 4: RÉSULTATS ====================
    elif step == "4️⃣ Résultats":
        if not st.session_state.trained:
            st.warning("⚠️ Veuillez d'abord entraîner un modèle!")
            return
        
        st.header("📊 Résultats et Évaluation")
        
        # RÉSULTATS ML CLASSIQUE
        if st.session_state.model_type == "ML Classique":
            results = st.session_state.ml_results
            
            # Tableau comparatif
            st.subheader("📈 Comparaison des modèles")
            
            metrics_df = pd.DataFrame({
                model: res['metrics']
                for model, res in results.items()
            }).T
            
            st.dataframe(metrics_df.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
            
            # Meilleur modèle
            if st.session_state.problem_type == 'classification':
                best_metric = 'Accuracy'
            else:
                best_metric = 'R² Score'
            
            best_model = metrics_df[best_metric].idxmax()
            best_score = metrics_df[best_metric].max()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🏆 Meilleur Modèle", best_model)
            with col2:
                st.metric(f"📊 {best_metric}", f"{best_score:.4f}")
            with col3:
                st.metric("📁 Modèles entraînés", len(results))
            
            # Visualisations
            st.subheader("📊 Visualisations")
            
            if st.session_state.problem_type == 'classification':
                # Matrice de confusion pour le meilleur modèle
                y_pred = results[best_model]['predictions']
                cm = confusion_matrix(st.session_state.y_test, y_pred)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title(f'Matrice de Confusion - {best_model}')
                ax.set_xlabel('Prédictions')
                ax.set_ylabel('Vraies valeurs')
                st.pyplot(fig)
            
            else:  # regression
                # Prédictions vs Réelles
                y_pred = results[best_model]['predictions']
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(st.session_state.y_test, y_pred, alpha=0.5)
                ax.plot([st.session_state.y_test.min(), st.session_state.y_test.max()],
                       [st.session_state.y_test.min(), st.session_state.y_test.max()],
                       'r--', lw=2)
                ax.set_xlabel('Valeurs réelles')
                ax.set_ylabel('Prédictions')
                ax.set_title(f'Prédictions vs Réelles - {best_model}')
                st.pyplot(fig)
            
            # Téléchargement du modèle
            st.subheader("💾 Télécharger le modèle")
            model_to_download = st.selectbox("Sélectionnez un modèle:", list(results.keys()))
            
            if st.button("📥 Télécharger"):
                model = st.session_state.ml_trainer.models[model_to_download]
                buffer = BytesIO()
                pickle.dump(model, buffer)
                buffer.seek(0)
                
                st.download_button(
                    label="💾 Télécharger le modèle",
                    data=buffer,
                    file_name=f"{model_to_download.replace(' ', '_')}_model.pkl",
                    mime="application/octet-stream"
                )
        
        # RÉSULTATS DEEP LEARNING
        else:
            dl_trainer = st.session_state.dl_trainer
            history = st.session_state.dl_history.history
            
            # Métriques
            metrics, predictions = dl_trainer.evaluate(
                st.session_state.X_test,
                st.session_state.y_test
            )
            
            st.subheader("📊 Métriques du modèle")
            cols = st.columns(len(metrics))
            for idx, (metric, value) in enumerate(metrics.items()):
                with cols[idx]:
                    st.metric(metric, f"{value:.4f}")
            
            # Courbes d'apprentissage
            st.subheader("📈 Courbes d'apprentissage")
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            
            # Loss
            axes[0].plot(history['loss'], label='Train Loss')
            axes[0].plot(history['val_loss'], label='Val Loss')
            axes[0].set_title('Loss durant l\'entraînement')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].legend()
            axes[0].grid(True)
            
            # Accuracy
            if 'accuracy' in history:
                axes[1].plot(history['accuracy'], label='Train Accuracy')
                axes[1].plot(history['val_accuracy'], label='Val Accuracy')
                axes[1].set_title('Accuracy durant l\'entraînement')
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('Accuracy')
                axes[1].legend()
                axes[1].grid(True)
            
            st.pyplot(fig)
            
            # Matrice de confusion pour DL
            if st.session_state.problem_type == 'classification':
                st.subheader("🎯 Matrice de Confusion")
                y_true = st.session_state.y_test
                cm = confusion_matrix(y_true, predictions)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title('Matrice de Confusion - Modèle DL')
                ax.set_xlabel('Prédictions')
                ax.set_ylabel('Vraies valeurs')
                st.pyplot(fig)
            
            # Téléchargement du modèle
            st.subheader("💾 Télécharger le modèle")
            if st.button("📥 Télécharger le modèle DL"):
                dl_trainer.model.save('model_dl.h5')
                with open('model_dl.h5', 'rb') as f:
                    st.download_button(
                        label="💾 Télécharger",
                        data=f,
                        file_name="neural_network_model.h5",
                        mime="application/octet-stream"
                    )
        
        # Bouton pour ré-entraîner
        st.divider()
        if st.button("🔄 Ré-entraîner avec d'autres paramètres"):
            st.session_state.trained = False
            st.rerun()

if __name__ == "__main__":
    main()
