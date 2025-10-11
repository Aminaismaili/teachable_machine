import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import pickle
from PIL import Image
import os
import cv2
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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuration de la page
st.set_page_config(
    page_title="Teachable Machine",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé - THÈME NOIR
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stApp {
        background: linear-gradient(135deg, #0E1117 0%, #1E1E1E 50%, #0E1117 100%);
        color: #FAFAFA;
    }
    .stSidebar {
        background: linear-gradient(180deg, #1E1E1E 0%, #0E1117 100%);
    }
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF;
        font-weight: bold;
    }
    .stRadio > label {
        color: #FFFFFF !important;
    }
    .stSelectbox > label {
        color: #FFFFFF !important;
    }
    .stMultiSelect > label {
        color: #FFFFFF !important;
    }
    .stSlider > label {
        color: #FFFFFF !important;
    }
    .stNumberInput > label {
        color: #FFFFFF !important;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(to right, #667eea, #764ba2);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 15px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    .stDataFrame {
        background-color: #1E1E1E;
        color: #FAFAFA;
    }
    .stMetric {
        background-color: rgba(30, 30, 30, 0.7);
        border-radius: 10px;
        padding: 10px;
        border: 1px solid #333;
    }
    .css-1d391kg, .css-12oz5g7 {
        background-color: #0E1117;
    }
    .stAlert {
        background-color: rgba(30, 30, 30, 0.8);
        border: 1px solid #444;
    }
    .stProgress > div > div > div {
        background: linear-gradient(to right, #667eea, #764ba2);
    }
    .stMarkdown {
        color: #FAFAFA;
    }
    .stFileUploader > label {
        color: #FFFFFF !important;
    }
    .stFileUploader {
        background-color: rgba(30, 30, 30, 0.5);
        border-radius: 10px;
        padding: 20px;
        border: 2px dashed #444;
    }
    div[data-baseweb="select"] > div {
        background-color: #1E1E1E;
        color: #FAFAFA;
    }
    div[data-baseweb="input"] > div {
        background-color: #1E1E1E;
        color: #FAFAFA;
    }
    .stRadio [role="radiogroup"] {
        background-color: #1E1E1E;
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #333;
    }
    .stSelectbox [data-baseweb="select"] {
        background-color: #1E1E1E;
        color: #FAFAFA;
    }
    .stMultiSelect [data-baseweb="select"] {
        background-color: #1E1E1E;
        color: #FAFAFA;
    }
    .stSlider [data-baseweb="slider"] {
        color: #667eea;
    }
    </style>
""", unsafe_allow_html=True)

# Configuration matplotlib pour le thème noir
plt.rcParams['figure.facecolor'] = '#1E1E1E'
plt.rcParams['axes.facecolor'] = '#1E1E1E'
plt.rcParams['axes.edgecolor'] = '#FFFFFF'
plt.rcParams['axes.labelcolor'] = '#FFFFFF'
plt.rcParams['text.color'] = '#FFFFFF'
plt.rcParams['xtick.color'] = '#FFFFFF'
plt.rcParams['ytick.color'] = '#FFFFFF'
plt.rcParams['grid.color'] = '#444444'

# ==================== CLASSES ====================

class DataProcessor:
    """Classe pour le preprocessing des données"""
    
    def _init_(self):
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
    """Classe pour le traitement des images"""
    
    def _init_(self, img_size=(128, 128)):
        self.img_size = img_size
        self.label_encoder = LabelEncoder()
        
    def load_images_from_zip(self, zip_file):
        """Charge les images depuis un fichier zip"""
        import zipfile
        import tempfile
        
        images = []
        labels = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        # Le nom du dossier est le label
                        label = os.path.basename(root)
                        img_path = os.path.join(root, file)
                        
                        try:
                            # Charger et prétraiter l'image
                            img = Image.open(img_path)
                            img = img.convert('RGB')
                            img = img.resize(self.img_size)
                            img_array = np.array(img) / 255.0
                            
                            images.append(img_array)
                            labels.append(label)
                        except Exception as e:
                            st.warning(f"Impossible de charger l'image {file}: {e}")
        
        return np.array(images), np.array(labels)
    
    def load_images_from_folders(self, uploaded_files):
        """Charge les images depuis des fichiers uploadés"""
        images = []
        labels = []
        
        for uploaded_file in uploaded_files:
            if uploaded_file.name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                try:
                    # Utiliser le nom du fichier ou du dossier comme label
                    label = uploaded_file.name.split('_')[0]  # Première partie du nom comme label
                    
                    # Charger et prétraiter l'image
                    img = Image.open(uploaded_file)
                    img = img.convert('RGB')
                    img = img.resize(self.img_size)
                    img_array = np.array(img) / 255.0
                    
                    images.append(img_array)
                    labels.append(label)
                except Exception as e:
                    st.warning(f"Impossible de charger l'image {uploaded_file.name}: {e}")
        
        return np.array(images), np.array(labels)
    
    def preprocess_images(self, images, labels, test_size=0.2, random_state=42):
        """Prétraitement des images"""
        # Encodage des labels
        y_encoded = self.label_encoder.fit_transform(labels)
        
        # Split des données
        X_train, X_test, y_train, y_test = train_test_split(
            images, y_encoded, test_size=test_size, random_state=random_state
        )
        
        return X_train, X_test, y_train, y_test

class ClassicalMLTrainer:
    """Classe pour entraîner les modèles ML classiques"""
    
    def _init_(self, problem_type='classification'):
        self.problem_type = problem_type
        self.models = {}
        self.results = {}
        
    def get_models(self):
        """Retourne les modèles disponibles selon le type de problème"""
        if self.problem_type == 'classification':
            return {
                'Random Forest': RandomForestClassifier(random_state=42),
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                'SVM': SVC(random_state=42),
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
        available_models = self.get_models()
        
        for model_name in selected_models:
            if model_name in available_models:
                with st.spinner(f'🔄 Entraînement de {model_name}...'):
                    model = available_models[model_name]
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
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
    
    def _init_(self, problem_type='classification', input_shape=None):
        self.problem_type = problem_type
        self.input_shape = input_shape
        self.model = None
        self.history = None
        
    def build_model(self, input_dim, output_dim, architecture='default', custom_params=None):
        """Construit un réseau de neurones pour données tabulaires"""
        model = models.Sequential()
        
        if architecture == 'default':
            # Architecture par défaut
            model.add(layers.Dense(128, activation='relu', input_shape=(input_dim,)))
            model.add(layers.Dropout(0.3))
            model.add(layers.Dense(64, activation='relu'))
            model.add(layers.Dropout(0.3))
            model.add(layers.Dense(32, activation='relu'))
        else:
            # Architecture personnalisée
            n_layers = custom_params.get('n_layers', 3)
            neurons = custom_params.get('neurons', 128)
            dropout = custom_params.get('dropout', 0.3)
            
            model.add(layers.Dense(neurons, activation='relu', input_shape=(input_dim,)))
            model.add(layers.Dropout(dropout))
            
            for i in range(n_layers - 1):
                neurons = neurons // 2
                model.add(layers.Dense(neurons, activation='relu'))
                model.add(layers.Dropout(dropout))
        
        # Couche de sortie
        if self.problem_type == 'classification':
            if output_dim == 2:
                model.add(layers.Dense(1, activation='sigmoid'))
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            else:
                model.add(layers.Dense(output_dim, activation='softmax'))
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        else:  # regression
            model.add(layers.Dense(1))
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        self.model = model
        return model
    
    def build_cnn_model(self, output_dim, architecture='default', custom_params=None):
        """Construit un modèle CNN pour les images"""
        model = models.Sequential()
        
        if architecture == 'default':
            # Architecture CNN par défaut
            model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Conv2D(64, (3, 3), activation='relu'))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Conv2D(64, (3, 3), activation='relu'))
            model.add(layers.Flatten())
            model.add(layers.Dense(64, activation='relu'))
            model.add(layers.Dropout(0.5))
        else:
            # Architecture personnalisée
            n_conv_layers = custom_params.get('n_conv_layers', 3)
            filters = custom_params.get('filters', 32)
            dense_neurons = custom_params.get('dense_neurons', 64)
            dropout = custom_params.get('dropout', 0.5)
            
            model.add(layers.Conv2D(filters, (3, 3), activation='relu', input_shape=self.input_shape))
            model.add(layers.MaxPooling2D((2, 2)))
            
            for i in range(n_conv_layers - 1):
                filters *= 2
                model.add(layers.Conv2D(filters, (3, 3), activation='relu'))
                model.add(layers.MaxPooling2D((2, 2)))
            
            model.add(layers.Flatten())
            model.add(layers.Dense(dense_neurons, activation='relu'))
            model.add(layers.Dropout(dropout))
        
        # Couche de sortie
        if output_dim == 2:
            model.add(layers.Dense(1, activation='sigmoid'))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        else:
            model.add(layers.Dense(output_dim, activation='softmax'))
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Entraîne le modèle"""
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=0
        )
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """Évalue le modèle"""
        y_pred = self.model.predict(X_test)
        
        if self.problem_type == 'classification':
            if y_pred.shape[1] == 1:  # Binary
                y_pred_classes = (y_pred > 0.5).astype(int).flatten()
            else:  # Multiclass
                y_pred_classes = np.argmax(y_pred, axis=1)
            
            return {
                'Accuracy': accuracy_score(y_test, y_pred_classes),
                'Precision': precision_score(y_test, y_pred_classes, average='weighted', zero_division=0),
                'Recall': recall_score(y_test, y_pred_classes, average='weighted', zero_division=0),
                'F1-Score': f1_score(y_test, y_pred_classes, average='weighted', zero_division=0)
            }, y_pred_classes
        else:
            return {
                'R² Score': r2_score(y_test, y_pred),
                'MSE': mean_squared_error(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'MAE': mean_absolute_error(y_test, y_pred)
            }, y_pred

# ==================== INTERFACE PRINCIPALE ====================

def main():
    # Header avec style noir
    st.markdown("""
        <h1 style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea, #764ba2); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: bold;'>
            🤖 Teachable Machine - Dark Mode
        </h1>
        <p style='text-align: center; color: #CCCCCC; font-size: 18px;'>
            Entraînez vos modèles de Machine Learning avec des données tabulaires ou des images
        </p>
        <hr style='border: 1px solid #333; margin: 20px 0;'>
    """, unsafe_allow_html=True)
    
    # Initialisation des sessions
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'trained' not in st.session_state:
        st.session_state.trained = False
    if 'data_type' not in st.session_state:
        st.session_state.data_type = None
    
    # Sidebar avec style noir
    with st.sidebar:
        st.markdown("""
            <div style='text-align: center; padding: 10px;'>
                <img src='https://cdn-icons-png.flaticon.com/512/8637/8637099.png' width='80' style='filter: invert(1);'>
            </div>
            <h3 style='text-align: center; color: #FFFFFF;'>📊 Navigation</h3>
            <hr style='border: 1px solid #333; margin: 10px 0;'>
        """, unsafe_allow_html=True)
        
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
            "📊 Type de données:",
            ["📈 Données Tabulaire (CSV/Excel)", "🖼 Images"],
            horizontal=True
        )
        
        st.session_state.data_type = data_type
        
        if "Tabulaire" in data_type:
            uploaded_file = st.file_uploader(
                "Choisissez un fichier CSV ou Excel",
                type=['csv', 'xlsx', 'xls'],
                help="Formats supportés: CSV, Excel"
            )
            
            if uploaded_file:
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
                        st.metric("🔢 Types", df.select_dtypes(include=[np.number]).shape[1])
                    
                    st.subheader("📋 Aperçu des données")
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    st.subheader("📊 Informations")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("*Types de colonnes:*")
                        st.dataframe(df.dtypes.to_frame('Type'), use_container_width=True)
                    
                    with col2:
                        st.write("*Valeurs manquantes:*")
                        missing = df.isnull().sum()
                        st.dataframe(missing.to_frame('Missing'), use_container_width=True)
        
        else:  # Images
            st.subheader("🖼 Upload d'Images")
            
            upload_option = st.radio(
                "Méthode d'upload:",
                ["📁 Fichiers multiples", "🗜 Archive ZIP"],
                horizontal=True
            )
            
            if upload_option == "📁 Fichiers multiples":
                uploaded_files = st.file_uploader(
                    "Choisissez vos images",
                    type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
                    accept_multiple_files=True,
                    help="Formats supportés: PNG, JPG, JPEG, BMP, TIFF"
                )
                
                if uploaded_files:
                    st.success(f"✅ {len(uploaded_files)} images chargées!")
                    
                    # Afficher un aperçu des images
                    st.subheader("👀 Aperçu des images")
                    cols = st.columns(4)
                    for idx, uploaded_file in enumerate(uploaded_files[:8]):
                        with cols[idx % 4]:
                            img = Image.open(uploaded_file)
                            st.image(img, caption=uploaded_file.name, width=150)
                    
                    st.session_state.uploaded_files = uploaded_files
                    st.session_state.data_loaded = True
                    st.session_state.image_processor = ImageProcessor()
            
            else:  # Archive ZIP
                zip_file = st.file_uploader(
                    "Choisissez une archive ZIP contenant vos images",
                    type=['zip'],
                    help="Archive ZIP avec des dossiers organisés par classe"
                )
                
                if zip_file:
                    st.success("✅ Archive ZIP chargée!")
                    st.session_state.zip_file = zip_file
                    st.session_state.data_loaded = True
                    st.session_state.image_processor = ImageProcessor()
    
    # ==================== ÉTAPE 2: CONFIGURATION ====================
    elif step == "2️⃣ Configuration":
        if not st.session_state.data_loaded:
            st.warning("⚠ Veuillez d'abord charger des données!")
            return
        
        st.header("⚙ Configuration du modèle")
        
        # Configuration différente selon le type de données
        if "Tabulaire" in st.session_state.data_type:
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
                    ["ML Classique", "Deep Learning"],
                    horizontal=True
                )
                st.session_state.model_type = model_type
            
            st.subheader("🎯 Variable cible")
            target_col = st.selectbox("Sélectionnez la colonne cible:", df.columns)
            st.session_state.target_col = target_col
            
        else:  # Images
            st.subheader("🎯 Configuration pour les Images")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Type de problème")
                problem_type = st.radio(
                    "Sélectionnez:",
                    ["Classification"],
                    horizontal=True
                )
                st.session_state.problem_type = "classification"
                
                st.subheader("🧠 Type de modèle")
                model_type = st.radio(
                    "Sélectionnez:",
                    ["Deep Learning"],
                    horizontal=True
                )
                st.session_state.model_type = "Deep Learning"
            
            with col2:
                st.subheader("🖼 Paramètres des images")
                img_size = st.selectbox(
                    "Taille des images:",
                    [(64, 64), (128, 128), (224, 224)],
                    format_func=lambda x: f"{x[0]}x{x[1]}"
                )
                st.session_state.img_size = img_size
        
        st.subheader("✂ Split des données")
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
            st.warning("⚠ Veuillez d'abord charger des données!")
            return
        
        if 'target_col' not in st.session_state and "Tabulaire" in st.session_state.data_type:
            st.warning("⚠ Veuillez d'abord configurer votre modèle!")
            return
        
        st.header("🚀 Entraînement du modèle")
        
        # Préparation des données selon le type
        if "Tabulaire" in st.session_state.data_type:
            # Données tabulaires
            if 'X_train' not in st.session_state:
                with st.spinner("🔄 Préparation des données tabulaires..."):
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
                        # Reshape pour les modèles classiques si nécessaire
                        if len(st.session_state.X_train.shape) > 2:
                            X_train_flat = st.session_state.X_train.reshape(st.session_state.X_train.shape[0], -1)
                            X_test_flat = st.session_state.X_test.reshape(st.session_state.X_test.shape[0], -1)
                        else:
                            X_train_flat = st.session_state.X_train
                            X_test_flat = st.session_state.X_test
                        
                        results = trainer.train_models(
                            X_train_flat,
                            X_test_flat,
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
                        st.warning("⚠ Veuillez sélectionner au moins un algorithme!")
            
            # DEEP LEARNING pour données tabulaires
            else:
                st.subheader("🧠 Deep Learning pour données tabulaires")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    architecture = st.radio(
                        "Architecture:",
                        ["default", "custom"]
                    )
                
                custom_params = {}
                if architecture == "custom":
                    with col2:
                        st.write("*Paramètres personnalisés:*")
                        custom_params['n_layers'] = st.number_input("Nombre de couches", 2, 10, 3)
                        custom_params['neurons'] = st.number_input("Neurons (1ère couche)", 32, 512, 128)
                        custom_params['dropout'] = st.slider("Dropout", 0.0, 0.5, 0.3)
                
                epochs = st.slider("Epochs", 10, 200, 100)
                batch_size = st.selectbox("Batch size", [16, 32, 64, 128], index=1)
                
                if st.button("🚀 Lancer l'entraînement DL", type="primary"):
                    with st.spinner("🔄 Entraînement en cours..."):
                        dl_trainer = DeepLearningTrainer(st.session_state.problem_type)
                        
                        input_dim = st.session_state.X_train.shape[1]
                        output_dim = len(np.unique(st.session_state.y_train)) if st.session_state.problem_type == 'classification' else 1
                        
                        model = dl_trainer.build_model(input_dim, output_dim, architecture, custom_params)
                        
                        # Entraînement
                        progress_bar = st.progress(0)
                        history = dl_trainer.train(
                            st.session_state.X_train,
                            st.session_state.y_train,
                            st.session_state.X_test,
                            st.session_state.y_test,
                            epochs=epochs,
                            batch_size=batch_size
                        )
                        progress_bar.progress(100)
                        
                        st.session_state.dl_trainer = dl_trainer
                        st.session_state.dl_history = history
                        st.session_state.trained = True
                        
                    st.success("✅ Entraînement terminé!")
                    st.balloons()
        
        else:  # Images
            st.subheader("🖼 Deep Learning pour Images")
            
            # Préparation des images
            if 'X_train' not in st.session_state:
                with st.spinner("🔄 Préparation des images..."):
                    image_processor = st.session_state.image_processor
                    image_processor.img_size = st.session_state.img_size
                    
                    if hasattr(st.session_state, 'uploaded_files'):
                        images, labels = image_processor.load_images_from_folders(st.session_state.uploaded_files)
                    else:
                        images, labels = image_processor.load_images_from_zip(st.session_state.zip_file)
                    
                    if len(images) == 0:
                        st.error("❌ Aucune image valide n'a été trouvée!")
                        return
                    
                    X_train, X_test, y_train, y_test = image_processor.preprocess_images(
                        images, labels, st.session_state.test_size
                    )
                    
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.image_processor = image_processor
                    st.session_state.labels = labels
                    
                    st.success(f"✅ {len(images)} images préparées! Classes: {np.unique(labels)}")
            
            # Configuration CNN
            col1, col2 = st.columns(2)
            
            with col1:
                architecture = st.radio(
                    "Architecture CNN:",
                    ["default", "custom"]
                )
            
            custom_params = {}
            if architecture == "custom":
                with col2:
                    st.write("*Paramètres CNN personnalisés:*")
                    custom_params['n_conv_layers'] = st.number_input("Nombre de couches convolutionnelles", 2, 5, 3)
                    custom_params['filters'] = st.number_input("Filtres (1ère couche)", 16, 128, 32)
                    custom_params['dense_neurons'] = st.number_input("Neurons couches denses", 32, 256, 64)
                    custom_params['dropout'] = st.slider("Dropout", 0.0, 0.7, 0.5)
            
            epochs = st.slider("Epochs", 10, 200, 50)
            batch_size = st.selectbox("Batch size", [16, 32, 64], index=1)
            
            if st.button("🚀 Lancer l'entraînement CNN", type="primary"):
                with st.spinner("🔄 Entraînement CNN en cours..."):
                    input_shape = st.session_state.X_train.shape[1:]  # (height, width, channels)
                    dl_trainer = DeepLearningTrainer(
                        problem_type='classification',
                        input_shape=input_shape
                    )
                    
                    output_dim = len(np.unique(st.session_state.y_train))
                    
                    model = dl_trainer.build_cnn_model(output_dim, architecture, custom_params)
                    
                    # Aperçu du modèle
                    st.subheader("📋 Architecture du modèle CNN")
                    model_summary = []
                    model.summary(print_fn=lambda x: model_summary.append(x))
                    st.text("\n".join(model_summary))
                    
                    # Entraînement
                    progress_bar = st.progress(0)
                    history = dl_trainer.train(
                        st.session_state.X_train,
                        st.session_state.y_train,
                        st.session_state.X_test,
                        st.session_state.y_test,
                        epochs=epochs,
                        batch_size=batch_size
                    )
                    progress_bar.progress(100)
                    
                    st.session_state.dl_trainer = dl_trainer
                    st.session_state.dl_history = history
                    st.session_state.trained = True
                    
                st.success("✅ Entraînement CNN terminé!")
                st.balloons()
    
    # ==================== ÉTAPE 4: RÉSULTATS ====================
    elif step == "4️⃣ Résultats":
        if not st.session_state.trained:
            st.warning("⚠ Veuillez d'abord entraîner un modèle!")
            return
        
        st.header("📊 Résultats et Évaluation")
        
        # RÉSULTATS ML CLASSIQUE (seulement pour données tabulaires)
        if st.session_state.model_type == "ML Classique" and "Tabulaire" in st.session_state.data_type:
            results = st.session_state.ml_results
            
            # Tableau comparatif
            st.subheader("📈 Comparaison des modèles")
            
            metrics_df = pd.DataFrame({
                model: res['metrics']
                for model, res in results.items()
            }).T
            
            st.dataframe(metrics_df.style.highlight_max(axis=0, color='#667eea'), use_container_width=True)
            
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
                im = ax.imshow(cm, interpolation='nearest', cmap='plasma')
                ax.figure.colorbar(im, ax=ax)
                
                # Ajouter les annotations
                thresh = cm.max() / 2.
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, format(cm[i, j], 'd'),
                               ha="center", va="center",
                               color="white" if cm[i, j] > thresh else "black",
                               fontweight='bold')
                
                ax.set_title(f'Matrice de Confusion - {best_model}', fontweight='bold', pad=20)
                ax.set_xlabel('Prédictions', fontweight='bold')
                ax.set_ylabel('Vraies valeurs', fontweight='bold')
                ax.set_xticks(range(len(cm)))
                ax.set_yticks(range(len(cm)))
                st.pyplot(fig)
            
            else:  # regression
                # Prédictions vs Réelles
                y_pred = results[best_model]['predictions']
                
                fig, ax = plt.subplots(figsize=(10, 6))
                scatter = ax.scatter(st.session_state.y_test, y_pred, alpha=0.6, c='#667eea', s=50)
                ax.plot([st.session_state.y_test.min(), st.session_state.y_test.max()],
                       [st.session_state.y_test.min(), st.session_state.y_test.max()],
                       'r--', lw=2, alpha=0.8)
                ax.set_xlabel('Valeurs réelles', fontweight='bold')
                ax.set_ylabel('Prédictions', fontweight='bold')
                ax.set_title(f'Prédictions vs Réelles - {best_model}', fontweight='bold', pad=20)
                ax.grid(True, alpha=0.3)
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
        
        # RÉSULTATS DEEP LEARNING (pour données tabulaires et images)
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
            axes[0].plot(history['loss'], label='Train Loss', linewidth=2, color='#667eea')
            axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2, color='#764ba2')
            axes[0].set_title('Loss durant l\'entraînement', fontweight='bold')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Métrique principale
            metric_key = 'accuracy' if st.session_state.problem_type == 'classification' else 'mae'
            if metric_key in history:
                axes[1].plot(history[metric_key], label=f'Train {metric_key}', linewidth=2, color='#667eea')
                axes[1].plot(history[f'val_{metric_key}'], label=f'Val {metric_key}', linewidth=2, color='#764ba2')
                axes[1].set_title(f'{metric_key.upper()} durant l\'entraînement', fontweight='bold')
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel(metric_key.upper())
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Matrice de confusion pour la classification
            if st.session_state.problem_type == 'classification':
                st.subheader("🎯 Matrice de Confusion")
                y_pred_classes = np.argmax(dl_trainer.model.predict(st.session_state.X_test), axis=1)
                cm = confusion_matrix(st.session_state.y_test, y_pred_classes)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(cm, interpolation='nearest', cmap='plasma')
                ax.figure.colorbar(im, ax=ax)
                
                thresh = cm.max() / 2.
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, format(cm[i, j], 'd'),
                               ha="center", va="center",
                               color="white" if cm[i, j] > thresh else "black",
                               fontweight='bold')
                
                ax.set_title('Matrice de Confusion', fontweight='bold', pad=20)
                ax.set_xlabel('Prédictions', fontweight='bold')
                ax.set_ylabel('Vraies valeurs', fontweight='bold')
                st.pyplot(fig)
            
            # Téléchargement du modèle
            st.subheader("💾 Télécharger le modèle")
            if st.button("📥 Télécharger le modèle DL"):
                if "Images" in st.session_state.data_type:
                    model_name = "cnn_model.h5"
                else:
                    model_name = "dense_model.h5"
                
                dl_trainer.model.save(model_name)
                with open(model_name, 'rb') as f:
                    st.download_button(
                        label="💾 Télécharger",
                        data=f,
                        file_name=model_name,
                        mime="application/octet-stream"
                    )
        
        # Bouton pour ré-entraîner
        st.divider()
        if st.button("🔄 Ré-entraîner avec d'autres paramètres"):
            st.session_state.trained = False
            st.rerun()


if _name_ == "_main_":
    main()
