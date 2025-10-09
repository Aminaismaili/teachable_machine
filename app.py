"""
🤖 TEACHABLE MACHINE - ALL IN ONE
Application complète de Machine Learning et Deep Learning
Tout en un seul fichier !
"""

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import zipfile
import io
import pickle
import time

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# ML Models
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier, XGBRegressor

# ML Metrics
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, mean_squared_error, 
                             mean_absolute_error, r2_score)

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================================
# 📊 CLASSE 1: DATA PROCESSOR
# ============================================================================

class DataProcessor:
    """Prétraitement des données (tabulaires et images)"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.target_name = None
        self.data_type = None
        self.image_data = []
        self.image_labels = []
    
    def load_images_from_zip(self, zip_file, target_size=(224, 224)):
        """Charge images depuis ZIP"""
        self.data_type = 'image'
        self.image_data = []
        self.image_labels = []
        class_names = []
        
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                
                for file_path in file_list:
                    if file_path.endswith('/') or '/__MACOSX' in file_path or file_path.startswith('.'):
                        continue
                    
                    if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        parts = file_path.split('/')
                        class_name = parts[-2] if len(parts) >= 2 else 'unknown'
                        
                        if class_name not in class_names:
                            class_names.append(class_name)
                        
                        image_bytes = zip_ref.read(file_path)
                        image = Image.open(io.BytesIO(image_bytes))
                        
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        
                        image = image.resize(target_size)
                        img_array = np.array(image) / 255.0
                        
                        self.image_data.append(img_array)
                        self.image_labels.append(class_name)
            
            le = LabelEncoder()
            self.image_labels = le.fit_transform(self.image_labels)
            self.label_encoders['image_labels'] = le
            
            return {
                'n_images': len(self.image_data),
                'n_classes': len(class_names),
                'class_names': class_names,
                'image_shape': self.image_data[0].shape if self.image_data else None
            }
        except Exception as e:
            raise Exception(f"Erreur ZIP: {str(e)}")
    
    def load_images_from_uploads(self, uploaded_files, labels, target_size=(224, 224)):
        """Charge images individuelles"""
        self.data_type = 'image'
        self.image_data = []
        self.image_labels = []
        
        try:
            for uploaded_file, label in zip(uploaded_files, labels):
                image = Image.open(uploaded_file)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image = image.resize(target_size)
                img_array = np.array(image) / 255.0
                self.image_data.append(img_array)
                self.image_labels.append(label)
            
            le = LabelEncoder()
            self.image_labels = le.fit_transform(self.image_labels)
            self.label_encoders['image_labels'] = le
            class_names = le.classes_.tolist()
            
            return {
                'n_images': len(self.image_data),
                'n_classes': len(class_names),
                'class_names': class_names,
                'image_shape': self.image_data[0].shape if self.image_data else None
            }
        except Exception as e:
            raise Exception(f"Erreur upload: {str(e)}")
    
    def get_image_data(self):
        """Retourne images et labels"""
        if self.data_type != 'image':
            raise ValueError("Pas de données image")
        return np.array(self.image_data), np.array(self.image_labels)
    
    def get_column_info(self, df):
        """Info sur les colonnes"""
        return {
            'numeric_cols': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_cols': df.select_dtypes(include=['object', 'category']).columns.tolist(),
            'missing_values': df.isnull().sum().to_dict(),
            'shape': df.shape
        }
    
    def preprocess(self, df, target_column, problem_type='classification'):
        """Prétraitement données tabulaires"""
        df = df.copy()
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        self.feature_names = X.columns.tolist()
        self.target_name = target_column
        
        numeric_features = X.select_dtypes(include=[np.number]).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns
        
        if len(numeric_features) > 0:
            num_imputer = SimpleImputer(strategy='mean')
            X[numeric_features] = num_imputer.fit_transform(X[numeric_features])
        
        if len(categorical_features) > 0:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            X[categorical_features] = cat_imputer.fit_transform(X[categorical_features])
        
        for col in categorical_features:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        if problem_type == 'classification':
            if y.dtype == 'object' or y.dtype.name == 'category':
                le_target = LabelEncoder()
                y = le_target.fit_transform(y)
                self.label_encoders['target'] = le_target
        
        if y.isnull().any():
            y = y.fillna(y.mean() if problem_type == 'regression' else y.mode()[0])
        
        return X.values, y.values
    
    def split_data(self, X, y, test_size=0.2, random_state=42, is_classification=True):
        """Split train/test"""
        # Stratify seulement pour classification avec assez d'échantillons par classe
        stratify_param = None
        if is_classification:
            unique, counts = np.unique(y, return_counts=True)
            # Stratify seulement si chaque classe a au moins 2 échantillons
            if len(unique) > 1 and counts.min() >= 2:
                stratify_param = y
        
        return train_test_split(X, y, test_size=test_size, random_state=random_state,
                               stratify=stratify_param)
    
    def scale_features(self, X_train, X_test):
        """Normalisation"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled


# ============================================================================
# 🎓 CLASSE 2: CLASSICAL ML TRAINER
# ============================================================================

class ClassicalMLTrainer:
    """Entraînement modèles ML classiques"""
    
    def __init__(self, problem_type='classification'):
        self.problem_type = problem_type
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
    
    def get_available_models(self):
        """Liste des modèles disponibles"""
        if self.problem_type == 'classification':
            return {
                'Random Forest': RandomForestClassifier(random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(random_state=42),
                'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
                'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
                'SVM': SVC(random_state=42),
                'KNN': KNeighborsClassifier(),
                'Decision Tree': DecisionTreeClassifier(random_state=42),
                'Naive Bayes': GaussianNB(),
                'AdaBoost': AdaBoostClassifier(random_state=42)
            }
        else:
            return {
                'Random Forest': RandomForestRegressor(random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(random_state=42),
                'XGBoost': XGBRegressor(random_state=42),
                'Linear Regression': LinearRegression(),
                'Ridge': Ridge(random_state=42),
                'Lasso': Lasso(random_state=42),
                'SVR': SVR(),
                'KNN': KNeighborsRegressor(),
                'Decision Tree': DecisionTreeRegressor(random_state=42),
                'AdaBoost': AdaBoostRegressor(random_state=42)
            }
    
    def train_single_model(self, model_name, X_train, y_train, X_test, y_test):
        """Entraîne un modèle"""
        available_models = self.get_available_models()
        if model_name not in available_models:
            raise ValueError(f"Modèle {model_name} non disponible")
        
        model = available_models[model_name]
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        if self.problem_type == 'classification':
            metrics = {
                'train_accuracy': accuracy_score(y_train, y_pred_train),
                'test_accuracy': accuracy_score(y_test, y_pred_test),
                'precision': precision_score(y_test, y_pred_test, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred_test, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test, y_pred_test, average='weighted', zero_division=0),
                'confusion_matrix': confusion_matrix(y_test, y_pred_test).tolist(),
                'training_time': training_time
            }
        else:
            metrics = {
                'train_r2': r2_score(y_train, y_pred_train),
                'test_r2': r2_score(y_test, y_pred_test),
                'mse': mean_squared_error(y_test, y_pred_test),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'mae': mean_absolute_error(y_test, y_pred_test),
                'training_time': training_time
            }
        
        self.models[model_name] = model
        self.results[model_name] = metrics
        return model, metrics
    
    def train_all_models(self, X_train, y_train, X_test, y_test, selected_models=None):
        """Entraîne tous les modèles sélectionnés"""
        available_models = self.get_available_models()
        if selected_models is None:
            selected_models = list(available_models.keys())
        
        results = {}
        for model_name in selected_models:
            try:
                model, metrics = self.train_single_model(
                    model_name, X_train, y_train, X_test, y_test
                )
                results[model_name] = metrics
            except Exception as e:
                results[model_name] = {'error': str(e)}
        
        self._find_best_model()
        return results
    
    def _find_best_model(self):
        """Trouve le meilleur modèle"""
        if not self.results:
            return
        
        if self.problem_type == 'classification':
            best_name = max(self.results.items(), 
                          key=lambda x: x[1].get('test_accuracy', 0))[0]
        else:
            best_name = min(self.results.items(), 
                          key=lambda x: x[1].get('rmse', float('inf')))[0]
        
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
    
    def get_best_model(self):
        """Retourne le meilleur modèle"""
        if self.best_model is None:
            return None, None, None
        return self.best_model_name, self.best_model, self.results[self.best_model_name]
    
    def get_results_dataframe(self):
        """Résultats en DataFrame"""
        if not self.results:
            return pd.DataFrame()
        
        df_results = pd.DataFrame(self.results).T
        if self.problem_type == 'classification':
            cols = ['test_accuracy', 'precision', 'recall', 'f1_score', 'training_time']
            df_results = df_results[cols].sort_values('test_accuracy', ascending=False)
        else:
            cols = ['test_r2', 'rmse', 'mae', 'training_time']
            df_results = df_results[cols].sort_values('rmse', ascending=True)
        
        return df_results


# ============================================================================
# 🧠 CLASSE 3: DEEP LEARNING TRAINER
# ============================================================================

class DeepLearningTrainer:
    """Entraînement réseaux de neurones"""
    
    def __init__(self, problem_type='classification', network_type='simple'):
        self.problem_type = problem_type
        self.network_type = network_type
        self.model = None
        self.history = None
        self.config = None
    
    def create_simple_nn(self, input_shape, num_classes=None, config='default'):
        """Crée un Simple NN"""
        if config == 'default':
            config = {
                'hidden_layers': [128, 64, 32],
                'activation': 'relu',
                'use_dropout': True,
                'dropout_rate': 0.3,
                'use_batch_norm': True
            }
        
        self.config = config
        model = models.Sequential()
        model.add(layers.Input(shape=(input_shape,)))
        
        for units in config['hidden_layers']:
            model.add(layers.Dense(units, activation=config['activation']))
            if config['use_dropout']:
                model.add(layers.Dropout(config['dropout_rate']))
            if config['use_batch_norm']:
                model.add(layers.BatchNormalization())
        
        if self.problem_type == 'classification':
            if num_classes == 2:
                model.add(layers.Dense(1, activation='sigmoid'))
            else:
                model.add(layers.Dense(num_classes, activation='softmax'))
        else:
            model.add(layers.Dense(1))
        
        self.model = model
        return model
    
    def create_cnn(self, input_shape, num_classes=None, config='default'):
        """Crée un CNN"""
        if config == 'default':
            config = {
                'conv_layers': [32, 64, 128],
                'dense_layers': [256, 128],
                'use_dropout': True,
                'dropout_rate': 0.5
            }
        
        self.config = config
        model = models.Sequential()
        model.add(layers.Input(shape=input_shape))
        
        for filters in config['conv_layers']:
            model.add(layers.Conv2D(filters, (3, 3), activation='relu', padding='same'))
            model.add(layers.Conv2D(filters, (3, 3), activation='relu', padding='same'))
            model.add(layers.MaxPooling2D((2, 2)))
            if config['use_dropout']:
                model.add(layers.Dropout(0.25))
        
        model.add(layers.Flatten())
        for units in config['dense_layers']:
            model.add(layers.Dense(units, activation='relu'))
            if config['use_dropout']:
                model.add(layers.Dropout(config['dropout_rate']))
        
        if self.problem_type == 'classification':
            if num_classes == 2:
                model.add(layers.Dense(1, activation='sigmoid'))
            else:
                model.add(layers.Dense(num_classes, activation='softmax'))
        else:
            model.add(layers.Dense(1))
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.001):
        """Compile le modèle"""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        if self.problem_type == 'classification':
            if self.model.output_shape[-1] == 1:
                loss = 'binary_crossentropy'
            else:
                loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        else:
            loss = 'mse'
            metrics = ['mae']
        
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32, verbose=1):
        """Entraîne le modèle"""
        callbacks = [
            EarlyStopping(monitor='val_loss' if X_val is not None else 'loss', 
                         patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss' if X_val is not None else 'loss', 
                             factor=0.5, patience=5, min_lr=1e-7)
        ]
        
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        validation_split = 0.2 if validation_data is None else 0.0
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        return self.history
    
    def evaluate(self, X_test, y_test):
        """Évalue le modèle"""
        y_pred = self.model.predict(X_test, verbose=0)
        
        if self.problem_type == 'classification':
            if y_pred.shape[-1] == 1:
                y_pred_class = (y_pred > 0.5).astype(int).flatten()
            else:
                y_pred_class = np.argmax(y_pred, axis=1)
            
            return {
                'accuracy': accuracy_score(y_test, y_pred_class),
                'precision': precision_score(y_test, y_pred_class, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred_class, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test, y_pred_class, average='weighted', zero_division=0)
            }
        else:
            y_pred = y_pred.flatten()
            return {
                'r2_score': r2_score(y_test, y_pred),
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred)
            }
    
    def get_model_summary(self):
        """Résumé du modèle"""
        if self.model is None:
            return "Aucun modèle créé"
        summary = []
        self.model.summary(print_fn=lambda x: summary.append(x))
        return '\n'.join(summary)


# ============================================================================
# 🎨 INTERFACE STREAMLIT
# ============================================================================

# Configuration
st.set_page_config(
    page_title="Teachable Machine",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .step-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 20px;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialisation session_state
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'df' not in st.session_state:
    st.session_state.df = None
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()

# Header
st.markdown('<h1 class="main-header">🤖 Teachable Machine</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Créez vos modèles ML/DL en quelques clics</p>', unsafe_allow_html=True)

# Progress bar
progress_percentage = (st.session_state.step - 1) / 3
st.progress(progress_percentage)

# Sidebar
with st.sidebar:
    st.markdown("### 📍 Navigation")
    
    steps = {1: "📤 Upload Data", 2: "⚙️ Configuration", 3: "🚀 Training", 4: "📊 Results"}
    
    for step_num, step_name in steps.items():
        if st.button(step_name, key=f"nav_{step_num}", use_container_width=True):
            if step_num <= st.session_state.step + 1:
                st.session_state.step = step_num
                st.rerun()
    
    st.markdown("---")
    st.info("""
    **Étapes:**
    1. Upload données/images
    2. Configurez
    3. Entraînez
    4. Résultats
    """)

# ============================================================================
# ÉTAPE 1: UPLOAD DATA
# ============================================================================

if st.session_state.step == 1:
    st.markdown('<div class="step-badge">ÉTAPE 1: Upload Data</div>', unsafe_allow_html=True)
    
    data_format = st.radio(
        "Type de données",
        ["📄 Données tabulaires (CSV/Excel)", "🖼️ Images (CNN)"],
        horizontal=True
    )
    
    st.markdown("---")
    
    # DONNÉES TABULAIRES
    if "tabulaires" in data_format:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader("Fichier CSV/Excel", type=['csv', 'xlsx', 'xls'])
            
            if uploaded_file:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        st.session_state.df = pd.read_csv(uploaded_file)
                    else:
                        st.session_state.df = pd.read_excel(uploaded_file)
                    
                    st.session_state.data_processor.data_type = 'tabular'
                    st.success(f"✅ {uploaded_file.name} chargé!")
                    st.dataframe(st.session_state.df.head(10), use_container_width=True)
                    
                    if st.button("➡️ Continuer", type="primary", use_container_width=True):
                        st.session_state.step = 2
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ Erreur: {e}")
        
        with col2:
            if st.session_state.df is not None:
                st.metric("📏 Lignes", st.session_state.df.shape[0])
                st.metric("📊 Colonnes", st.session_state.df.shape[1])
                st.metric("⚠️ Manquantes", st.session_state.df.isnull().sum().sum())
    
    # IMAGES
    else:
        upload_method = st.radio("Méthode", ["📦 ZIP", "📁 Individuel"], horizontal=True)
        
        if "ZIP" in upload_method:
            st.info("""
            **Structure ZIP:**
            ```
            images.zip/
            ├── classe1/img1.jpg
            ├── classe2/img2.jpg
            ```
            """)
            
            zip_file = st.file_uploader("Upload ZIP", type=['zip'])
            
            if zip_file:
                col1, col2 = st.columns(2)
                with col1:
                    img_height = st.number_input("Hauteur (px)", value=224, min_value=32)
                with col2:
                    img_width = st.number_input("Largeur (px)", value=224, min_value=32)
                
                if st.button("📥 Charger", type="primary"):
                    with st.spinner("Chargement..."):
                        try:
                            info = st.session_state.data_processor.load_images_from_zip(
                                zip_file, (img_height, img_width)
                            )
                            st.session_state.image_info = info
                            st.success(f"✅ {info['n_images']} images chargées!")
                            
                            col1, col2, col3 = st.columns(3)
                            col1.metric("🖼️ Images", info['n_images'])
                            col2.metric("🏷️ Classes", info['n_classes'])
                            col3.metric("📐 Shape", str(info['image_shape']))
                            
                            st.write("**Classes:**", ", ".join(info['class_names']))
                            
                            X, y = st.session_state.data_processor.get_image_data()
                            cols = st.columns(5)
                            for i in range(min(5, len(X))):
                                cols[i].image(X[i], caption=info['class_names'][y[i]])
                            
                            if st.button("➡️ Continuer", type="primary", key="zip_next"):
                                st.session_state.step = 2
                                st.rerun()
                        except Exception as e:
                            st.error(f"❌ {e}")
        
        else:  # Images individuelles
            uploaded_images = st.file_uploader(
                "Upload images", 
                type=['jpg', 'jpeg', 'png'], 
                accept_multiple_files=True
            )
            
            if uploaded_images:
                st.success(f"✅ {len(uploaded_images)} images")
                
                col1, col2 = st.columns(2)
                with col1:
                    img_height = st.number_input("Hauteur", value=224, key="h2")
                with col2:
                    img_width = st.number_input("Largeur", value=224, key="w2")
                
                st.markdown("### 🏷️ Labels")
                labels = []
                cols_per_row = 3
                
                for idx in range(0, len(uploaded_images), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for col_idx, col in enumerate(cols):
                        img_idx = idx + col_idx
                        if img_idx < len(uploaded_images):
                            with col:
                                image = Image.open(uploaded_images[img_idx])
                                st.image(image, use_column_width=True)
                                label = st.text_input("Label", key=f"lbl_{img_idx}")
                                labels.append(label)
                
                if st.button("📥 Charger", type="primary", key="indiv"):
                    if all(labels):
                        try:
                            info = st.session_state.data_processor.load_images_from_uploads(
                                uploaded_images, labels, (img_height, img_width)
                            )
                            st.session_state.image_info = info
                            st.success("✅ Images chargées!")
                            
                            if st.button("➡️ Continuer", type="primary", key="indiv_next"):
                                st.session_state.step = 2
                                st.rerun()
                        except Exception as e:
                            st.error(f"❌ {e}")
                    else:
                        st.warning("⚠️ Assignez tous les labels")

# ============================================================================
# ÉTAPE 2: CONFIGURATION
# ============================================================================

elif st.session_state.step == 2:
    if st.session_state.df is None and st.session_state.data_processor.data_type != 'image':
        st.warning("⚠️ Chargez d'abord des données")
        if st.button("← Retour"):
            st.session_state.step = 1
            st.rerun()
    else:
        st.markdown('<div class="step-badge">ÉTAPE 2: Configuration</div>', unsafe_allow_html=True)
        
        # CONFIGURATION TABULAIRE
        if st.session_state.data_processor.data_type == 'tabular':
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🎯 Problème")
                problem_type = st.selectbox("Type", ["Classification", "Régression"])
                st.session_state.problem_type = problem_type.lower()
                
                target_column = st.selectbox("Cible", st.session_state.df.columns.tolist())
                st.session_state.target_column = target_column
                st.info(f"Valeurs uniques: {st.session_state.df[target_column].nunique()}")
            
            with col2:
                st.markdown("### 🤖 Modèle")
                model_type = st.radio("Type", ["ML Classique", "Deep Learning"], horizontal=True)
                st.session_state.model_type = model_type
                
                st.markdown("### ✂️ Split")
                test_size = st.slider("Test (%)", 10, 40, 20, 5)
                st.session_state.test_size = test_size / 100
                st.markdown(f"🟢 Train: {100-test_size}% | 🔵 Test: {test_size}%")
            
            if st.button("🔄 Prétraiter", type="primary", use_container_width=True):
                with st.spinner("Prétraitement..."):
                    try:
                        X, y = st.session_state.data_processor.preprocess(
                            st.session_state.df, target_column, st.session_state.problem_type
                        )
                        X_train, X_test, y_train, y_test = st.session_state.data_processor.split_data(
                            X, y, st.session_state.test_size, 
                            is_classification=(st.session_state.problem_type == 'classification')
                        )
                        X_train_scaled, X_test_scaled = st.session_state.data_processor.scale_features(
                            X_train, X_test
                        )
                        
                        st.session_state.X_train = X_train_scaled
                        st.session_state.X_test = X_test_scaled
                        st.session_state.y_train = y_train
                        st.session_state.y_test = y_test
                        
                        st.success("✅ Données prêtes!")
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("🟢 Train", len(X_train))
                        col2.metric("🔵 Test", len(X_test))
                        col3.metric("📊 Features", X_train.shape[1])
                        
                        if st.button("➡️ Training", type="primary"):
                            st.session_state.step = 3
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ {e}")
        
        # CONFIGURATION IMAGES (CNN)
        else:
            info = st.session_state.image_info
            st.markdown("### 🖼️ Configuration CNN")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("🖼️ Images", info['n_images'])
            col2.metric("🏷️ Classes", info['n_classes'])
            col3.metric("📐 Shape", str(info['image_shape']))
            
            st.write("**Classes:**", ", ".join(info['class_names']))
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### ✂️ Split")
                test_size = st.slider("Test (%)", 10, 40, 20, 5, key="split_img")
                st.session_state.test_size = test_size / 100
                st.markdown(f"🟢 Train: {100-test_size}% | 🔵 Test: {test_size}%")
            
            with col2:
                st.markdown("### 🧠 Architecture")
                st.info("**Classification d'images**")
                st.session_state.problem_type = 'classification'
                st.session_state.model_type = "Deep Learning"
                
                cnn_preset = st.selectbox("Config", ["Légère", "Moyenne", "Profonde"])
                st.session_state.cnn_preset = cnn_preset
            
            if st.button("🔄 Préparer CNN", type="primary", use_container_width=True):
                with st.spinner("Préparation..."):
                    try:
                        X, y = st.session_state.data_processor.get_image_data()
                        X_train, X_test, y_train, y_test = st.session_state.data_processor.split_data(
                            X, y, st.session_state.test_size
                        )
                        
                        st.session_state.X_train = X_train
                        st.session_state.X_test = X_test
                        st.session_state.y_train = y_train
                        st.session_state.y_test = y_test
                        
                        st.success("✅ Prêt pour CNN!")
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("🟢 Train", len(X_train))
                        col2.metric("🔵 Test", len(X_test))
                        col3.metric("🏷️ Classes", info['n_classes'])
                        
                        st.markdown("### 👀 Aperçu")
                        cols = st.columns(5)
                        for i in range(min(5, len(X_train))):
                            cols[i].image(X_train[i], caption=info['class_names'][y_train[i]])
                        
                        if st.button("➡️ Training CNN", type="primary", key="cnn_next"):
                            st.session_state.step = 3
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ {e}")

# ============================================================================
# ÉTAPE 3: TRAINING
# ============================================================================

elif st.session_state.step == 3:
    if not hasattr(st.session_state, 'X_train'):
        st.warning("⚠️ Configurez d'abord les données")
        if st.button("← Retour"):
            st.session_state.step = 2
            st.rerun()
    else:
        st.markdown('<div class="step-badge">ÉTAPE 3: Training</div>', unsafe_allow_html=True)
        
        # ML CLASSIQUE
        if st.session_state.model_type == "ML Classique":
            st.markdown("### 🎓 ML Classique")
            
            if 'ml_trainer' not in st.session_state:
                st.session_state.ml_trainer = ClassicalMLTrainer(st.session_state.problem_type)
            
            available_models = list(st.session_state.ml_trainer.get_available_models().keys())
            
            col1, col2 = st.columns([3, 1])
            with col1:
                selected_models = st.multiselect(
                    "Modèles à entraîner",
                    available_models,
                    default=available_models[:3]
                )
            with col2:
                st.metric("Sélectionnés", len(selected_models))
            
            if st.button("🚀 Entraîner", type="primary", use_container_width=True):
                if not selected_models:
                    st.warning("⚠️ Sélectionnez au moins un modèle")
                else:
                    progress_bar = st.progress(0)
                    status = st.empty()
                    
                    results = {}
                    for i, model_name in enumerate(selected_models):
                        status.text(f"Training {model_name}...")
                        progress_bar.progress((i + 1) / len(selected_models))
                        
                        try:
                            model, metrics = st.session_state.ml_trainer.train_single_model(
                                model_name,
                                st.session_state.X_train,
                                st.session_state.y_train,
                                st.session_state.X_test,
                                st.session_state.y_test
                            )
                            results[model_name] = metrics
                        except Exception as e:
                            st.error(f"❌ {model_name}: {e}")
                    
                    status.text("✅ Terminé!")
                    st.session_state.results = results
                    st.session_state.step = 4
                    st.rerun()
        
        # DEEP LEARNING
        else:
            st.markdown("### 🧠 Deep Learning")
            
            is_cnn = st.session_state.data_processor.data_type == 'image'
            
            if is_cnn:
                st.info("🖼️ Mode CNN activé")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    epochs = st.slider("Époques", 10, 200, 50)
                    batch_size = st.selectbox("Batch size", [16, 32, 64], index=1)
                with col2:
                    learning_rate = st.select_slider("Learning rate", [0.0001, 0.0005, 0.001, 0.005], value=0.001)
                    use_dropout = st.checkbox("Dropout", value=True)
                with col3:
                    dropout_rate = st.slider("Dropout rate", 0.1, 0.5, 0.3) if use_dropout else 0.0
                
                preset = st.session_state.get('cnn_preset', 'Moyenne')
                if "Légère" in preset:
                    conv_config = [32, 64]
                    dense_config = [128]
                elif "Profonde" in preset:
                    conv_config = [32, 64, 128, 256]
                    dense_config = [512, 256]
                else:
                    conv_config = [32, 64, 128]
                    dense_config = [256, 128]
                
                cnn_config = {
                    'conv_layers': conv_config,
                    'dense_layers': dense_config,
                    'use_dropout': use_dropout,
                    'dropout_rate': dropout_rate
                }
                
                st.code(f"Conv: {conv_config}\nDense: {dense_config}")
            
            else:
                # Simple NN
                config_type = st.selectbox("Config", ["Défaut", "Personnalisée"])
                
                if config_type == "Personnalisée":
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        hidden_layers = st.text_input("Couches", "128,64,32")
                        epochs = st.slider("Époques", 10, 500, 100)
                    with col2:
                        batch_size = st.selectbox("Batch", [16, 32, 64, 128], index=1)
                        learning_rate = st.select_slider("LR", [0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001)
                    with col3:
                        use_dropout = st.checkbox("Dropout", value=True, key="nn_drop")
                        dropout_rate = st.slider("Rate", 0.1, 0.5, 0.3) if use_dropout else 0.0
                    
                    custom_config = {
                        'hidden_layers': [int(x.strip()) for x in hidden_layers.split(',')],
                        'activation': 'relu',
                        'use_dropout': use_dropout,
                        'dropout_rate': dropout_rate,
                        'use_batch_norm': True
                    }
                else:
                    epochs = 100
                    batch_size = 32
                    learning_rate = 0.001
                    custom_config = 'default'
            
            button_text = "🚀 Entraîner CNN" if is_cnn else "🚀 Entraîner NN"
            
            if st.button(button_text, type="primary", use_container_width=True):
                with st.spinner("Entraînement..."):
                    try:
                        network_type = 'cnn' if is_cnn else 'simple'
                        st.session_state.dl_trainer = DeepLearningTrainer(
                            st.session_state.problem_type, network_type
                        )
                        
                        num_classes = len(np.unique(st.session_state.y_train))
                        
                        if is_cnn:
                            input_shape = st.session_state.X_train.shape[1:]
                            st.session_state.dl_trainer.create_cnn(input_shape, num_classes, cnn_config)
                        else:
                            input_shape = st.session_state.X_train.shape[1]
                            st.session_state.dl_trainer.create_simple_nn(
                                input_shape, num_classes, 
                                custom_config if config_type == "Personnalisée" else 'default'
                            )
                        
                        st.session_state.dl_trainer.compile_model(learning_rate)
                        
                        with st.expander("📐 Architecture"):
                            st.text(st.session_state.dl_trainer.get_model_summary())
                        
                        progress = st.progress(0)
                        status = st.empty()
                        status.text("🔄 Entraînement...")
                        
                        history = st.session_state.dl_trainer.train(
                            st.session_state.X_train,
                            st.session_state.y_train,
                            st.session_state.X_test,
                            st.session_state.y_test,
                            epochs=epochs,
                            batch_size=batch_size,
                            verbose=0
                        )
                        
                        progress.progress(100)
                        status.text("✅ Terminé!")
                        
                        metrics = st.session_state.dl_trainer.evaluate(
                            st.session_state.X_test,
                            st.session_state.y_test
                        )
                        
                        st.session_state.dl_results = metrics
                        st.session_state.dl_history = history.history
                        
                        st.success("✅ Entraînement terminé!")
                        
                        if is_cnn or st.session_state.problem_type == 'classification':
                            st.metric("🎯 Accuracy", f"{metrics['accuracy']:.4f}")
                        else:
                            st.metric("📊 R²", f"{metrics['r2_score']:.4f}")
                        
                        if st.button("➡️ Résultats", type="primary"):
                            st.session_state.step = 4
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ {e}")
                        st.exception(e)

# ============================================================================
# ÉTAPE 4: RESULTS
# ============================================================================

elif st.session_state.step == 4:
    st.markdown('<div class="step-badge">ÉTAPE 4: Résultats</div>', unsafe_allow_html=True)
    
    # RÉSULTATS ML CLASSIQUE
    if hasattr(st.session_state, 'results') and st.session_state.results:
        st.markdown("### 📊 Résultats ML Classique")
        
        results_df = st.session_state.ml_trainer.get_results_dataframe()
        st.dataframe(results_df.style.highlight_max(axis=0), use_container_width=True)
        
        best_name, best_model, best_metrics = st.session_state.ml_trainer.get_best_model()
        st.markdown(f"### 🏆 Meilleur: **{best_name}**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        if st.session_state.problem_type == 'classification':
            col1.metric("Accuracy", f"{best_metrics['test_accuracy']:.4f}")
            col2.metric("Precision", f"{best_metrics['precision']:.4f}")
            col3.metric("Recall", f"{best_metrics['recall']:.4f}")
            col4.metric("F1", f"{best_metrics['f1_score']:.4f}")
            
            # Confusion Matrix
            if 'confusion_matrix' in best_metrics:
                st.markdown("#### 🎯 Matrice de confusion")
                cm = np.array(best_metrics['confusion_matrix'])
                fig = px.imshow(cm, text_auto=True, aspect="auto",
                               labels=dict(x="Prédiction", y="Réel"),
                               color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)
        else:
            col1.metric("R²", f"{best_metrics['test_r2']:.4f}")
            col2.metric("RMSE", f"{best_metrics['rmse']:.4f}")
            col3.metric("MAE", f"{best_metrics['mae']:.4f}")
        
        # Comparaison
        st.markdown("#### 📈 Comparaison")
        metric = 'test_accuracy' if st.session_state.problem_type == 'classification' else 'test_r2'
        fig = px.bar(results_df, y=results_df.index, x=metric, orientation='h')
        st.plotly_chart(fig, use_container_width=True)
        
        # Download
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Télécharger modèle", use_container_width=True):
                model_bytes = pickle.dumps(best_model)
                st.download_button("📥 Download", model_bytes, 
                                 f"{best_name}_model.pkl", "application/octet-stream")
        with col2:
            if st.button("🔄 Ré-entraîner", use_container_width=True):
                st.session_state.step = 3
                st.rerun()
    
    # RÉSULTATS DEEP LEARNING
    if hasattr(st.session_state, 'dl_results'):
        st.markdown("### 🧠 Résultats Deep Learning")
        
        metrics = st.session_state.dl_results
        
        col1, col2, col3, col4 = st.columns(4)
        if st.session_state.problem_type == 'classification':
            col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            col2.metric("Precision", f"{metrics['precision']:.4f}")
            col3.metric("Recall", f"{metrics['recall']:.4f}")
            col4.metric("F1", f"{metrics['f1_score']:.4f}")
        else:
            col1.metric("R²", f"{metrics['r2_score']:.4f}")
            col2.metric("RMSE", f"{metrics['rmse']:.4f}")
            col3.metric("MAE", f"{metrics['mae']:.4f}")
        
        # Courbes d'apprentissage
        if hasattr(st.session_state, 'dl_history'):
            st.markdown("#### 📈 Courbes d'apprentissage")
            
            history = st.session_state.dl_history
            fig = make_subplots(rows=1, cols=2, subplot_titles=('Loss', 'Metric'))
            
            # Loss
            fig.add_trace(go.Scatter(y=history['loss'], name='Train Loss'), row=1, col=1)
            if 'val_loss' in history:
                fig.add_trace(go.Scatter(y=history['val_loss'], name='Val Loss'), row=1, col=1)
            
            # Metric
            metric_key = 'accuracy' if st.session_state.problem_type == 'classification' else 'mae'
            if metric_key in history:
                fig.add_trace(go.Scatter(y=history[metric_key], name=f'Train'), row=1, col=2)
            if f'val_{metric_key}' in history:
                fig.add_trace(go.Scatter(y=history[f'val_{metric_key}'], name=f'Val'), row=1, col=2)
            
            fig.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        
        # Actions
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Sauvegarder", use_container_width=True):
                st.session_state.dl_trainer.model.save("model.h5")
                st.success("✅ Modèle sauvegardé!")
        with col2:
            if st.button("🔄 Ré-entraîner", use_container_width=True, key="dl_retrain"):
                st.session_state.step = 3
                st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🤖 <strong>Teachable Machine All-in-One</strong></p>
    <p style='font-size: 0.8rem;'>ML Classique + Deep Learning + CNN</p>
</div>
""", unsafe_allow_html=True)
