"""
Teachable Machine - Application complète
Tout en un fichier Python avec Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import pickle
import json

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, roc_curve
)

# Classification algorithms
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    AdaBoostClassifier, ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# Regression algorithms
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    AdaBoostRegressor, ExtraTreesRegressor
)
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    BayesianRidge, HuberRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Configuration
st.set_page_config(
    page_title="Teachable Machine Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== STYLES CSS ====================
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to bottom right, #f8f9fa, #e9ecef);
    }
    h1 {
        color: #667eea;
        font-weight: bold;
        text-align: center;
    }
    h2 {
        color: #764ba2;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(to right, #667eea, #764ba2);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 15px;
        border: none;
        font-size: 16px;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    .metric-box {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)


# ==================== CLASSE DATA PROCESSOR ====================
class DataProcessor:
    """Classe pour gérer le preprocessing des données"""
    
    def __init__(self):
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = None
        self.target_encoder = None
        self.imputer = None
        
    def load_data(self, file_path=None, dataframe=None):
        """Charge les données"""
        try:
            if dataframe is not None:
                return dataframe.copy(), None
                
            if file_path is None:
                return None, "Aucun fichier fourni"
            
            if file_path.name.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.name.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file_path)
            elif file_path.name.endswith('.json'):
                df = pd.read_json(file_path)
            else:
                return None, "Format non supporté"
            
            return df, None
            
        except Exception as e:
            return None, f"Erreur: {str(e)}"
    
    def analyze_data(self, df):
        """Analyse les données"""
        analysis = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(df.select_dtypes(include=['object']).columns),
            'unique_values': {col: df[col].nunique() for col in df.columns}
        }
        
        if analysis['numeric_columns']:
            analysis['statistics'] = df[analysis['numeric_columns']].describe().to_dict()
        
        return analysis
    
    def handle_missing_values(self, df, strategy='mean'):
        """Gère les valeurs manquantes"""
        df = df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            self.imputer = SimpleImputer(strategy=strategy)
            df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
                df[col].fillna(mode_val, inplace=True)
        
        return df
    
    def encode_categorical(self, df, columns=None):
        """Encode les variables catégorielles"""
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=['object']).columns
        
        for col in columns:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        return df
    
    def prepare_data(self, df, target_col, test_size=0.2, random_state=42, 
                    scale_method='standard', handle_missing=True):
        """Prépare les données complètement"""
        try:
            df = df.copy()
            
            if target_col not in df.columns:
                raise ValueError(f"Colonne '{target_col}' non trouvée")
            
            if handle_missing:
                df = self.handle_missing_values(df)
            
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            self.feature_names = list(X.columns)
            
            # Encoder X
            categorical_cols = X.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                X = self.encode_categorical(X, categorical_cols)
            
            # Encoder y
            target_info = {
                'is_categorical': y.dtype == 'object' or y.nunique() < 20,
                'n_classes': None,
                'classes': None
            }
            
            if target_info['is_categorical']:
                self.target_encoder = LabelEncoder()
                y = self.target_encoder.fit_transform(y)
                target_info['n_classes'] = len(self.target_encoder.classes_)
                target_info['classes'] = list(self.target_encoder.classes_)
            
            X = X.values if isinstance(X, pd.DataFrame) else X
            y = y.values if isinstance(y, pd.Series) else y
            
            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                random_state=random_state,
                stratify=y if target_info['is_categorical'] and len(np.unique(y)) > 1 else None
            )
            
            # Scale
            if scale_method == 'standard':
                self.scaler = StandardScaler()
            else:
                self.scaler = MinMaxScaler()
            
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            
            return X_train, X_test, y_train, y_test, self.feature_names, target_info
            
        except Exception as e:
            raise Exception(f"Erreur préparation: {str(e)}")


# ==================== CLASSE ML CLASSIQUE ====================
class ClassicalMLTrainer:
    """Classe pour entraîner les modèles ML classiques"""
    
    def __init__(self, problem_type='classification'):
        self.problem_type = problem_type.lower()
        self.models = {}
        self.results = {}
        self.best_model_name = None
        self.best_model = None
        
    def get_available_models(self):
        """Retourne les modèles disponibles"""
        if self.problem_type == 'classification':
            return {
                'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                'SVM': SVC(random_state=42, probability=True),
                'KNN': KNeighborsClassifier(n_neighbors=5),
                'Decision Tree': DecisionTreeClassifier(random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(random_state=42),
                'AdaBoost': AdaBoostClassifier(random_state=42),
                'Naive Bayes': GaussianNB(),
                'Extra Trees': ExtraTreesClassifier(random_state=42, n_estimators=100),
                'Linear Discriminant': LinearDiscriminantAnalysis(),
                'Quadratic Discriminant': QuadraticDiscriminantAnalysis(),
                'Ridge Classifier': RidgeClassifier(random_state=42)
            }
        else:
            return {
                'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100),
                'Linear Regression': LinearRegression(),
                'Ridge': Ridge(random_state=42),
                'Lasso': Lasso(random_state=42),
                'ElasticNet': ElasticNet(random_state=42),
                'SVR': SVR(),
                'KNN': KNeighborsRegressor(n_neighbors=5),
                'Decision Tree': DecisionTreeRegressor(random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(random_state=42),
                'AdaBoost': AdaBoostRegressor(random_state=42),
                'Extra Trees': ExtraTreesRegressor(random_state=42, n_estimators=100),
                'Bayesian Ridge': BayesianRidge(),
                'Huber': HuberRegressor()
            }
    
    def train_single_model(self, model_name, X_train, y_train, X_test, y_test):
        """Entraîne un modèle"""
        available_models = self.get_available_models()
        
        if model_name not in available_models:
            return None
        
        try:
            model = available_models[model_name]
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            y_train_pred = model.predict(X_train)
            
            metrics = self._calculate_metrics(y_test, y_pred, y_train, y_train_pred, model, X_test)
            
            self.models[model_name] = model
            result = {
                'model': model,
                'predictions': y_pred,
                'train_predictions': y_train_pred,
                'metrics': metrics
            }
            self.results[model_name] = result
            
            return result
            
        except Exception as e:
            st.error(f"Erreur {model_name}: {str(e)}")
            return None
    
    def train_all_models(self, X_train, y_train, X_test, y_test, selected_models=None):
        """Entraîne plusieurs modèles"""
        available_models = self.get_available_models()
        
        if selected_models is None:
            selected_models = list(available_models.keys())
        
        for model_name in selected_models:
            self.train_single_model(model_name, X_train, y_train, X_test, y_test)
        
        self._identify_best_model()
        return self.results
    
    def _calculate_metrics(self, y_test, y_pred, y_train, y_train_pred, model, X_test):
        """Calcule les métriques"""
        metrics = {}
        
        if self.problem_type == 'classification':
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            metrics['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics['f1_score'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics['train_accuracy'] = accuracy_score(y_train, y_train_pred)
            metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)
            
            try:
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)
                    if len(np.unique(y_test)) == 2:
                        metrics['roc_auc'] = roc_auc_score(y_test, y_proba[:, 1])
                    else:
                        metrics['roc_auc'] = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
            except:
                metrics['roc_auc'] = None
        else:
            metrics['r2_score'] = r2_score(y_test, y_pred)
            metrics['mse'] = mean_squared_error(y_test, y_pred)
            metrics['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
            metrics['mae'] = mean_absolute_error(y_test, y_pred)
            metrics['train_r2'] = r2_score(y_train, y_train_pred)
            
            try:
                metrics['mape'] = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100
            except:
                metrics['mape'] = None
        
        return metrics
    
    def _identify_best_model(self):
        """Identifie le meilleur modèle"""
        if not self.results:
            return
        
        metric_key = 'accuracy' if self.problem_type == 'classification' else 'r2_score'
        best_score = -float('inf')
        
        for model_name, result in self.results.items():
            score = result['metrics'].get(metric_key, -float('inf'))
            if score > best_score:
                best_score = score
                self.best_model_name = model_name
                self.best_model = result['model']
    
    def get_comparison_dataframe(self):
        """Retourne un DataFrame comparatif"""
        if not self.results:
            return None
        
        comparison_data = {}
        for model_name, result in self.results.items():
            comparison_data[model_name] = result['metrics']
        
        df = pd.DataFrame(comparison_data).T
        
        if self.problem_type == 'classification':
            df = df.sort_values('accuracy', ascending=False)
        else:
            df = df.sort_values('r2_score', ascending=False)
        
        return df
    
    def get_feature_importance(self, model_name=None, feature_names=None):
        """Retourne l'importance des features"""
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_).flatten()
        else:
            return None
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importance))]
        
        df_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df_importance
    
    def save_model(self, model_name=None, path=None):
        """Sauvegarde un modèle"""
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.models:
            return None
        
        if path is None:
            path = f"{model_name.replace(' ', '_').lower()}_model.pkl"
        
        with open(path, 'wb') as f:
            pickle.dump(self.models[model_name], f)
        
        return path


# ==================== CLASSE DEEP LEARNING ====================
class NeuralNetworkTrainer:
    """Classe pour les réseaux de neurones"""
    
    def __init__(self, problem_type='classification'):
        self.problem_type = problem_type.lower()
        self.model = None
        self.history = None
        self.config = {}
        
    def build_simple_nn(self, input_dim, output_dim, n_layers=3, neurons=128, 
                       dropout=0.3, activation='relu', optimizer='adam', learning_rate=0.001):
        """Construit un réseau de neurones simple"""
        model = models.Sequential()
        
        # Première couche
        model.add(layers.Dense(neurons, activation=activation, input_shape=(input_dim,)))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout))
        
        # Couches cachées
        current_neurons = neurons
        for i in range(n_layers - 1):
            current_neurons = max(16, current_neurons // 2)
            model.add(layers.Dense(current_neurons, activation=activation))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout))
        
        # Couche de sortie
        if self.problem_type == 'classification':
            if output_dim == 2:
                model.add(layers.Dense(1, activation='sigmoid'))
                loss = 'binary_crossentropy'
                metrics = ['accuracy']
            else:
                model.add(layers.Dense(output_dim, activation='softmax'))
                loss = 'sparse_categorical_crossentropy'
                metrics = ['accuracy']
        else:
            model.add(layers.Dense(1))
            loss = 'mse'
            metrics = ['mae']
        
        # Compilation
        if optimizer == 'adam':
            opt = Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = SGD(learning_rate=learning_rate, momentum=0.9)
        else:
            opt = RMSprop(learning_rate=learning_rate)
        
        model.compile(optimizer=opt, loss=loss, metrics=metrics)
        
        self.config = {
            'input_dim': input_dim,
            'output_dim': output_dim,
            'n_layers': n_layers,
            'neurons': neurons,
            'dropout': dropout,
            'activation': activation,
            'optimizer': optimizer,
            'learning_rate': learning_rate
        }
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32,
             early_stopping=True, patience=10, verbose=1):
        """Entraîne le modèle"""
        if self.model is None:
            raise ValueError("Le modèle doit être construit")
        
        callback_list = []
        
        if early_stopping:
            es = EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            )
            callback_list.append(es)
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        callback_list.append(reduce_lr)
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=verbose
        )
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """Évalue le modèle"""
        if self.model is None:
            raise ValueError("Le modèle doit être entraîné")
        
        y_pred_raw = self.model.predict(X_test, verbose=0)
        
        metrics = {}
        
        if self.problem_type == 'classification':
            if y_pred_raw.shape[-1] == 1:
                y_pred = (y_pred_raw > 0.5).astype(int).flatten()
            else:
                y_pred = np.argmax(y_pred_raw, axis=1)
            
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            metrics['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics['f1_score'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)
            
            predictions = y_pred
        else:
            y_pred = y_pred_raw.flatten()
            
            metrics['r2_score'] = r2_score(y_test, y_pred)
            metrics['mse'] = mean_squared_error(y_test, y_pred)
            metrics['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
            metrics['mae'] = mean_absolute_error(y_test, y_pred)
            
            try:
                metrics['mape'] = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100
            except:
                metrics['mape'] = None
            
            predictions = y_pred
        
        return metrics, predictions
    
    def get_model_summary(self):
        """Retourne un résumé du modèle"""
        if self.model is None:
            return "Aucun modèle"
        
        summary_list = []
        self.model.summary(print_fn=lambda x: summary_list.append(x))
        return '\n'.join(summary_list)
    
    def save_model(self, path='neural_network_model.h5'):
        """Sauvegarde le modèle"""
        if self.model is None:
            return None
        
        self.model.save(path)
        
        config_path = path.replace('.h5', '_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        return path


# ==================== INTERFACE STREAMLIT ====================

def initialize_session_state():
    """Initialise les variables de session"""
    defaults = {
        'step': 1,
        'data_loaded': False,
        'configured': False,
        'trained': False,
        'df': None,
        'processor': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_header():
    """Affiche le header"""
    st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h1>🤖 Teachable Machine Pro</h1>
            <p style='font-size: 18px; color: #666;'>
                Entraînez vos modèles de Machine Learning et Deep Learning facilement
            </p>
        </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Affiche la sidebar"""
    with st.sidebar:
        st.markdown("### 🎯 Navigation")
        
        steps = [
            ("1️⃣ Upload Data", 1, st.session_state.data_loaded),
            ("2️⃣ Configuration", 2, st.session_state.configured),
            ("3️⃣ Entraînement", 3, st.session_state.get('trained', False)),
            ("4️⃣ Résultats", 4, st.session_state.get('trained', False))
        ]
        
        for label, step_num, completed in steps:
            icon = "✅" if completed else "▶️" if st.session_state.step == step_num else "⭕"
            
            if st.button(f"{icon} {label}", key=f"nav_{step_num}"):
                if step_num <= st.session_state.step or completed:
                    st.session_state.step = step_num
                    st.rerun()
        
        st.markdown("---")
        
        if st.session_state.data_loaded:
            st.markdown("### 📊 Dataset")
            st.metric("Lignes", st.session_state.df.shape[0])
            st.metric("Colonnes", st.session_state.df.shape[1])
        
        if st.session_state.configured:
            st.markdown("### ⚙️ Config")
            st.info(f"**Type**: {st.session_state.problem_type.capitalize()}")
            st.info(f"**Modèle**: {st.session_state.model_type}")


def step1_upload_data():
    """Étape 1: Upload"""
    st.header("📁 Étape 1: Upload vos données")
    
    uploaded_file = st.file_uploader(
        "Choisissez un fichier",
        type=['csv', 'xlsx', 'xls', 'json'],
        help="CSV, Excel ou JSON"
    )
    
    if uploaded_file:
        processor = DataProcessor()
        df, error = processor.load_data(file_path=uploaded_file)
        
        if error:
            st.error(f"❌ {error}")
            return
        
        st.session_state.df = df
        st.session_state.processor = processor
        st.session_state.data_loaded = True
        
        st.success("✅ Données chargées!")
        
        # Métriques
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Lignes", df.shape[0])
        with col2:
            st.metric("📋 Colonnes", df.shape[1])
        with col3:
            st.metric("🔢 Numériques", len(df.select_dtypes(include=[np.number]).columns))
        with col4:
            st.metric("📝 Catégorielles", len(df.select_dtypes(include=['object']).columns))
        
        # Aperçu
        st.subheader("👀 Aperçu")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Stats
        with st.expander("📊 Statistiques"):
            if len(df.select_dtypes(include=[np.number]).columns) > 0:
                st.dataframe(df.describe(), use_container_width=True)
        
        # Valeurs manquantes
        missing = df.isnull().sum()
        if missing.sum() > 0:
            with st.expander("❓ Valeurs manquantes"):
                missing_df = pd.DataFrame({
                    'Colonne': missing.index,
                    'Manquantes': missing.values,
                    'Pourcentage': (missing.values / len(df) * 100).round(2)
                })
                st.dataframe(missing_df[missing_df['Manquantes'] > 0], use_container_width=True)
        
        st.markdown("---")
        if st.button("➡️ Configuration", type="primary"):
            st.session_state.step = 2
            st.rerun()


def step2_configuration():
    """Étape 2: Configuration"""
    if not st.session_state.data_loaded:
        st.warning("⚠️ Chargez d'abord des données!")
        return
    
    st.header("⚙️ Étape 2: Configuration")
    
    df = st.session_state.df
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Type de problème")
        problem_type = st.radio(
            "Type:",
            ["Classification", "Régression"],
            horizontal=True
        )
        st.session_state.problem_type = problem_type.lower()
    
    with col2:
        st.subheader("🧠 Type de modèle")
        model_type = st.radio(
            "Type:",
            ["ML Classique", "Deep Learning"],
            horizontal=True
        )
        st.session_state.model_type = model_type
    
    st.markdown("---")
    
    # Variable cible
    st.subheader("🎯 Variable cible")
    target_col = st.selectbox("Sélectionnez la colonne cible:", df.columns)
    st.session_state.target_col = target_col
    
    col1, col2 = st.columns([3, 1])
    with col2:
        st.metric("Valeurs uniques", df[target_col].nunique())
    
    # Distribution
    st.write("**Distribution:**")
    if df[target_col].nunique() < 20:
        fig, ax = plt.subplots(figsize=(10, 4))
        df[target_col].value_counts().plot(kind='bar', ax=ax, color='#667eea')
        ax.set_title(f'Distribution de {target_col}')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(df[target_col].dropna(), bins=30, color='#667eea', edgecolor='black')
        ax.set_title(f'Distribution de {target_col}')
        st.pyplot(fig)
    
    st.markdown("---")
    
    # Split
    st.subheader("✂️ Split Train/Test")
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        test_size = st.slider("Test set (%)", 10, 40, 20, 5)
        st.session_state.test_size = test_size / 100
    
    with col2:
        st.metric("🎓 Train", f"{100-test_size}%")
    
    with col3:
        st.metric("🧪 Test", f"{test_size}%")
    
    # Options avancées
    with st.expander("🔧 Options avancées"):
        scale_method = st.selectbox("Normalisation:", ["standard", "minmax"])
        st.session_state.scale_method = scale_method
        
        random_state = st.number_input("Random State:", 0, 999, 42)
        st.session_state.random_state = random_state
    
    # Navigation
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("⬅️ Retour", use_container_width=True):
            st.session_state.step = 1
            st.rerun()
    
    with col2:
        if st.button("✅ Valider", type="primary", use_container_width=True):
            st.session_state.configured = True
            st.session_state.step = 3
            st.rerun()


def step3_training():
    """Étape 3: Entraînement"""
    if not st.session_state.configured:
        st.warning("⚠️ Configurez d'abord!")
        return
    
    st.header("🚀 Étape 3: Entraînement")
    
    # Préparation des données
    if 'X_train' not in st.session_state:
        with st.spinner("🔄 Préparation..."):
            try:
                processor = st.session_state.processor
                X_train, X_test, y_train, y_test, feature_names, target_info = processor.prepare_data(
                    st.session_state.df,
                    st.session_state.target_col,
                    test_size=st.session_state.test_size,
                    random_state=st.session_state.get('random_state', 42),
                    scale_method=st.session_state.get('scale_method', 'standard')
                )
                
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.feature_names = feature_names
                st.session_state.target_info = target_info
                
                st.success("✅ Données prêtes!")
                
            except Exception as e:
                st.error(f"❌ Erreur: {str(e)}")
                return
    
    # Afficher infos
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎓 Train", st.session_state.X_train.shape[0])
    with col2:
        st.metric("🧪 Test", st.session_state.X_test.shape[0])
    with col3:
        st.metric("📊 Features", st.session_state.X_train.shape[1])
    with col4:
        if st.session_state.target_info['is_categorical']:
            st.metric("🎯 Classes", st.session_state.target_info['n_classes'])
        else:
            st.metric("🎯 Type", "Numérique")
    
    st.markdown("---")
    
    # ML CLASSIQUE
    if st.session_state.model_type == "ML Classique":
        st.subheader("🤖 Machine Learning Classique")
        
        trainer = ClassicalMLTrainer(st.session_state.problem_type)
        available_models = list(trainer.get_available_models().keys())
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_models = st.multiselect(
                "Algorithmes:",
                available_models,
                default=available_models[:3]
            )
        
        with col2:
            if st.checkbox("Tout"):
                selected_models = available_models
        
        if selected_models:
            st.info(f"📊 {len(selected_models)} modèle(s)")
            
            if st.button("🚀 Entraîner", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status = st.empty()
                
                for idx, model_name in enumerate(selected_models):
                    status.text(f"⏳ {model_name}...")
                    trainer.train_single_model(
                        model_name,
                        st.session_state.X_train,
                        st.session_state.y_train,
                        st.session_state.X_test,
                        st.session_state.y_test
                    )
                    progress_bar.progress((idx + 1) / len(selected_models))
                
                status.text("✅ Terminé!")
                
                st.session_state.ml_trainer = trainer
                st.session_state.ml_results = trainer.results
                st.session_state.trained = True
                
                st.success("🎉 Entraînement réussi!")
                st.balloons()
                
                st.session_state.step = 4
                st.rerun()
        else:
            st.warning("⚠️ Sélectionnez au moins un algorithme")
    
    # DEEP LEARNING
    else:
        st.subheader("🧠 Deep Learning")
        
        col1, col2 = st.columns(2)
        
        with col1:
            config_type = st.radio("Config:", ["Par défaut", "Personnalisée"])
        
        if config_type == "Personnalisée":
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                n_layers = st.number_input("Couches", 2, 10, 3)
                neurons = st.number_input("Neurons", 32, 512, 128, 32)
            
            with col2:
                dropout = st.slider("Dropout", 0.0, 0.5, 0.3, 0.05)
                activation = st.selectbox("Activation", ["relu", "tanh", "sigmoid"])
            
            with col3:
                optimizer = st.selectbox("Optimiseur", ["adam", "sgd", "rmsprop"])
                learning_rate = st.select_slider("LR", [0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                epochs = st.slider("Epochs", 10, 200, 100, 10)
            with col2:
                batch_size = st.selectbox("Batch", [16, 32, 64, 128], 1)
            with col3:
                patience = st.number_input("Patience", 5, 20, 10)
        else:
            n_layers, neurons, dropout = 3, 128, 0.3
            activation, optimizer, learning_rate = 'relu', 'adam', 0.001
            epochs, batch_size, patience = 100, 32, 10
            
            st.info("Config par défaut: 3 couches, 128 neurons, dropout 0.3")
        
        st.markdown("---")
        
        if st.button("🚀 Entraîner DL", type="primary", use_container_width=True):
            with st.spinner("🔄 Entraînement..."):
                try:
                    dl_trainer = NeuralNetworkTrainer(st.session_state.problem_type)
                    
                    input_dim = st.session_state.X_train.shape[1]
                    output_dim = st.session_state.target_info['n_classes'] if st.session_state.problem_type == 'classification' else 1
                    
                    status = st.empty()
                    status.text("🏗️ Construction...")
                    
                    model = dl_trainer.build_simple_nn(
                        input_dim=input_dim,
                        output_dim=output_dim,
                        n_layers=n_layers,
                        neurons=neurons,
                        dropout=dropout,
                        activation=activation,
                        optimizer=optimizer,
                        learning_rate=learning_rate
                    )
                    
                    status.text("🏃 Entraînement...")
                    
                    progress_bar = st.progress(0)
                    epoch_text = st.empty()
                    
                    # Callback pour progression
                    class ProgressCallback(tf.keras.callbacks.Callback):
                        def on_epoch_end(self, epoch, logs=None):
                            progress = (epoch + 1) / epochs
                            progress_bar.progress(progress)
                            epoch_text.text(f"Epoch {epoch+1}/{epochs} - Loss: {logs['loss']:.4f}")
                    
                    history = dl_trainer.train(
                        st.session_state.X_train,
                        st.session_state.y_train,
                        st.session_state.X_test,
                        st.session_state.y_test,
                        epochs=epochs,
                        batch_size=batch_size,
                        patience=patience,
                        verbose=0
                    )
                    
                    progress_bar.progress(1.0)
                    status.text("✅ Terminé!")
                    
                    st.session_state.dl_trainer = dl_trainer
                    st.session_state.dl_history = history
                    st.session_state.trained = True
                    
                    st.success("🎉 Modèle entraîné!")
                    st.balloons()
                    
                    st.session_state.step = 4
                    st.rerun()
                
                except Exception as e:
                    st.error(f"❌ Erreur: {str(e)}")
    
    st.markdown("---")
    if st.button("⬅️ Retour"):
        st.session_state.step = 2
        st.rerun()


def step4_results():
    """Étape 4: Résultats"""
    if not st.session_state.trained:
        st.warning("⚠️ Entraînez d'abord!")
        return
    
    st.header("📊 Étape 4: Résultats")
    
    # ML CLASSIQUE
    if st.session_state.model_type == "ML Classique":
        trainer = st.session_state.ml_trainer
        results = st.session_state.ml_results
        
        st.subheader("📈 Comparaison")
        comparison_df = trainer.get_comparison_dataframe()
        st.dataframe(
            comparison_df.style.highlight_max(axis=0, color='lightgreen'),
            use_container_width=True
        )
        
        # Meilleur modèle
        best_name = trainer.best_model_name
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🏆 Meilleur", best_name)
        with col2:
            if st.session_state.problem_type == 'classification':
                score = comparison_df.loc[best_name, 'accuracy']
                st.metric("🎯 Accuracy", f"{score:.4f}")
            else:
                score = comparison_df.loc[best_name, 'r2_score']
                st.metric("🎯 R²", f"{score:.4f}")
        with col3:
            st.metric("📁 Modèles", len(results))
        
        st.markdown("---")
        
        # Visualisations
        tabs = st.tabs(["📊 Performance", "🎯 Détails", "⭐ Importance"])
        
        with tabs[0]:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            if st.session_state.problem_type == 'classification':
                metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            else:
                metrics = ['r2_score', 'rmse', 'mae']
            
            comparison_df[metrics].plot(kind='bar', ax=ax, rot=45)
            ax.set_title('Comparaison des performances')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        
        with tabs[1]:
            model_select = st.selectbox("Modèle:", list(results.keys()))
            
            if st.session_state.problem_type == 'classification':
                cm = results[model_select]['metrics']['confusion_matrix']
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title(f'Matrice - {model_select}')
                ax.set_xlabel('Prédictions')
                ax.set_ylabel('Vraies valeurs')
                st.pyplot(fig)
            else:
                y_pred = results[model_select]['predictions']
                y_test = st.session_state.y_test
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                
                ax1.scatter(y_test, y_pred, alpha=0.5)
                ax1.plot([y_test.min(), y_test.max()], 
                        [y_test.min(), y_test.max()], 'r--', lw=2)
                ax1.set_xlabel('Réel')
                ax1.set_ylabel('Prédit')
                ax1.set_title('Prédictions vs Réelles')
                ax1.grid(True, alpha=0.3)
                
                residuals = y_test - y_pred
                ax2.scatter(y_pred, residuals, alpha=0.5)
                ax2.axhline(y=0, color='r', linestyle='--')
                ax2.set_xlabel('Prédictions')
                ax2.set_ylabel('Résidus')
                ax2.set_title('Résidus')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
        
        with tabs[2]:
            fi_model = st.selectbox("Modèle:", list(results.keys()), key='fi')
            fi_df = trainer.get_feature_importance(fi_model, st.session_state.feature_names)
            
            if fi_df is not None:
                top_n = min(15, len(fi_df))
                fi_top = fi_df.head(top_n)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(range(len(fi_top)), fi_top['importance'], color='#667eea')
                ax.set_yticks(range(len(fi_top)))
                ax.set_yticklabels(fi_top['feature'])
                ax.set_xlabel('Importance')
                ax.set_title(f'Top {top_n} Features')
                ax.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig)
                
                st.dataframe(fi_df, use_container_width=True)
            else:
                st.info("Non disponible")
        
        st.markdown("---")
        
        # Téléchargement
        st.subheader("💾 Télécharger")
        dl_model = st.selectbox("Modèle:", list(results.keys()))
        
        if st.button("📥 Télécharger", use_container_width=True):
            path = trainer.save_model(dl_model)
            with open(path, 'rb') as f:
                st.download_button(
                    "💾 Télécharger fichier",
                    f,
                    file_name=f"{dl_model.replace(' ', '_')}_model.pkl",
                    mime="application/octet-stream",
                    use_container_width=True
                )
    
    # DEEP LEARNING
    else:
        dl_trainer = st.session_state.dl_trainer
        history = st.session_state.dl_history
        
        metrics, predictions = dl_trainer.evaluate(
            st.session_state.X_test,
            st.session_state.y_test
        )
        
        st.subheader("🎯 Métriques")
        cols = st.columns(4)
        idx = 0
        for name, value in metrics.items():
            if name != 'confusion_matrix':
                with cols[idx % 4]:
                    st.metric(name.replace('_', ' ').title(), f"{value:.4f}")
                idx += 1
        
        st.markdown("---")
        st.subheader("📈 Courbes")
        
        history_data = history.history
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        ax1.plot(history_data['loss'], label='Train', linewidth=2)
        ax1.plot(history_data['val_loss'], label='Val', linewidth=2)
        ax1.set_title('Loss')
        ax1.set_xlabel('Epoch')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Métrique
        metric_key = 'accuracy' if st.session_state.problem_type == 'classification' else 'mae'
        if metric_key in history_data:
            ax2.plot(history_data[metric_key], label='Train', linewidth=2)
            ax2.plot(history_data[f'val_{metric_key}'], label='Val', linewidth=2)
            ax2.set_title(metric_key.upper())
            ax2.set_xlabel('Epoch')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("---")
        
        # Visualisations spécifiques
        if st.session_state.problem_type == 'classification' and 'confusion_matrix' in metrics:
            st.subheader("🎯 Matrice de Confusion")
            cm = metrics['confusion_matrix']
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', ax=ax)
            ax.set_title('Matrice de Confusion')
            st.pyplot(fig)
        
        elif st.session_state.problem_type == 'regression':
            st.subheader("📊 Prédictions")
            y_test = st.session_state.y_test
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            ax1.scatter(y_test, predictions, alpha=0.5)
            ax1.plot([y_test.min(), y_test.max()],
                    [y_test.min(), y_test.max()], 'r--', lw=2)
            ax1.set_xlabel('Réel')
            ax1.set_ylabel('Prédit')
            ax1.set_title('Prédictions vs Réelles')
            ax1.grid(True, alpha=0.3)
            
            residuals = y_test - predictions
            ax2.scatter(predictions, residuals, alpha=0.5)
            ax2.axhline(y=0, color='r', linestyle='--')
            ax2.set_xlabel('Prédictions')
            ax2.set_ylabel('Résidus')
            ax2.set_title('Résidus')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        st.markdown("---")
        
        with st.expander("🏗️ Architecture"):
            st.text(dl_trainer.get_model_summary())
        
        with st.expander("⚙️ Configuration"):
            st.json(dl_trainer.config)
        
        st.markdown("---")
        
        st.subheader("💾 Télécharger")
        if st.button("📥 Sauvegarder", use_container_width=True):
            path = dl_trainer.save_model('model_dl.h5')
            with open(path, 'rb') as f:
                st.download_button(
                    "💾 Télécharger (.h5)",
                    f,
                    file_name="neural_network_model.h5",
                    mime="application/octet-stream",
                    use_container_width=True
                )
    
    # Actions
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("⬅️ Retour", use_container_width=True):
            st.session_state.step = 3
            st.rerun()
    
    with col2:
        if st.button("🔄 Ré-entraîner", use_container_width=True):
            st.session_state.trained = False
            if 'X_train' in st.session_state:
                del st.session_state['X_train']
            st.session_state.step = 3
            st.rerun()
    
    with col3:
        if st.button("🆕 Nouveau", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


def main():
    """Fonction principale"""
    initialize_session_state()
    render_header()
    render_sidebar()
    
    if st.session_state.step == 1:
        step1_upload_data()
    elif st.session_state.step == 2:
        step2_configuration()
    elif st.session_state.step == 3:
        step3_training()
    elif st.session_state.step == 4:
        step4_results()
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p>🤖 <strong>Teachable Machine Pro</strong> | Made with ❤️</p>
            <p style='font-size: 12px;'>scikit-learn & TensorFlow/Keras</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
