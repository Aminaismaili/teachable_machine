"""
🤖 Teachable Machine - Application ML Interactive
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import zipfile
import json
from datetime import datetime
from PIL import Image

# Machine Learning imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_squared_error, mean_absolute_error
)

# Classification Models
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, AdaBoostClassifier
)
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Regression Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor,
    GradientBoostingRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION DE LA PAGE
# =============================================================================

st.set_page_config(
    page_title="🤖 Teachable Machine",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CSS PERSONNALISÉ
# =============================================================================

st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    h1, h2, h3 {
        color: #2d3748;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# INITIALISATION SESSION STATE
# =============================================================================

if 'step' not in st.session_state:
    st.session_state.step = 1
if 'data' not in st.session_state:
    st.session_state.data = None
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = None
if 'data_type' not in st.session_state:
    st.session_state.data_type = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'label_encoder' not in st.session_state:
    st.session_state.label_encoder = None
if 'images' not in st.session_state:
    st.session_state.images = []
if 'labels' not in st.session_state:
    st.session_state.labels = []
if 'classes' not in st.session_state:
    st.session_state.classes = []
if 'img_size' not in st.session_state:
    st.session_state.img_size = 128

# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def reset_app():
    """Réinitialise l'application"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.step = 1

def next_step():
    """Passe à l'étape suivante"""
    st.session_state.step += 1

def prev_step():
    """Retourne à l'étape précédente"""
    if st.session_state.step > 1:
        st.session_state.step -= 1

def detect_problem_type(df, target_column):
    """Détecte automatiquement le type de problème"""
    if df[target_column].dtype in ['object', 'category']:
        return 'classification'
    elif df[target_column].nunique() < 10:
        return 'classification'
    else:
        return 'regression'

def load_data_file(uploaded_file):
    """Charge un fichier de données tabulaires"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("❌ Format de fichier non supporté")
            return None
        return df
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement: {str(e)}")
        return None

def load_images_from_zip(zip_file):
    """Charge des images depuis un fichier ZIP"""
    images = []
    labels = []
    classes = []
    
    try:
        with zipfile.ZipFile(zip_file, 'r') as z:
            for file_path in z.namelist():
                if file_path.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    parts = file_path.split('/')
                    if len(parts) > 1:
                        class_name = parts[-2]
                        if class_name not in classes:
                            classes.append(class_name)
                        
                        with z.open(file_path) as img_file:
                            img = Image.open(img_file).convert('RGB')
                            images.append(np.array(img))
                            labels.append(class_name)
        
        return images, labels, classes
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du ZIP: {str(e)}")
        return None, None, None

def preprocess_images(images, labels, classes, img_size):
    """Préprocesse les images pour l'entraînement"""
    processed_images = []
    for img in images:
        img_resized = np.array(Image.fromarray(img).resize((img_size, img_size)))
        processed_images.append(img_resized)
    
    processed_images = np.array(processed_images)
    processed_images = processed_images.astype('float32') / 255.0
    
    label_to_idx = {label: idx for idx, label in enumerate(classes)}
    y = np.array([label_to_idx[label] for label in labels])
    y_encoded = to_categorical(y, num_classes=len(classes))
    
    return processed_images, y_encoded, label_to_idx

def create_cnn_model(input_shape, num_classes):
    """Crée un modèle CNN"""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_distribution(df, column, problem_type):
    """Crée un graphique de distribution"""
    if problem_type == 'classification':
        fig = px.histogram(df, x=column, color=column,
                          title=f"Distribution de {column}",
                          labels={column: 'Classe'},
                          template='plotly_white')
    else:
        fig = px.histogram(df, x=column,
                          title=f"Distribution de {column}",
                          labels={column: 'Valeur'},
                          template='plotly_white')
    
    fig.update_layout(showlegend=False, height=400)
    return fig

def display_stats(df):
    """Affiche les statistiques descriptives"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 {len(df)}</h3>
            <p>Lignes</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📋 {len(df.columns)}</h3>
            <p>Colonnes</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        st.markdown(f"""
        <div class="metric-card">
            <h3>🔢 {len(numeric_cols)}</h3>
            <p>Numériques</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        st.markdown(f"""
        <div class="metric-card">
            <h3>📝 {len(cat_cols)}</h3>
            <p>Catégorielles</p>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# ÉTAPE 1: UPLOAD DES DONNÉES
# =============================================================================

def step1_upload():
    st.title("📁 Étape 1: Upload des Données")
    st.markdown("---")
    
    data_type = st.radio(
        "🎯 Quel type de données souhaitez-vous utiliser?",
        ["📊 Données Tabulaires (CSV/Excel)", "🖼️ Images"],
        horizontal=True
    )
    
    st.markdown("---")
    
    if "Tabulaires" in data_type:
        st.session_state.data_type = 'tabular'
        
        st.markdown("### 📤 Uploadez votre fichier")
        st.info("💡 Formats supportés: CSV, XLSX, XLS")
        
        uploaded_file = st.file_uploader(
            "Choisissez un fichier",
            type=['csv', 'xlsx', 'xls'],
            help="Le fichier doit contenir des colonnes de features et une colonne cible"
        )
        
        if uploaded_file is not None:
            df = load_data_file(uploaded_file)
            
            if df is not None:
                st.success("✅ Fichier chargé avec succès!")
                st.session_state.data = df
                
                st.markdown("### 👀 Aperçu des données")
                st.dataframe(df.head(10), use_container_width=True)
                
                st.markdown("### 📊 Statistiques descriptives")
                display_stats(df)
                
                st.markdown("### 🔍 Analyse des types de données")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Colonnes numériques:**")
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    for col in numeric_cols:
                        st.write(f"• {col}")
                
                with col2:
                    st.markdown("**Colonnes catégorielles:**")
                    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                    for col in cat_cols:
                        st.write(f"• {col}")
                
                st.markdown("### 🎯 Configuration du problème")
                target_column = st.selectbox(
                    "Sélectionnez la colonne cible (variable à prédire):",
                    df.columns.tolist()
                )
                
                if target_column:
                    problem_type = detect_problem_type(df, target_column)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"🤖 Type détecté: **{problem_type.upper()}**")
                    
                    with col2:
                        problem_type = st.selectbox(
                            "Confirmer le type de problème:",
                            ['classification', 'regression'],
                            index=0 if problem_type == 'classification' else 1
                        )
                    
                    st.session_state.problem_type = problem_type
                    st.session_state.target_column = target_column
                    
                    st.markdown("### 📈 Distribution de la variable cible")
                    fig = plot_distribution(df, target_column, problem_type)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("---")
                    if st.button("➡️ Passer à la configuration", type="primary", use_container_width=True):
                        next_step()
                        st.rerun()
    
    else:  # Images
        st.session_state.data_type = 'images'
        st.session_state.problem_type = 'classification'
        
        st.markdown("### 📤 Uploadez vos images")
        
        upload_method = st.radio(
            "Méthode d'upload:",
            ["📦 Archive ZIP (recommandé)", "📸 Fichiers individuels"],
            horizontal=True
        )
        
        if upload_method == "📦 Archive ZIP (recommandé)":
            st.info("💡 Organisez vos images dans des dossiers par classe: classe1/, classe2/, etc.")
            
            zip_file = st.file_uploader("Uploadez votre archive ZIP", type=['zip'])
            
            if zip_file is not None:
                with st.spinner("🔄 Extraction des images..."):
                    images, labels, classes = load_images_from_zip(zip_file)
                
                if images is not None:
                    st.success(f"✅ {len(images)} images chargées avec {len(classes)} classes!")
                    
                    st.session_state.images = images
                    st.session_state.labels = labels
                    st.session_state.classes = classes
                    
                    st.markdown("### 📋 Résumé du dataset")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Images totales", len(images))
                    with col2:
                        st.metric("Nombre de classes", len(classes))
                    with col3:
                        st.metric("Moyenne par classe", len(images) // len(classes))
                    
                    st.markdown("### 📊 Distribution des classes")
                    class_counts = pd.Series(labels).value_counts()
                    fig = px.bar(x=class_counts.index, y=class_counts.values,
                                labels={'x': 'Classe', 'y': 'Nombre d\'images'},
                                title="Répartition des images par classe")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("### 👀 Aperçu des images")
                    cols = st.columns(5)
                    for i, (img, label) in enumerate(zip(images[:5], labels[:5])):
                        with cols[i]:
                            st.image(img, caption=label, use_column_width=True)
                    
                    st.markdown("### ⚙️ Configuration")
                    img_size = st.select_slider(
                        "Taille de redimensionnement:",
                        options=[64, 128, 224, 256],
                        value=128
                    )
                    st.session_state.img_size = img_size
                    
                    st.markdown("---")
                    if st.button("➡️ Passer à l'entraînement", type="primary", use_container_width=True):
                        next_step()
                        st.rerun()
        
        else:  # Fichiers individuels
            st.warning("⚠️ Cette méthode nécessite d'attribuer manuellement les classes")
            
            uploaded_files = st.file_uploader(
                "Uploadez vos images",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                accept_multiple_files=True
            )
            
            if uploaded_files:
                st.success(f"✅ {len(uploaded_files)} images uploadées")
                
                num_classes = st.number_input("Nombre de classes:", min_value=2, max_value=10, value=2)
                class_names = []
                for i in range(num_classes):
                    class_name = st.text_input(f"Nom de la classe {i+1}:", value=f"Classe_{i+1}")
                    class_names.append(class_name)
                
                st.session_state.classes = class_names
                
                images = []
                labels = []
                
                st.markdown("### 🏷️ Attribution des classes")
                for i, file in enumerate(uploaded_files):
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        img = Image.open(file).convert('RGB')
                        st.image(img, use_column_width=True)
                    
                    with col2:
                        label = st.selectbox(
                            f"Classe pour l'image {i+1}:",
                            class_names,
                            key=f"label_{i}"
                        )
                        images.append(np.array(img))
                        labels.append(label)
                
                st.session_state.images = images
                st.session_state.labels = labels
                
                if st.button("➡️ Continuer", type="primary", use_container_width=True):
                    next_step()
                    st.rerun()

# =============================================================================
# ÉTAPE 2: CONFIGURATION ET PREPROCESSING
# =============================================================================

def step2_config():
    st.title("⚙️ Étape 2: Configuration et Preprocessing")
    st.markdown("---")
    
    if st.session_state.data_type == 'tabular':
        df = st.session_state.data
        target_column = st.session_state.target_column
        problem_type = st.session_state.problem_type
        
        st.markdown("### 🔧 Preprocessing automatique")
        
        with st.expander("ℹ️ Transformations appliquées", expanded=True):
            st.markdown("""
            **Pour les features numériques:**
            - 🔧 StandardScaler (normalisation)
            
            **Pour les features catégorielles:**
            - 🎯 OneHotEncoder (encodage)
            
            **Pour la target:**
            - 📊 LabelEncoder (classification)
            
            **Split des données:**
            - 📈 80% entraînement / 20% test
            - 🎯 Stratification (classification)
            """)
        
        if st.button("🚀 Lancer le preprocessing", type="primary", use_container_width=True):
            with st.spinner("🔄 Preprocessing en cours..."):
                try:
                    X = df.drop(columns=[target_column])
                    y = df[target_column]
                    
                    numeric_features = X.select_dtypes(include=[np.number]).columns
                    categorical_features = X.select_dtypes(include=['object', 'category']).columns
                    
                    if len(categorical_features) > 0:
                        X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)
                    else:
                        X_encoded = X.copy()
                    
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X_encoded)
                    
                    if problem_type == 'classification':
                        le = LabelEncoder()
                        y_encoded = le.fit_transform(y)
                        st.session_state.label_encoder = le
                    else:
                        y_encoded = y.values
                    
                    if problem_type == 'classification':
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
                        )
                    else:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_scaled, y_encoded, test_size=0.2, random_state=42
                        )
                    
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.scaler = scaler
                    st.session_state.feature_names = X_encoded.columns.tolist()
                    
                    st.success("✅ Preprocessing terminé avec succès!")
                    
                    st.markdown("### 📊 Résumé du preprocessing")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Features", X_train.shape[1])
                    with col2:
                        st.metric("Train samples", X_train.shape[0])
                    with col3:
                        st.metric("Test samples", X_test.shape[0])
                    with col4:
                        if problem_type == 'classification':
                            st.metric("Classes", len(np.unique(y_train)))
                        else:
                            st.metric("Target range", f"{y_train.min():.2f} - {y_train.max():.2f}")
                
                except Exception as e:
                    st.error(f"❌ Erreur lors du preprocessing: {str(e)}")
        
        # Bouton de navigation en dehors du bloc de preprocessing
        if st.session_state.X_train is not None:
            st.markdown("---")
            if st.button("➡️ Passer à l'entraînement", type="primary", use_container_width=True, key="btn_next_training"):
                next_step()
                st.rerun()
    
    else:  # Images
        st.markdown("### 📐 Configuration du preprocessing des images")
        
        with st.expander("ℹ️ Transformations appliquées", expanded=True):
            st.markdown(f"""
            **Redimensionnement:**
            - 📐 {st.session_state.img_size}x{st.session_state.img_size} pixels
            
            **Normalisation:**
            - 🌈 Pixels de [0, 255] vers [0, 1]
            
            **Encodage:**
            - 🎯 One-hot encoding des labels
            
            **Split:**
            - 📈 80% entraînement / 20% test
            """)
        
        if st.button("🚀 Lancer le preprocessing", type="primary", use_container_width=True):
            with st.spinner("🔄 Preprocessing des images..."):
                try:
                    images = st.session_state.images
                    labels = st.session_state.labels
                    classes = st.session_state.classes
                    img_size = st.session_state.img_size
                    
                    # Preprocessing (returns 3 values: X, y, label_to_idx)
                    X, y, label_to_idx = preprocess_images(images, labels, classes, img_size)
                    st.session_state.label_to_idx = label_to_idx
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
                    
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    
                    st.success("✅ Preprocessing terminé!")
                    
                    st.markdown("### 📊 Résumé")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Train images", X_train.shape[0])
                    with col2:
                        st.metric("Test images", X_test.shape[0])
                    with col3:
                        st.metric("Taille", f"{img_size}x{img_size}")
                    with col4:
                        st.metric("Classes", len(classes))
                
                except Exception as e:
                    st.error(f"❌ Erreur: {str(e)}")
        
        # Bouton de navigation en dehors du bloc de preprocessing
        if st.session_state.X_train is not None:
            st.markdown("---")
            if st.button("➡️ Passer à l'entraînement", type="primary", use_container_width=True, key="btn_next_training_img"):
                next_step()
                st.rerun()

# =============================================================================
# ÉTAPE 3: ENTRAÎNEMENT DES MODÈLES
# =============================================================================

def step3_training():
    st.title("🚀 Étape 3: Entraînement des Modèles")
    st.markdown("---")
    
    if st.session_state.data_type == 'tabular':
        problem_type = st.session_state.problem_type
        
        st.markdown(f"### 🎯 Problème: **{problem_type.upper()}**")
        
        if problem_type == 'classification':
            models_config = {
                "📊 Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "📊 SGD Classifier": SGDClassifier(random_state=42),
                "🌳 Decision Tree": DecisionTreeClassifier(random_state=42),
                "🌳 Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "🌳 Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42),
                "🌳 Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "🌳 AdaBoost": AdaBoostClassifier(random_state=42),
                "🔍 SVM (Linear)": SVC(kernel='linear', random_state=42),
                "🔍 SVM (RBF)": SVC(kernel='rbf', random_state=42),
                "📈 Gaussian NB": GaussianNB(),
                "📈 KNN": KNeighborsClassifier(),
                "🧠 MLP": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
            }
        else:
            models_config = {
                "📊 Linear Regression": LinearRegression(),
                "📊 Ridge": Ridge(random_state=42),
                "📊 Lasso": Lasso(random_state=42),
                "📊 SGD Regressor": SGDRegressor(random_state=42),
                "🌳 Decision Tree": DecisionTreeRegressor(random_state=42),
                "🌳 Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                "🌳 Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=42),
                "🌳 Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "🔍 SVR (Linear)": SVR(kernel='linear'),
                "🔍 SVR (RBF)": SVR(kernel='rbf'),
                "📈 KNN": KNeighborsRegressor(),
                "🧠 MLP": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
            }
        
        st.markdown("### 🎯 Sélectionnez les modèles à entraîner")
        
        col1, col2, col3 = st.columns(3)
        
        model_names = list(models_config.keys())
        selected_models = []
        
        for i, model_name in enumerate(model_names):
            col = [col1, col2, col3][i % 3]
            with col:
                if st.checkbox(model_name, value=(i < 3), key=f"model_{i}"):
                    selected_models.append(model_name)
        
        st.markdown("---")
        
        if st.button("🚀 Entraîner les modèles sélectionnés", type="primary", use_container_width=True):
            if not selected_models:
                st.warning("⚠️ Veuillez sélectionner au moins un modèle")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                X_train = st.session_state.X_train
                X_test = st.session_state.X_test
                y_train = st.session_state.y_train
                y_test = st.session_state.y_test
                
                results = {}
                models = {}
                
                for i, model_name in enumerate(selected_models):
                    status_text.text(f"⏳ Entraînement: {model_name}...")
                    
                    try:
                        model = models_config[model_name]
                        
                        import time
                        start_time = time.time()
                        model.fit(X_train, y_train)
                        training_time = time.time() - start_time
                        
                        y_pred = model.predict(X_test)
                        
                        if problem_type == 'classification':
                            accuracy = accuracy_score(y_test, y_pred)
                            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                            
                            results[model_name] = {
                                'accuracy': accuracy,
                                'precision': precision,
                                'recall': recall,
                                'f1_score': f1,
                                'training_time': training_time,
                                'predictions': y_pred
                            }
                        else:
                            r2 = r2_score(y_test, y_pred)
                            mse = mean_squared_error(y_test, y_pred)
                            mae = mean_absolute_error(y_test, y_pred)
                            rmse = np.sqrt(mse)
                            
                            results[model_name] = {
                                'r2_score': r2,
                                'mse': mse,
                                'mae': mae,
                                'rmse': rmse,
                                'training_time': training_time,
                                'predictions': y_pred
                            }
                        
                        models[model_name] = model
                        
                    except Exception as e:
                        st.error(f"❌ Erreur avec {model_name}: {str(e)}")
                    
                    progress_bar.progress((i + 1) / len(selected_models))
                
                st.session_state.results = results
                st.session_state.models = models
                
                status_text.text("")
                st.success(f"✅ {len(results)} modèles entraînés avec succès!")
        
        # Bouton de navigation en dehors du bloc d'entraînement
        if st.session_state.results and st.session_state.data_type == 'tabular':
            st.markdown("---")
            if st.button("➡️ Voir les résultats", type="primary", use_container_width=True, key="btn_next_results_tab"):
                next_step()
                st.rerun()
    
    else:  # Images - CNN
        st.markdown("### 🧠 Configuration du CNN")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            epochs = st.slider("Époques:", 5, 100, 20)
        with col2:
            batch_size = st.selectbox("Batch size:", [8, 16, 32, 64, 128], index=2)
        with col3:
            learning_rate = st.select_slider("Learning rate:", 
                                            options=[0.0001, 0.001, 0.01, 0.1],
                                            value=0.001,
                                            format_func=lambda x: f"{x}")
        
        st.markdown("---")
        
        if st.button("🚀 Entraîner le CNN", type="primary", use_container_width=True):
            with st.spinner("🔄 Entraînement en cours..."):
                try:
                    X_train = st.session_state.X_train
                    X_test = st.session_state.X_test
                    y_train = st.session_state.y_train
                    y_test = st.session_state.y_test
                    
                    input_shape = X_train.shape[1:]
                    num_classes = y_train.shape[1]
                    
                    model = create_cnn_model(input_shape, num_classes)
                    model.optimizer.learning_rate = learning_rate
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    history_data = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
                    
                    class ProgressCallback(keras.callbacks.Callback):
                        def on_epoch_end(self, epoch, logs=None):
                            progress = (epoch + 1) / epochs
                            progress_bar.progress(progress)
                            status_text.text(f"Époque {epoch + 1}/{epochs} - Loss: {logs['loss']:.4f} - Acc: {logs['accuracy']:.4f}")
                            
                            history_data['loss'].append(logs['loss'])
                            history_data['accuracy'].append(logs['accuracy'])
                            history_data['val_loss'].append(logs['val_loss'])
                            history_data['val_accuracy'].append(logs['val_accuracy'])
                    
                    history = model.fit(
                        X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(X_test, y_test),
                        callbacks=[ProgressCallback()],
                        verbose=0
                    )
                    
                    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
                    y_pred = model.predict(X_test, verbose=0)
                    
                    st.session_state.models['CNN'] = model
                    st.session_state.results['CNN'] = {
                        'accuracy': test_accuracy,
                        'loss': test_loss,
                        'history': history_data,
                        'predictions': y_pred
                    }
                    
                    status_text.text("")
                    st.success(f"✅ CNN entraîné! Accuracy: {test_accuracy:.4f}")
                    
                    st.markdown("### 📈 Courbes d'apprentissage")
                    
                    fig = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=('Loss', 'Accuracy')
                    )
                    
                    fig.add_trace(
                        go.Scatter(y=history_data['loss'], name='Train Loss', mode='lines'),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(y=history_data['val_loss'], name='Val Loss', mode='lines'),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(y=history_data['accuracy'], name='Train Acc', mode='lines'),
                        row=1, col=2
                    )
                    fig.add_trace(
                        go.Scatter(y=history_data['val_accuracy'], name='Val Acc', mode='lines'),
                        row=1, col=2
                    )
                    
                    fig.update_layout(height=400, showlegend=True)
                    st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"❌ Erreur lors de l'entraînement: {str(e)}")
        
        # Bouton de navigation en dehors du bloc d'entraînement
        if st.session_state.results and st.session_state.data_type == 'images':
            st.markdown("---")
            if st.button("➡️ Voir les résultats détaillés", type="primary", use_container_width=True, key="btn_next_results_img"):
                next_step()
                st.rerun()

# =============================================================================
# ÉTAPE 4: RÉSULTATS ET ANALYSE
# =============================================================================

def step4_results():
    st.title("📊 Étape 4: Résultats et Analyse")
    st.markdown("---")
    
    results = st.session_state.results
    
    if not results:
        st.warning("⚠️ Aucun résultat disponible")
        return
    
    if st.session_state.data_type == 'tabular':
        problem_type = st.session_state.problem_type
        
        st.markdown("### 🏆 Comparaison des modèles")
        
        if problem_type == 'classification':
            comparison_df = pd.DataFrame({
                'Modèle': list(results.keys()),
                'Accuracy': [results[m]['accuracy'] for m in results.keys()],
                'Precision': [results[m]['precision'] for m in results.keys()],
                'Recall': [results[m]['recall'] for m in results.keys()],
                'F1-Score': [results[m]['f1_score'] for m in results.keys()],
                'Temps (s)': [results[m]['training_time'] for m in results.keys()]
            })
            
            best_model = comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Modèle']
            best_score = comparison_df['Accuracy'].max()
            
            st.success(f"🏆 Meilleur modèle: **{best_model}** (Accuracy: {best_score:.4f})")
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=comparison_df['Modèle'],
                y=comparison_df['Accuracy'],
                name='Accuracy',
                marker_color='#667eea'
            ))
            
            fig.add_trace(go.Bar(
                x=comparison_df['Modèle'],
                y=comparison_df['F1-Score'],
                name='F1-Score',
                marker_color='#764ba2'
            ))
            
            fig.update_layout(
                title="Comparaison des performances",
                xaxis_title="Modèle",
                yaxis_title="Score",
                barmode='group',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 📋 Résultats détaillés")
            
            def highlight_best(s):
                if s.name in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
                    is_max = s == s.max()
                    return ['background-color: #d4edda' if v else '' for v in is_max]
                return ['' for _ in s]
            
            styled_df = comparison_df.style.apply(highlight_best)
            st.dataframe(styled_df, use_container_width=True)
        
        else:  # Regression
            comparison_df = pd.DataFrame({
                'Modèle': list(results.keys()),
                'R² Score': [results[m]['r2_score'] for m in results.keys()],
                'MSE': [results[m]['mse'] for m in results.keys()],
                'MAE': [results[m]['mae'] for m in results.keys()],
                'RMSE': [results[m]['rmse'] for m in results.keys()],
                'Temps (s)': [results[m]['training_time'] for m in results.keys()]
            })
            
            best_model = comparison_df.loc[comparison_df['R² Score'].idxmax(), 'Modèle']
            best_score = comparison_df['R² Score'].max()
            
            st.success(f"🏆 Meilleur modèle: **{best_model}** (R²: {best_score:.4f})")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = px.bar(comparison_df, x='Modèle', y='R² Score',
                             title="R² Score par modèle",
                             color='R² Score',
                             color_continuous_scale='Viridis')
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                fig2 = px.bar(comparison_df, x='Modèle', y='RMSE',
                             title="RMSE par modèle",
                             color='RMSE',
                             color_continuous_scale='Reds')
                st.plotly_chart(fig2, use_container_width=True)
            
            st.markdown("### 📋 Résultats détaillés")
            
            def highlight_best_reg(s):
                if s.name == 'R² Score':
                    is_max = s == s.max()
                    return ['background-color: #d4edda' if v else '' for v in is_max]
                elif s.name in ['MSE', 'MAE', 'RMSE']:
                    is_min = s == s.min()
                    return ['background-color: #d4edda' if v else '' for v in is_min]
                return ['' for _ in s]
            
            styled_df = comparison_df.style.apply(highlight_best_reg)
            st.dataframe(styled_df, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 🔍 Analyse détaillée d'un modèle")
        
        selected_model = st.selectbox("Choisissez un modèle:", list(results.keys()))
        
        if selected_model:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Métriques")
                model_results = results[selected_model]
                
                if problem_type == 'classification':
                    st.metric("Accuracy", f"{model_results['accuracy']:.4f}")
                    st.metric("Precision", f"{model_results['precision']:.4f}")
                    st.metric("Recall", f"{model_results['recall']:.4f}")
                    st.metric("F1-Score", f"{model_results['f1_score']:.4f}")
                else:
                    st.metric("R² Score", f"{model_results['r2_score']:.4f}")
                    st.metric("MSE", f"{model_results['mse']:.4f}")
                    st.metric("MAE", f"{model_results['mae']:.4f}")
                    st.metric("RMSE", f"{model_results['rmse']:.4f}")
                
                st.metric("Temps d'entraînement", f"{model_results['training_time']:.2f}s")
            
            with col2:
                st.markdown("#### 🔮 Visualisation des prédictions")
                
                y_test = st.session_state.y_test
                y_pred = model_results['predictions']
                
                if problem_type == 'classification':
                    n_samples = min(10, len(y_test))
                    sample_indices = np.random.choice(len(y_test), n_samples, replace=False)
                    
                    sample_df = pd.DataFrame({
                        'Vrai': y_test[sample_indices],
                        'Prédit': y_pred[sample_indices],
                        'Correct': y_test[sample_indices] == y_pred[sample_indices]
                    })
                    
                    st.dataframe(sample_df, use_container_width=True)
                    
                    local_acc = (sample_df['Correct'].sum() / len(sample_df))
                    st.info(f"Accuracy sur cet échantillon: {local_acc:.2%}")
                
                else:
                    fig = px.scatter(
                        x=y_test[:100],
                        y=y_pred[:100],
                        labels={'x': 'Valeurs réelles', 'y': 'Prédictions'},
                        title="Prédictions vs Valeurs réelles"
                    )
                    
                    min_val = min(y_test.min(), y_pred.min())
                    max_val = max(y_test.max(), y_pred.max())
                    fig.add_trace(go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='Perfection',
                        line=dict(dash='dash', color='red')
                    ))
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    else:  # Images - CNN
        st.markdown("### 🧠 Résultats du CNN")
        
        cnn_results = results['CNN']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Accuracy", f"{cnn_results['accuracy']:.4f}")
        with col2:
            st.metric("Loss", f"{cnn_results['loss']:.4f}")
        with col3:
            final_val_acc = cnn_results['history']['val_accuracy'][-1]
            st.metric("Val Accuracy", f"{final_val_acc:.4f}")
        
        st.markdown("### 📈 Courbes d'apprentissage")
        
        history = cnn_results['history']
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Loss au cours de l\'entraînement', 'Accuracy au cours de l\'entraînement')
        )
        
        fig.add_trace(
            go.Scatter(y=history['loss'], name='Train Loss', mode='lines', line=dict(color='#667eea')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(y=history['val_loss'], name='Val Loss', mode='lines', line=dict(color='#764ba2')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(y=history['accuracy'], name='Train Acc', mode='lines', line=dict(color='#667eea')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(y=history['val_accuracy'], name='Val Acc', mode='lines', line=dict(color='#764ba2')),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 🔮 Exemples de prédictions")
        
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        y_pred = cnn_results['predictions']
        classes = st.session_state.classes
        
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
        
        n_examples = min(5, len(X_test))
        cols = st.columns(n_examples)
        
        for i in range(n_examples):
            with cols[i]:
                st.image(X_test[i], use_column_width=True)
                true_label = classes[y_test_classes[i]]
                pred_label = classes[y_pred_classes[i]]
                
                if true_label == pred_label:
                    st.success(f"✅ {pred_label}")
                else:
                    st.error(f"❌ Prédit: {pred_label}\nVrai: {true_label}")
        
        st.markdown("### 🏗️ Architecture du modèle")
        model = st.session_state.models['CNN']
        
        with st.expander("Voir l'architecture"):
            summary_list = []
            model.summary(print_fn=lambda x: summary_list.append(x))
            summary_text = '\n'.join(summary_list)
            st.text(summary_text)
    
    st.markdown("---")
    st.markdown("### 💾 Export des résultats")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.data_type == 'tabular':
            csv_buffer = io.StringIO()
            comparison_df.to_csv(csv_buffer, index=False)
            
            st.download_button(
                label="📊 Télécharger CSV",
                data=csv_buffer.getvalue(),
                file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        json_data = {
            'timestamp': datetime.now().isoformat(),
            'data_type': st.session_state.data_type,
            'problem_type': st.session_state.problem_type,
            'results': {k: {key: (float(val) if isinstance(val, (np.floating, np.integer)) else val) 
                           for key, val in v.items() if key != 'predictions'} 
                       for k, v in results.items()}
        }
        
        st.download_button(
            label="📄 Télécharger JSON",
            data=json.dumps(json_data, indent=2),
            file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    st.markdown("---")
    st.markdown("### 💡 Recommandations")
    
    with st.expander("🚀 Conseils pour améliorer vos modèles", expanded=True):
        if st.session_state.data_type == 'tabular':
            st.markdown("""
            **Pour les données tabulaires:**
            - 🔧 Essayez l'optimisation des hyperparamètres (GridSearch, RandomSearch)
            - 📊 Analysez l'importance des features
            - 🎯 Vérifiez l'équilibrage des classes (pour classification)
            - 📈 Testez l'engineering de features
            - 🔄 Augmentez la taille du dataset si possible
            """)
        else:
            st.markdown("""
            **Pour les images:**
            - 📐 Essayez différentes tailles d'images
            - 🔄 Utilisez l'augmentation de données (rotation, flip, zoom)
            - 🧠 Testez des architectures plus profondes
            - 📚 Utilisez le transfer learning (VGG, ResNet, etc.)
            - ⏱️ Augmentez le nombre d'époques
            - 📊 Collectez plus de données si possible
            """)
        
        st.markdown("""
        **Prochaines étapes:**
        - 💾 Sauvegardez vos meilleurs modèles
        - 🔍 Analysez les erreurs de prédiction
        - 📊 Créez un rapport détaillé pour les stakeholders
        - 🚀 Déployez votre modèle en production
        """)

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.title("🤖 Teachable Machine")
    st.markdown("---")
    
    st.markdown("### 📍 Progression")
    steps = ["📁 Upload", "⚙️ Config", "🚀 Entraînement", "📊 Résultats"]
    
    for i, step_name in enumerate(steps, 1):
        if i < st.session_state.step:
            st.success(f"✅ {step_name}")
        elif i == st.session_state.step:
            st.info(f"▶️ {step_name}")
        else:
            st.text(f"⭕ {step_name}")
    
    st.markdown("---")
    
    st.markdown("### 🔄 Navigation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("⬅️ Retour", use_container_width=True, disabled=(st.session_state.step == 1)):
            prev_step()
            st.rerun()
    
    with col2:
        if st.button("🔄 Reset", use_container_width=True):
            reset_app()
            st.rerun()
    
    st.markdown("---")
    
    if st.session_state.data is not None or st.session_state.images:
        st.markdown("### ℹ️ Informations")
        
        if st.session_state.data_type == 'tabular':
            st.text(f"Type: Tabulaire")
            st.text(f"Problème: {st.session_state.problem_type or 'N/A'}")
            if st.session_state.data is not None:
                st.text(f"Lignes: {len(st.session_state.data)}")
                st.text(f"Colonnes: {len(st.session_state.data.columns)}")
        else:
            st.text(f"Type: Images")
            st.text(f"Images: {len(st.session_state.images)}")
            st.text(f"Classes: {len(st.session_state.classes)}")
        
        if st.session_state.results:
            st.text(f"Modèles: {len(st.session_state.results)}")
    
    st.markdown("---")
    
    st.markdown("""
    <div style='text-align: center; color: #888; font-size: 12px;'>
        <p>🎓 Projet Ingénieur IA</p>
        <p>Data Science - 5ème année</p>
        <p>v1.0.0</p>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# MAIN CONTENT
# =============================================================================

st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>
            🤖 Teachable Machine
        </h1>
        <p style='color: white; font-size: 18px;'>
            Créez et entraînez des modèles ML sans code
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

with st.container():
    if st.session_state.step == 1:
        step1_upload()
    elif st.session_state.step == 2:
        step2_config()
    elif st.session_state.step == 3:
        step3_training()
    elif st.session_state.step == 4:
        step4_results()
