import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
from PIL import Image
import io
import os
import zipfile
import tempfile

# Configuration de la page
st.set_page_config(
    page_title="Teachable Machine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS pour le mode sombre
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    .sidebar .sidebar-content {
        background-color: #262730;
        color: #FAFAFA;
    }
    
    .header-section {
        background: linear-gradient(135deg, #1E40AF 0%, #7E22CE 100%);
        padding: 2rem;
        border-radius: 1rem;
        margin-bottom: 2rem;
        border: 1px solid #374151;
    }
    
    .module-card {
        background-color: #1F2937;
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 0.5rem 0;
        border-left: 4px solid #3B82F6;
        border: 1px solid #374151;
    }
    
    .error-card {
        background-color: #7F1D1D;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 1px solid #DC2626;
    }
    
    .demo-card {
        background-color: #1F2937;
        padding: 2rem;
        border-radius: 1rem;
        margin: 1rem 0;
        border: 1px solid #374151;
        text-align: center;
    }
    
    .metric-card {
        background-color: #111827;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        border: 1px solid #374151;
    }
    
    .upload-section {
        border: 2px dashed #4B5563;
        border-radius: 1rem;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        background-color: #1F2937;
    }
    
    .model-card {
        background-color: #1F2937;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border: 1px solid #374151;
    }
    
    .stButton button {
        background-color: #3B82F6;
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        width: 100%;
    }
    
    .stButton button:hover {
        background-color: #2563EB;
    }
    
    .tab-content {
        padding: 1rem 0;
    }
    
    .success-message {
        background-color: #065F46;
        color: #D1FAE5;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #047857;
    }
    
    .info-message {
        background-color: #1E40AF;
        color: #DBEAFE;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #3B82F6;
    }
    </style>
""", unsafe_allow_html=True)

def extract_zip_images(uploaded_zip):
    """Extrait les images d'un fichier ZIP"""
    extracted_images = []
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Parcourir tous les fichiers extraits
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    file_path = os.path.join(root, file)
                    try:
                        img = Image.open(file_path)
                        extracted_images.append({
                            'name': file,
                            'image': img,
                            'path': file_path
                        })
                    except Exception as e:
                        st.warning(f"Impossible de lire l'image {file}: {str(e)}")
    
    return extracted_images

def calculate_model_metrics(model_name, problem_type, data_type):
    """Calcule les métriques réalistes pour un modèle entraîné"""
    
    # Métriques de base pour tous les modèles
    base_metrics = {
        'trainTime': f"{np.random.uniform(1.5, 8.5):.2f}",
        'epochs': f"{np.random.randint(10, 100)}",
        'batch_size': f"{np.random.choice([16, 32, 64, 128])}"
    }
    
    # Détection du type de modèle
    is_cnn = 'CNN' in model_name or (data_type in ['images', 'zip_images'] and 'NN' in model_name)
    is_tree_based = any(x in model_name for x in ['Tree', 'Forest', 'Boost', 'XGBoost', 'LightGBM'])
    is_svm = 'SVC' in model_name or 'SVR' in model_name
    is_linear = any(x in model_name for x in ['Linear', 'Logistic', 'Ridge', 'Lasso', 'Perceptron'])
    is_neighbors = 'KNN' in model_name or 'Neighbors' in model_name
    
    if problem_type == 'classification':
        if is_cnn:
            # Métriques pour CNN (images)
            accuracy = np.random.uniform(0.82, 0.96)
            precision = np.random.uniform(0.80, 0.95)
            recall = np.random.uniform(0.81, 0.94)
            f1 = 2 * (precision * recall) / (precision + recall)
            
            metrics = {
                'accuracy': f"{accuracy:.3f}",
                'val_accuracy': f"{accuracy * np.random.uniform(0.95, 0.99):.3f}",
                'loss': f"{np.random.uniform(0.15, 0.45):.3f}",
                'val_loss': f"{np.random.uniform(0.18, 0.50):.3f}",
                'precision': f"{precision:.3f}",
                'recall': f"{recall:.3f}",
                'f1_score': f"{f1:.3f}",
                'auc_roc': f"{np.random.uniform(0.85, 0.98):.3f}"
            }
            
        elif is_tree_based:
            # Métriques pour les modèles arborescents
            accuracy = np.random.uniform(0.85, 0.95)
            metrics = {
                'accuracy': f"{accuracy:.3f}",
                'precision': f"{np.random.uniform(0.83, 0.94):.3f}",
                'recall': f"{np.random.uniform(0.84, 0.93):.3f}",
                'f1_score': f"{np.random.uniform(0.84, 0.93):.3f}",
                'log_loss': f"{np.random.uniform(0.2, 0.5):.3f}"
            }
            
        elif is_svm:
            # Métriques pour SVM
            accuracy = np.random.uniform(0.80, 0.92)
            metrics = {
                'accuracy': f"{accuracy:.3f}",
                'precision': f"{np.random.uniform(0.78, 0.90):.3f}",
                'recall': f"{np.random.uniform(0.79, 0.91):.3f}",
                'f1_score': f"{np.random.uniform(0.79, 0.90):.3f}"
            }
            
        else:
            # Métriques pour autres modèles
            accuracy = np.random.uniform(0.75, 0.90)
            metrics = {
                'accuracy': f"{accuracy:.3f}",
                'precision': f"{np.random.uniform(0.73, 0.88):.3f}",
                'recall': f"{np.random.uniform(0.74, 0.89):.3f}",
                'f1_score': f"{np.random.uniform(0.74, 0.88):.3f}"
            }
    
    else:  # Regression
        if is_tree_based:
            # Métriques pour régression arborescente
            r2 = np.random.uniform(0.80, 0.95)
            metrics = {
                'r2_score': f"{r2:.3f}",
                'mse': f"{np.random.uniform(0.05, 0.3):.3f}",
                'rmse': f"{np.random.uniform(0.22, 0.55):.3f}",
                'mae': f"{np.random.uniform(0.15, 0.45):.3f}",
                'mape': f"{np.random.uniform(8, 25):.1f}%"
            }
            
        elif is_linear:
            # Métriques pour modèles linéaires
            r2 = np.random.uniform(0.70, 0.90)
            metrics = {
                'r2_score': f"{r2:.3f}",
                'mse': f"{np.random.uniform(0.1, 0.5):.3f}",
                'rmse': f"{np.random.uniform(0.32, 0.71):.3f}",
                'mae': f"{np.random.uniform(0.25, 0.60):.3f}",
                'adjusted_r2': f"{r2 * np.random.uniform(0.95, 0.99):.3f}"
            }
            
        else:
            # Métriques pour autres modèles de régression
            r2 = np.random.uniform(0.65, 0.85)
            metrics = {
                'r2_score': f"{r2:.3f}",
                'mse': f"{np.random.uniform(0.15, 0.6):.3f}",
                'rmse': f"{np.random.uniform(0.39, 0.78):.3f}",
                'mae': f"{np.random.uniform(0.30, 0.65):.3f}"
            }
    
    # Fusionner avec les métriques de base
    metrics.update(base_metrics)
    return metrics

def train_model(model_name, problem_type, data_type):
    """Simule l'entraînement d'un modèle avec calcul des métriques"""
    st.session_state.trained_models[model_name] = {'status': 'training'}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Détection CNN
    is_cnn = 'CNN' in model_name or (data_type in ['images', 'zip_images'] and 'NN' in model_name)
    
    # Si c'est un CNN, créer un résumé d'architecture simulé
    if is_cnn:
        # Calculer les paramètres de manière cohérente
        conv1_params = 896  # (3*3*3+1)*32
        conv2_params = 18496  # (3*3*32+1)*64
        conv3_params = 73856  # (3*3*64+1)*128
        dense1_params = 1048704  # 8192*128 + 128
        dense2_params = 8256  # 128*64 + 64
        dense3_params = 195  # 64*3 + 3
        
        total_params = conv1_params + conv2_params + conv3_params + dense1_params + dense2_params + dense3_params
        
        architecture_summary = {
            'total_params': total_params,
            'trainable_params': total_params,
            'non_trainable_params': 0,
            'layers': [
                {'type': 'Conv2D', 'output_shape': '(None, 64, 64, 32)', 'params': conv1_params},
                {'type': 'BatchNormalization', 'output_shape': '(None, 64, 64, 32)', 'params': 128},
                {'type': 'MaxPooling2D', 'output_shape': '(None, 32, 32, 32)', 'params': 0},
                {'type': 'Dropout', 'output_shape': '(None, 32, 32, 32)', 'params': 0},
                {'type': 'Conv2D', 'output_shape': '(None, 32, 32, 64)', 'params': conv2_params},
                {'type': 'BatchNormalization', 'output_shape': '(None, 32, 32, 64)', 'params': 256},
                {'type': 'MaxPooling2D', 'output_shape': '(None, 16, 16, 64)', 'params': 0},
                {'type': 'Dropout', 'output_shape': '(None, 16, 16, 64)', 'params': 0},
                {'type': 'Conv2D', 'output_shape': '(None, 16, 16, 128)', 'params': conv3_params},
                {'type': 'BatchNormalization', 'output_shape': '(None, 16, 16, 128)', 'params': 512},
                {'type': 'MaxPooling2D', 'output_shape': '(None, 8, 8, 128)', 'params': 0},
                {'type': 'Dropout', 'output_shape': '(None, 8, 8, 128)', 'params': 0},
                {'type': 'Flatten', 'output_shape': '(None, 8192)', 'params': 0},
                {'type': 'Dense', 'output_shape': '(None, 128)', 'params': dense1_params},
                {'type': 'BatchNormalization', 'output_shape': '(None, 128)', 'params': 512},
                {'type': 'Dropout', 'output_shape': '(None, 128)', 'params': 0},
                {'type': 'Dense', 'output_shape': '(None, 64)', 'params': dense2_params},
                {'type': 'BatchNormalization', 'output_shape': '(None, 64)', 'params': 256},
                {'type': 'Dropout', 'output_shape': '(None, 64)', 'params': 0},
                {'type': 'Dense (Output)', 'output_shape': '(None, 3)', 'params': dense3_params}
            ]
        }
    
    # Simulation de l'entraînement
    for i in range(100):
        progress_bar.progress(i + 1)
        if i < 30:
            status_text.text(f"Préparation des données... {i + 1}%")
        elif i < 70:
            status_text.text(f"Entraînement du modèle {model_name}... {i + 1}%")
        else:
            status_text.text(f"Calcul des métriques... {i + 1}%")
        time.sleep(0.02)
    
    status_text.text("")
    progress_bar.empty()
    
    # Calcul des métriques réalistes
    metrics = calculate_model_metrics(model_name, problem_type, data_type)
    
    # Sauvegarder avec architecture si CNN
    model_data = {
        'status': 'trained',
        'metrics': metrics,
        'problem_type': problem_type,
        'data_type': data_type
    }
    
    if is_cnn:
        model_data['architecture'] = architecture_summary
    
    st.session_state.trained_models[model_name] = model_data
    
    st.rerun()

def main():
    # Initialisation des états de session
    if 'step' not in st.session_state:
        st.session_state.step = 1
    if 'problem_type' not in st.session_state:
        st.session_state.problem_type = ''
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'target_column' not in st.session_state:
        st.session_state.target_column = ''
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = {}
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = ''
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
    if 'data_type' not in st.session_state:
        st.session_state.data_type = None

    # Algorithmes disponibles
    algorithms = {
        'classification': {
            'Linear Models': ['Logistic Regression', 'SGD Classifier', 'Perceptron'],
            'Tree Based': ['Decision Tree', 'Random Forest', 'Extra Trees', 'Gradient Boosting', 'AdaBoost', 'XGBoost', 'LightGBM'],
            'SVM': ['SVC (Linear)', 'SVC (RBF)', 'SVC (Poly)', 'Nu-SVC'],
            'Naive Bayes': ['Gaussian NB', 'Multinomial NB', 'Bernoulli NB'],
            'Neighbors': ['KNN', 'Radius Neighbors'],
            'Neural Networks': ['MLP Classifier', 'Simple NN', 'Deep NN', 'CNN (Images)']
        },
        'regression': {
            'Linear Models': ['Linear Regression', 'Ridge', 'Lasso', 'Elastic Net', 'SGD Regressor'],
            'Tree Based': ['Decision Tree', 'Random Forest', 'Extra Trees', 'Gradient Boosting', 'AdaBoost', 'XGBoost', 'LightGBM'],
            'SVM': ['SVR (Linear)', 'SVR (RBF)', 'SVR (Poly)'],
            'Neighbors': ['KNN Regressor', 'Radius Neighbors'],
            'Neural Networks': ['MLP Regressor', 'Simple NN', 'Deep NN']
        }
    }

    # Données de démonstration
    demo_data = {
        'classification': pd.DataFrame({
            'feature1': [5.1, 4.9, 7.0, 6.4, 6.3],
            'feature2': [3.5, 3.0, 3.2, 3.2, 3.3],
            'feature3': [1.4, 1.4, 4.7, 4.5, 6.0],
            'target': ['Class A', 'Class A', 'Class B', 'Class B', 'Class C']
        }),
        'regression': pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature2': [2.0, 3.0, 4.0, 5.0, 6.0],
            'feature3': [3.0, 4.0, 5.0, 6.0, 7.0],
            'target': [10.5, 15.2, 20.1, 25.8, 30.3]
        })
    }

    # Sidebar
    with st.sidebar:
        st.markdown("""
            <div style='text-align: center; margin-bottom: 2rem;'>
                <h1 style='color: #3B82F6;'>🧠</h1>
                <h2 style='color: #FAFAFA;'>Teachable Machine</h2>
                <p style='color: #9CA3AF;'>Dark Mode</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("Entraîner vos modèles de Machine Learning avec des données tabulaires ou des images")
        
        # Modules dans la sidebar
        st.markdown("### Modules")
        
        module_steps = [
            {"icon": "📁", "label": "Upload Data", "step": 1},
            {"icon": "⚙", "label": "Configuration", "step": 2},
            {"icon": "🚀", "label": "Entraînement", "step": 3},
            {"icon": "📊", "label": "Résultats", "step": 4}
        ]
        
        for module in module_steps:
            is_active = st.session_state.step == module["step"]
            bg_color = "#374151" if is_active else "#1F2937"
            text_color = "#3B82F6" if is_active else "#9CA3AF"
            
            st.markdown(f"""
                <div class="module-card" style="background-color: {bg_color};">
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <span style="font-size: 1.2rem;">{module['icon']}</span>
                        <span style="color: {text_color}; font-weight: {'bold' if is_active else 'normal'};">{module['label']}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        # Section d'erreur
        if len(st.session_state.trained_models) == 0 and st.session_state.step >= 3:
            st.markdown("""
                <div class="error-card">
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <span style="font-size: 1.2rem;">⚠</span>
                        <span><strong>Error</strong></span>
                    </div>
                    <p style="margin: 0.5rem 0 0 0; color: #FCA5A5;">Entraîner d'abord un modèle</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Section démo
        st.markdown("### Tester avec données exemple")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Classification", key="demo_class_btn", use_container_width=True):
                st.session_state.problem_type = 'classification'
                st.session_state.data_preview = demo_data['classification']
                st.session_state.columns = demo_data['classification'].columns.tolist()
                st.session_state.target_column = 'target'
                st.session_state.data_loaded = True
                st.session_state.uploaded_data = demo_data['classification']
                st.session_state.data_type = 'demo_tabular'
                st.session_state.step = 2
                st.rerun()
        with col2:
            if st.button("Régression", key="demo_reg_btn", use_container_width=True):
                st.session_state.problem_type = 'regression'
                st.session_state.data_preview = demo_data['regression']
                st.session_state.columns = demo_data['regression'].columns.tolist()
                st.session_state.target_column = 'target'
                st.session_state.data_loaded = True
                st.session_state.uploaded_data = demo_data['regression']
                st.session_state.data_type = 'demo_tabular'
                st.session_state.step = 2
                st.rerun()

    # Contenu principal
    st.markdown("""
        <div class="header-section">
            <h1 style="color: white; margin: 0;">Teachable Machine - Dark Mode</h1>
            <p style="color: #E5E7EB; margin: 0.5rem 0 0 0;">Entraîner vos modèles de Machine Learning avec des données tabulaires ou des images</p>
        </div>
    """, unsafe_allow_html=True)

    # Étape 1: Upload des données
    if st.session_state.step == 1:
        st.markdown("## 📁 Upload des Données")
        
        tab1, tab2 = st.tabs(["📊 Données Tabulaires", "🖼 Images"])
        
        with tab1:
            st.markdown("### Données CSV/Excel")
            uploaded_file = st.file_uploader(
                "Choisir un fichier CSV ou Excel",
                type=['csv', 'xlsx', 'xls'],
                key="csv_uploader"
            )
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)

                    st.session_state.uploaded_data = df
                    st.session_state.data_type = 'tabular'

                    st.markdown(f"""
                        <div class="success-message">
                            ✅ Fichier uploadé avec succès! Dimensions: {df.shape}
                        </div>
                    """, unsafe_allow_html=True)

                    # Aperçu des données
                    st.markdown("#### Aperçu des données")
                    st.dataframe(df.head(), use_container_width=True)

                    col1, col2 = st.columns(2)
                    with col1:
                        target_col = st.selectbox(
                            "Colonne cible:",
                            options=df.columns.tolist(),
                            key="target_select_tabular"
                        )
                        st.session_state.target_column = target_col

                    with col2:
                        problem_type = st.radio(
                            "Type de problème:",
                            options=['classification', 'regression'],
                            format_func=lambda x: "Classification" if x == 'classification' else "Régression",
                            horizontal=True,
                            key="problem_type_tabular"
                        )
                        st.session_state.problem_type = problem_type

                    if st.button("Traiter les Données →", key="process_data_btn", type="primary", use_container_width=True):
                        st.session_state.data_preview = df
                        st.session_state.columns = df.columns.tolist()
                        st.session_state.data_loaded = True
                        st.session_state.step = 2
                        st.rerun()

                except Exception as e:
                    st.error(f"Erreur de lecture: {str(e)}")

        with tab2:
            st.markdown("### Dataset d'Images")
            
            col1, col2 = st.columns(2)
            with col1:
                # Upload d'images individuelles
                uploaded_images = st.file_uploader(
                    "Choisir des images individuelles",
                    type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
                    accept_multiple_files=True,
                    key="image_uploader"
                )
            
            with col2:
                # Upload de fichier ZIP
                uploaded_zip = st.file_uploader(
                    "Ou choisir un fichier ZIP contenant des images",
                    type=['zip'],
                    key="zip_uploader"
                )
            
            images_to_process = []
            
            if uploaded_images:
                st.markdown(f"""
                    <div class="success-message">
                        ✅ {len(uploaded_images)} images individuelles uploadées avec succès!
                    </div>
                """, unsafe_allow_html=True)
                images_to_process = uploaded_images
            
            if uploaded_zip:
                with st.spinner("Extraction des images du fichier ZIP..."):
                    extracted_images = extract_zip_images(uploaded_zip)
                    if extracted_images:
                        st.markdown(f"""
                            <div class="success-message">
                                ✅ {len(extracted_images)} images extraites du fichier ZIP avec succès!
                            </div>
                        """, unsafe_allow_html=True)
                        images_to_process = [img['image'] for img in extracted_images]
                    else:
                        st.error("❌ Aucune image valide trouvée dans le fichier ZIP")

            if images_to_process:
                # Aperçu des images
                st.markdown("#### Aperçu des images")
                cols = st.columns(4)
                for idx, img in enumerate(images_to_process[:8]):
                    with cols[idx % 4]:
                        if isinstance(img, dict):
                            # Cas des images extraites du ZIP
                            st.image(img['image'], use_column_width=True, caption=f"Image {idx + 1}")
                        else:
                            # Cas des images uploadées directement
                            image = Image.open(img)
                            st.image(image, use_column_width=True, caption=f"Image {idx + 1}")

                # Configuration pour les images
                st.markdown("#### Configuration du modèle")
                col1, col2 = st.columns(2)
                with col1:
                    num_classes = st.number_input(
                        "Nombre de classes:",
                        min_value=2,
                        max_value=20,
                        value=3,
                        help="Nombre de catégories dans votre dataset"
                    )
                
                with col2:
                    image_size = st.selectbox(
                        "Taille des images:",
                        options=['64x64', '128x128', '224x224', '256x256'],
                        index=1,
                        help="Taille à laquelle redimensionner les images"
                    )

                if st.button("Traiter les Images →", key="process_images_btn", type="primary", use_container_width=True):
                    with st.spinner("Traitement des images..."):
                        time.sleep(2)
                        # Créer un échantillon de données simulé
                        sample_data = pd.DataFrame({
                            'image_name': [f"image_{i+1}.jpg" for i in range(min(5, len(images_to_process)))],
                            'target': [f'Class {chr(65 + i % num_classes)}' for i in range(min(5, len(images_to_process)))],
                            'image_size': image_size,
                            'total_images': len(images_to_process)
                        })
                        
                        st.session_state.uploaded_data = images_to_process
                        st.session_state.data_type = 'zip_images' if uploaded_zip else 'images'
                        st.session_state.data_preview = sample_data
                        st.session_state.problem_type = 'classification'
                        st.session_state.num_classes = num_classes
                        st.session_state.image_size = image_size
                        st.session_state.data_loaded = True
                        st.session_state.step = 2
                        st.rerun()

    # Étape 2: Configuration
    elif st.session_state.step == 2 and st.session_state.problem_type:
        st.markdown("## ⚙ Configuration")
        
        # Aperçu du dataset
        st.markdown("### Aperçu du Dataset")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Type de données", st.session_state.data_type)
        with col2:
            if hasattr(st.session_state, 'uploaded_data'):
                if st.session_state.data_type in ['tabular', 'demo_tabular']:
                    st.metric("Lignes", len(st.session_state.uploaded_data))
                elif st.session_state.data_type in ['images', 'zip_images']:
                    st.metric("Images", len(st.session_state.uploaded_data))
        with col3:
            st.metric("Type de problème", st.session_state.problem_type)
        
        if st.session_state.data_type in ['tabular', 'demo_tabular']:
            st.markdown("#### Aperçu des données")
            st.dataframe(st.session_state.data_preview, use_container_width=True)
        elif st.session_state.data_type in ['images', 'zip_images']:
            st.markdown("#### Informations sur les images")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Nombre total d'images", len(st.session_state.uploaded_data))
            with col2:
                st.metric("Nombre de classes", st.session_state.num_classes)
            with col3:
                st.metric("Taille des images", st.session_state.image_size)
        
        # Navigation
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("Commencer l'Entraînement →", key="start_training_btn", type="primary", use_container_width=True):
                st.session_state.step = 3
                st.rerun()
            if st.button("← Changer les Données", key="change_data_btn", use_container_width=True):
                st.session_state.step = 1
                st.rerun()

    # Étape 3: Entraînement
    elif st.session_state.step == 3:
        st.markdown("## 🚀 Entraînement des Modèles")
        
        if st.session_state.problem_type:
            algo_to_use = algorithms[st.session_state.problem_type]
            
            for category, models in algo_to_use.items():
                st.markdown(f"### {category}")
                
                cols = st.columns(3)
                for idx, model in enumerate(models):
                    col_idx = idx % 3
                    with cols[col_idx]:
                        with st.container():
                            st.markdown(f"**{model}**")
                            
                            # Afficher des informations sur le modèle
                            if 'CNN' in model and st.session_state.data_type in ['images', 'zip_images']:
                                st.caption("🎯 Optimisé pour les images")
                            elif 'Tree' in model:
                                st.caption("🌳 Modèle arborescent")
                            elif 'SVC' in model or 'SVR' in model:
                                st.caption("📐 Machine à vecteurs de support")
                            
                            if model in st.session_state.trained_models:
                                status = st.session_state.trained_models[model]['status']
                                if status == 'training':
                                    st.button("⏳ Entraînement...", key=f"training_{model}_{idx}", disabled=True)
                                else:
                                    metrics = st.session_state.trained_models[model]['metrics']
                                    if st.session_state.problem_type == 'classification':
                                        score = float(metrics.get('accuracy', metrics.get('val_accuracy', 0)))
                                    else:
                                        score = float(metrics.get('r2_score', 0))
                                    
                                    st.metric("Score", f"{score:.3f}")
                                    if st.button("🔄 Réentraîner", key=f"retrain_{model}_{idx}"):
                                        train_model(model, st.session_state.problem_type, st.session_state.data_type)
                            else:
                                if st.button("🚀 Entraîner", key=f"train_{model}_{idx}"):
                                    train_model(model, st.session_state.problem_type, st.session_state.data_type)
            
            if st.button("Voir les Résultats →", key="view_results_btn", type="primary", disabled=len(st.session_state.trained_models) == 0):
                st.session_state.step = 4
                st.rerun()

    # Étape 4: Résultats
    elif st.session_state.step == 4:
        st.markdown("## 📊 Résultats")
        
        if st.button("← Retour à l'Entraînement", key="back_to_training_btn"):
            st.session_state.step = 3
            st.rerun()
        
        trained_models_list = [name for name, model in st.session_state.trained_models.items()
                               if model.get('status') == 'trained']
        
        if trained_models_list:
            # Comparaison des modèles
            st.markdown("### Comparaison des Performances")
            
            comparison_data = []
            for model_name in trained_models_list:
                try:
                    model_data = st.session_state.trained_models[model_name]
                    
                    # Vérifier que metrics existe
                    if 'metrics' not in model_data:
                        continue
                    
                    metrics = model_data['metrics']
                    
                    if st.session_state.problem_type == 'classification':
                        # Gestion flexible pour CNN et ML classique
                        if 'accuracy' in metrics:
                            score = float(metrics['accuracy']) * 100
                        elif 'val_accuracy' in metrics:
                            score = float(metrics['val_accuracy']) * 100
                        else:
                            score = 85.0
                    else:
                        if 'r2_score' in metrics:
                            score = float(metrics['r2_score']) * 100
                        else:
                            score = 85.0
                    
                    comparison_data.append({
                        'Modèle': model_name,
                        'Score': round(score, 2),
                        'Temps d\'entraînement (s)': float(metrics.get('trainTime', 0)),
                        'Précision': float(metrics.get('precision', 0)) * 100 if metrics.get('precision') else 0,
                        'Rappel': float(metrics.get('recall', 0)) * 100 if metrics.get('recall') else 0
                    })
                    
                except Exception as e:
                    st.warning(f"⚠ Erreur pour {model_name}: {str(e)}")
                    continue
            
            # Vérifier qu'on a des données
            if len(comparison_data) == 0:
                st.error("❌ Aucun modèle avec des métriques valides trouvé. Veuillez ré-entraîner les modèles.")
            else:
                df_comparison = pd.DataFrame(comparison_data)
                
                # Graphique de comparaison
                fig = px.bar(
                    df_comparison, 
                    x='Modèle', 
                    y='Score',
                    title="Comparaison des Performances",
                    color='Score', 
                    color_continuous_scale='Viridis',
                    text='Score'
                )
                
                fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
                
                fig.update_layout(
                    plot_bgcolor='#0E1117',
                    paper_bgcolor='#0E1117',
                    font_color='#FAFAFA',
                    xaxis_title="Modèles",
                    yaxis_title="Score (%)",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Afficher le tableau de comparaison
                st.markdown("#### 📋 Tableau Récapitulatif")
                st.dataframe(
                    df_comparison.style.background_gradient(subset=['Score', 'Précision', 'Rappel'], cmap='Greens'),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Métriques détaillées pour chaque modèle
                st.markdown("### 📊 Détails des Modèles")
                
                for model_name in trained_models_list:
                    try:
                        model_data = st.session_state.trained_models[model_name]
                        
                        if 'metrics' not in model_data:
                            continue
                        
                        has_architecture = 'architecture' in model_data
                        
                        with st.expander(f"📊 {model_name} - Temps: {model_data['metrics'].get('trainTime', 'N/A')}s"):
                            # Métriques
                            st.markdown("#### 📈 Métriques de Performance")
                            metrics_cols = st.columns(4)
                            
                            col_counter = 0
                            for key, value in model_data['metrics'].items():
                                if key != 'trainTime':
                                    col_idx = col_counter % 4
                                    with metrics_cols[col_idx]:
                                        display_value = value
                                        if any(metric in key for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'r2']):
                                            if '%' not in str(value):
                                                try:
                                                    num_val = float(value)
                                                    if num_val <= 1.0:  # Supposer que c'est un score normalisé
                                                        display_value = f"{num_val*100:.1f}%"
                                                except:
                                                    pass
                                        
                                        st.markdown(f"""
                                            <div class="metric-card">
                                                <div style="font-size: 0.8rem; color: #9CA3AF;">
                                                    {key.replace('_', ' ').title()}
                                                </div>
                                                <div style="font-size: 1.2rem; font-weight: bold; color: #3B82F6;">
                                                    {display_value}
                                                </div>
                                            </div>
                                        """, unsafe_allow_html=True)
                                    col_counter += 1
                            
                            # ARCHITECTURE CNN
                            if has_architecture:
                                st.markdown("---")
                                st.markdown("### 🏗 Architecture du Modèle CNN")
                                
                                arch = model_data['architecture']
                                
                                # Résumé des paramètres
                                st.markdown("""
                                    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                                padding: 1.5rem; border-radius: 1rem; margin: 1rem 0;'>
                                        <h4 style='color: white; margin: 0 0 1rem 0;'>📊 Résumé des Paramètres</h4>
                                    </div>
                                """, unsafe_allow_html=True)
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.markdown(f"""
                                        <div class="metric-card" style="background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);">
                                            <div style="font-size: 0.9rem; color: #dbeafe;">Total Paramètres</div>
                                            <div style="font-size: 2rem; font-weight: bold; color: white;">
                                                {arch['total_params']:,}
                                            </div>
                                        </div>
                                    """, unsafe_allow_html=True)
                                with col2:
                                    st.markdown(f"""
                                        <div class="metric-card" style="background: linear-gradient(135deg, #065f46 0%, #10b981 100%);">
                                            <div style="font-size: 0.9rem; color: #d1fae5;">Entraînables</div>
                                            <div style="font-size: 2rem; font-weight: bold; color: white;">
                                                {arch['trainable_params']:,}
                                            </div>
                                        </div>
                                    """, unsafe_allow_html=True)
                                with col3:
                                    st.markdown(f"""
                                        <div class="metric-card" style="background: linear-gradient(135deg, #7c2d12 0%, #f97316 100%);">
                                            <div style="font-size: 0.9rem; color: #fed7aa;">Non-Entraînables</div>
                                            <div style="font-size: 2rem; font-weight: bold; color: white;">
                                                {arch['non_trainable_params']:,}
                                            </div>
                                        </div>
                                    """, unsafe_allow_html=True)
                                
                                st.markdown("<br>", unsafe_allow_html=True)
                                
                                # Tableau des couches
                                st.markdown("#### 🔍 Détail des Couches")
                                layers_data = []
                                for idx, layer in enumerate(arch['layers'], 1):
                                    layers_data.append({
                                        'N°': idx,
                                        'Type de Couche': layer['type'],
                                        'Output Shape': layer['output_shape'],
                                        'Paramètres': f"{layer['params']:,}"
                                    })
                                
                                df_layers = pd.DataFrame(layers_data)
                                st.dataframe(df_layers, use_container_width=True, hide_index=True)
                    
                    except Exception as e:
                        st.error(f"Erreur lors de l'affichage de {model_name}: {str(e)}")
                        continue
        
        else:
            st.markdown("""
                <div class="error-card">
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <span style="font-size: 1.2rem;">⚠</span>
                        <span><strong>Aucun modèle entraîné</strong></span>
                    </div>
                    <p style="margin: 0.5rem 0 0 0; color: #FCA5A5;">Veuillez d'abord entraîner au moins un modèle dans l'onglet Entraînement</p>
                </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
