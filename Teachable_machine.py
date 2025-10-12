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

def train_model(model_name, problem_type, data_type):
    """Simule l'entraînement d'un modèle"""
    st.session_state.trained_models[model_name] = {'status': 'training'}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Détection CNN
    is_cnn = 'CNN' in model_name or (data_type in ['images', 'camera'] and 'NN' in model_name)
    
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
    
    for i in range(100):
        progress_bar.progress(i + 1)
        status_text.text(f"Entraînement {model_name}... {i + 1}%")
        time.sleep(0.02)
    
    status_text.text("")
    progress_bar.empty()
    
    if problem_type == 'classification':
        if is_cnn:
            metrics = {
                'accuracy': f"{0.85 + np.random.random() * 0.15:.3f}",
                'val_accuracy': f"{0.83 + np.random.random() * 0.15:.3f}",
                'loss': f"{0.2 + np.random.random() * 0.3:.3f}",
                'val_loss': f"{0.25 + np.random.random() * 0.3:.3f}",
                'precision': f"{0.80 + np.random.random() * 0.15:.3f}",
                'recall': f"{0.82 + np.random.random() * 0.15:.3f}",
                'trainTime': f"{np.random.random() * 5:.2f}"
            }
        else:
            metrics = {
                'accuracy': f"{0.85 + np.random.random() * 0.15:.3f}",
                'precision': f"{0.80 + np.random.random() * 0.15:.3f}",
                'recall': f"{0.82 + np.random.random() * 0.15:.3f}",
                'f1_score': f"{0.83 + np.random.random() * 0.15:.3f}",
                'trainTime': f"{np.random.random() * 2:.2f}"
            }
    else:
        metrics = {
            'mse': f"{0.1 + np.random.random() * 0.5:.3f}",
            'rmse': f"{0.3 + np.random.random() * 0.3:.3f}",
            'mae': f"{0.2 + np.random.random() * 0.3:.3f}",
            'r2Score': f"{0.85 + np.random.random() * 0.15:.3f}",
            'trainTime': f"{np.random.random() * 2:.2f}"
        }
    
    # Sauvegarder avec architecture si CNN
    model_data = {
        'status': 'trained',
        'metrics': metrics
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
    if 'camera_capture' not in st.session_state:
        st.session_state.camera_capture = None

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
        
        tab1, tab2, tab3 = st.tabs(["📊 Données Tabulaires", "🖼 Images", "📷 Caméra"])
        
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
            uploaded_images = st.file_uploader(
                "Choisir des images",
                type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
                accept_multiple_files=True,
                key="image_uploader"
            )

            if uploaded_images:
                st.markdown(f"""
                    <div class="success-message">
                        ✅ {len(uploaded_images)} images uploadées avec succès!
                    </div>
                """, unsafe_allow_html=True)

                # Aperçu des images
                st.markdown("#### Aperçu des images")
                cols = st.columns(4)
                for idx, img_file in enumerate(uploaded_images[:4]):
                    with cols[idx % 4]:
                        image = Image.open(img_file)
                        st.image(image, use_column_width=True, caption=f"Image {idx + 1}")

                if st.button("Traiter les Images →", key="process_images_btn", type="primary", use_container_width=True):
                    with st.spinner("Traitement des images..."):
                        time.sleep(2)
                        sample_data = pd.DataFrame({
                            'image_name': [img.name for img in uploaded_images[:5]],
                            'target': ['Class A', 'Class B', 'Class A', 'Class C', 'Class B']
                        })
                        st.session_state.uploaded_data = uploaded_images
                        st.session_state.data_type = 'images'
                        st.session_state.data_preview = sample_data
                        st.session_state.problem_type = 'classification'
                        st.session_state.data_loaded = True
                        st.session_state.step = 2
                        st.rerun()

        with tab3:
            st.markdown("### Capture Caméra")
            st.markdown("""
                <div class="info-message">
                    💡 Fonctionnalité caméra - Utilisez des images existantes ou simulez une capture
                </div>
            """, unsafe_allow_html=True)
            
            camera_images = st.file_uploader(
                "Uploader des images de caméra",
                type=['jpg', 'jpeg', 'png'],
                accept_multiple_files=True,
                key="camera_uploader"
            )
            
            if st.button("📸 Simuler Capture Caméra", key="simulate_camera_btn", use_container_width=True):
                simulated_images = []
                for i in range(4):
                    img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                    img = Image.fromarray(img_array)
                    simulated_images.append(img)
                st.session_state.camera_capture = simulated_images
                st.success("✅ 4 images simulées capturées!")
                
                cols = st.columns(4)
                for idx, img in enumerate(simulated_images):
                    with cols[idx]:
                        st.image(img, use_column_width=True, caption=f"Capture {idx + 1}")

            if camera_images or st.session_state.camera_capture:
                images_to_use = camera_images if camera_images else st.session_state.camera_capture
                if st.button("Utiliser Images Capturées →", key="use_camera_btn", type="primary", use_container_width=True):
                    with st.spinner("Traitement des images capturées..."):
                        time.sleep(2)
                        if camera_images:
                            sample_data = pd.DataFrame({
                                'image_name': [img.name for img in camera_images[:5]],
                                'source': ['camera'] * min(5, len(camera_images)),
                                'target': ['Class A', 'Class B', 'Class A', 'Class A', 'Class B']
                            })
                        else:
                            sample_data = pd.DataFrame({
                                'image_name': [f"capture_{i + 1}.jpg" for i in range(4)],
                                'source': ['camera'] * 4,
                                'target': ['Class A', 'Class B', 'Class A', 'Class B']
                            })
                        st.session_state.uploaded_data = images_to_use
                        st.session_state.data_type = 'camera'
                        st.session_state.data_preview = sample_data
                        st.session_state.problem_type = 'classification'
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
                elif st.session_state.data_type in ['images', 'camera']:
                    st.metric("Images", len(st.session_state.uploaded_data))
        with col3:
            st.metric("Type de problème", st.session_state.problem_type)
        
        if st.session_state.data_type in ['tabular', 'demo_tabular']:
            st.markdown("#### Aperçu des données")
            st.dataframe(st.session_state.data_preview, use_container_width=True)
        
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
                            st.markdown(f"{model}")
                            
                            if model in st.session_state.trained_models:
                                status = st.session_state.trained_models[model]['status']
                                if status == 'training':
                                    st.button("⏳ Entraînement...", key=f"training_{model}_{idx}", disabled=True)
                                else:
                                    if st.button("🔄 Réentraîner", key=f"retrain_{model}_{idx}"):
                                        train_model(model, st.session_state.problem_type, st.session_state.data_type)
                            else:
                                if st.button("🚀 Entraîner", key=f"train_{model}_{idx}"):
                                    train_model(model, st.session_state.problem_type, st.session_state.data_type)
            
            if st.button("Voir les Résultats →", key="view_results_btn", type="primary", disabled=len(st.session_state.trained_models) == 0):
                st.session_state.step = 4
                st.rerun()

    # Étape 4: Résultats (SECTION CORRIGÉE)
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
                        if 'r2Score' in metrics:
                            score = float(metrics['r2Score']) * 100
                        else:
                            score = 85.0
                    
                    comparison_data.append({
                        'Modèle': model_name,
                        'Score': round(score, 2),
                        'Temps d\'entraînement': float(metrics.get('trainTime', 0))
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
                    df_comparison.style.background_gradient(subset=['Score'], cmap='Greens'),
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
                                        st.markdown(f"""
                                            <div class="metric-card">
                                                <div style="font-size: 0.8rem; color: #9CA3AF;">
                                                    {key.replace('_', ' ').title()}
                                                </div>
                                                <div style="font-size: 1.5rem; font-weight: bold; color: #3B82F6;">
                                                    {value}
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
                                
                                # Visualisation graphique
                                st.markdown("#### 📊 Visualisation de l'Architecture")
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # Graphique des paramètres par couche
                                    layer_names = [f"{layer['type']}" for layer in arch['layers']]
                                    layer_params = [layer['params'] for layer in arch['layers']]
                                    
                                    non_zero_layers = [(name, params) for name, params in zip(layer_names, layer_params) if params > 0]
                                    if non_zero_layers:
                                        names, params = zip(*non_zero_layers)
                                        
                                        fig1 = go.Figure(data=[
                                            go.Bar(
                                                y=list(names)[::-1],
                                                x=list(params)[::-1],
                                                orientation='h',
                                                marker=dict(
                                                    color=list(params)[::-1],
                                                    colorscale='Viridis',
                                                    showscale=True
                                                ),
                                                text=[f"{p:,}" for p in params[::-1]],
                                                textposition='auto'
                                            )
                                        ])
                                        
                                        fig1.update_layout(
                                            title="Paramètres par Couche",
                                            xaxis_title="Nombre de paramètres",
                                            height=500,
                                            plot_bgcolor='#0E1117',
                                            paper_bgcolor='#0E1117',
                                            font_color='#FAFAFA'
                                        )
                                        
                                        st.plotly_chart(fig1, use_container_width=True)
                                
                                with col2:
                                    # Diagramme circulaire
                                    layer_types = {}
                                    for layer in arch['layers']:
                                        layer_type = layer['type'].split()[0]
                                        if layer['params'] > 0:
                                            layer_types[layer_type] = layer_types.get(layer_type, 0) + layer['params']
                                    
                                    if layer_types:
                                        fig2 = go.Figure(data=[go.Pie(
                                            labels=list(layer_types.keys()),
                                            values=list(layer_types.values()),
                                            hole=0.4,
                                            textinfo='label+percent'
                                        )])
                                        
                                        fig2.update_layout(
                                            title="Distribution des Paramètres",
                                            height=500,
                                            plot_bgcolor='#0E1117',
                                            paper_bgcolor='#0E1117',
                                            font_color='#FAFAFA'
                                        )
                                        
                                        st.plotly_chart(fig2, use_container_width=True)
                    
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

if _name_ == "_main_":
    main()
