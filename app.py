import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import cv2
from PIL import Image
import io
import os

# Configuration de la page
st.set_page_config(
    page_title="Teachable Machine",
    page_icon="🧠",
    layout="wide"
)


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
            'Tree Based': ['Decision Tree', 'Random Forest', 'Extra Trees', 'Gradient Boosting', 'AdaBoost', 'XGBoost',
                           'LightGBM'],
            'SVM': ['SVC (Linear)', 'SVC (RBF)', 'SVC (Poly)', 'Nu-SVC'],
            'Naive Bayes': ['Gaussian NB', 'Multinomial NB', 'Bernoulli NB'],
            'Neighbors': ['KNN', 'Radius Neighbors'],
            'Neural Networks': ['MLP Classifier', 'Simple NN', 'Deep NN', 'CNN (Images)']
        },
        'regression': {
            'Linear Models': ['Linear Regression', 'Ridge', 'Lasso', 'Elastic Net', 'SGD Regressor'],
            'Tree Based': ['Decision Tree', 'Random Forest', 'Extra Trees', 'Gradient Boosting', 'AdaBoost', 'XGBoost',
                           'LightGBM'],
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

    # Header
    st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 1rem;
            color: white;
            margin-bottom: 2rem;
        }
        .step-card {
            background-color: #f8fafc;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
            border-left: 4px solid #6366f1;
        }
        .model-card {
            border: 1px solid #e2e8f0;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .metric-card {
            background-color: #f1f5f9;
            padding: 1rem;
            border-radius: 0.5rem;
            text-align: center;
        }
        .upload-section {
            border: 2px dashed #cbd5e1;
            border-radius: 1rem;
            padding: 2rem;
            text-align: center;
            margin: 1rem 0;
            background-color: #f8fafc;
        }
        .tab-content {
            padding: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header principal
    col1, col2 = st.columns([1, 4])
    with col1:
        st.markdown("<h1 style='text-align: center;'>🧠</h1>", unsafe_allow_html=True)
    with col2:
        st.title("Teachable Machine")
        st.subheader("Train ML models without writing code")

    # Barre de progression
    st.markdown("### Progress")
    progress_cols = st.columns(3)
    steps = [
        {"num": 1, "label": "Upload Data", "icon": "📁"},
        {"num": 2, "label": "Select Algorithm", "icon": "⚙️"},
        {"num": 3, "label": "Train & Evaluate", "icon": "📈"}
    ]

    for idx, step_info in enumerate(steps):
        with progress_cols[idx]:
            is_active = st.session_state.step >= step_info["num"]
            is_completed = st.session_state.step > step_info["num"]

            bg_color = "#e0e7ff" if is_active else "#f1f5f9"
            text_color = "#6366f1" if is_active else "#64748b"
            icon = "✅" if is_completed else step_info["icon"]

            st.markdown(f"""
                <div style="background-color: {bg_color}; color: {text_color}; 
                         padding: 1rem; border-radius: 0.5rem; text-align: center;">
                    <div style="font-size: 1.5rem;">{icon}</div>
                    <div style="font-weight: bold;">{step_info['label']}</div>
                </div>
            """, unsafe_allow_html=True)

    # Étape 1: Upload des données
    if st.session_state.step == 1:
        st.markdown("## Upload Your Dataset")

        # Onglets pour les différents types de données
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📊 CSV/Excel Data", "🖼️ Image Dataset", "📷 Camera Capture", "🎯 Use Demo Data"])

        with tab1:
            st.markdown("### Upload CSV or Excel File")

            uploaded_file = st.file_uploader(
                "Choose a CSV or Excel file",
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

                    st.success(f"✅ File uploaded successfully! Shape: {df.shape}")

                    # Aperçu des données
                    st.markdown("#### Data Preview")
                    st.dataframe(df.head(), use_container_width=True)

                    # Informations sur les données
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Rows", df.shape[0])
                    with col2:
                        st.metric("Columns", df.shape[1])
                    with col3:
                        st.metric("Data Type", "Tabular")

                    # Sélection de la colonne cible
                    st.markdown("#### Select Target Column")
                    target_col = st.selectbox(
                        "Choose the target variable column:",
                        options=df.columns.tolist(),
                        key="target_select"
                    )

                    st.session_state.target_column = target_col

                    # Sélection du type de problème
                    st.markdown("#### Select Problem Type")
                    problem_type = st.radio(
                        "What type of problem are you solving?",
                        options=['classification', 'regression'],
                        format_func=lambda x: "Classification" if x == 'classification' else "Regression",
                        horizontal=True
                    )

                    st.session_state.problem_type = problem_type

                    if st.button("Process Data →", type="primary", use_container_width=True):
                        st.session_state.data_preview = df
                        st.session_state.columns = df.columns.tolist()
                        st.session_state.data_loaded = True
                        st.session_state.step = 2
                        st.rerun()

                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")

        with tab2:
            st.markdown("### Upload Image Dataset")

            uploaded_images = st.file_uploader(
                "Choose image files",
                type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
                accept_multiple_files=True,
                key="image_uploader"
            )

            if uploaded_images:
                st.success(f"✅ {len(uploaded_images)} images uploaded successfully!")

                # Afficher un échantillon d'images
                st.markdown("#### Image Preview")
                cols = st.columns(4)
                for idx, img_file in enumerate(uploaded_images[:8]):
                    with cols[idx % 4]:
                        image = Image.open(img_file)
                        st.image(image, use_column_width=True, caption=f"Image {idx + 1}")

                # Options pour les images
                st.markdown("#### Image Processing Options")
                col1, col2 = st.columns(2)

                with col1:
                    image_size = st.selectbox(
                        "Image Size",
                        options=['64x64', '128x128', '224x224', '256x256', 'Original']
                    )

                with col2:
                    model_type = st.selectbox(
                        "Model Type for Images",
                        options=['CNN (Simple)', 'CNN (Deep)', 'ResNet', 'VGG', 'Custom CNN']
                    )

                # Sélection du type de problème pour les images
                st.markdown("#### Select Problem Type")
                img_problem_type = st.radio(
                    "What type of problem are you solving?",
                    options=['classification', 'regression'],
                    format_func=lambda x: "Classification" if x == 'classification' else "Regression",
                    horizontal=True,
                    key="img_problem"
                )

                if st.button("Process Images →", type="primary", use_container_width=True):
                    # Simulation du traitement d'images
                    with st.spinner("Processing images..."):
                        time.sleep(2)

                        # Créer des données simulées pour l'aperçu
                        sample_data = pd.DataFrame({
                            'image_name': [img.name for img in uploaded_images[:5]],
                            'image_size': [image_size] * 5,
                            'target': ['Class A', 'Class B', 'Class A', 'Class C', 'Class B']
                        })

                        st.session_state.uploaded_data = uploaded_images
                        st.session_state.data_type = 'images'
                        st.session_state.data_preview = sample_data
                        st.session_state.problem_type = img_problem_type
                        st.session_state.image_size = image_size
                        st.session_state.model_type = model_type
                        st.session_state.data_loaded = True
                        st.session_state.step = 2
                        st.rerun()

        with tab3:
            st.markdown("### Capture Images from Camera")

            # Option 1: Upload d'images existantes de la caméra
            st.markdown("#### Upload Camera Images")
            camera_images = st.file_uploader(
                "Upload images from your camera",
                type=['jpg', 'jpeg', 'png'],
                accept_multiple_files=True,
                key="camera_uploader"
            )

            # Option 2: Capture en direct (simulée pour Streamlit Cloud)
            st.markdown("#### Live Camera Capture")
            st.info("💡 In a local environment, you can use OpenCV for live camera capture.")

            # Simulation de capture
            if st.button("📸 Simulate Camera Capture", use_container_width=True):
                # Créer des images simulées
                simulated_images = []
                for i in range(4):
                    # Créer une image colorée simulée
                    img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                    img = Image.fromarray(img_array)
                    simulated_images.append(img)

                st.session_state.camera_capture = simulated_images
                st.success("✅ 4 sample images captured!")

                # Afficher les images capturées
                st.markdown("#### Captured Images")
                cols = st.columns(4)
                for idx, img in enumerate(simulated_images):
                    with cols[idx]:
                        st.image(img, use_column_width=True, caption=f"Capture {idx + 1}")

            if camera_images or st.session_state.camera_capture:
                images_to_use = camera_images if camera_images else st.session_state.camera_capture

                if st.button("Use Captured Images →", type="primary", use_container_width=True):
                    with st.spinner("Processing captured images..."):
                        time.sleep(2)

                        # Créer des données simulées
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

        with tab4:
            st.markdown("### Use Demo Dataset")
            st.info("🎯 Quickly test the platform with sample data")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                    <div style='border: 2px solid #e2e8f0; border-radius: 1rem; padding: 2rem; text-align: center;'>
                        <div style='font-size: 3rem;'>📊</div>
                        <h3>Classification Demo</h3>
                        <p>Sample data with categorical targets</p>
                """, unsafe_allow_html=True)

                if st.button("Use Classification Demo", key="demo_class", use_container_width=True):
                    st.session_state.problem_type = 'classification'
                    st.session_state.data_preview = demo_data['classification']
                    st.session_state.columns = demo_data['classification'].columns.tolist()
                    st.session_state.target_column = 'target'
                    st.session_state.data_loaded = True
                    st.session_state.uploaded_data = demo_data['classification']
                    st.session_state.data_type = 'demo_tabular'
                    st.session_state.step = 2
                    st.rerun()

                st.markdown("</div>", unsafe_allow_html=True)

            with col2:
                st.markdown("""
                    <div style='border: 2px solid #e2e8f0; border-radius: 1rem; padding: 2rem; text-align: center;'>
                        <div style='font-size: 3rem;'>📈</div>
                        <h3>Regression Demo</h3>
                        <p>Sample data with numerical targets</p>
                """, unsafe_allow_html=True)

                if st.button("Use Regression Demo", key="demo_reg", use_container_width=True):
                    st.session_state.problem_type = 'regression'
                    st.session_state.data_preview = demo_data['regression']
                    st.session_state.columns = demo_data['regression'].columns.tolist()
                    st.session_state.target_column = 'target'
                    st.session_state.data_loaded = True
                    st.session_state.uploaded_data = demo_data['regression']
                    st.session_state.data_type = 'demo_tabular'
                    st.session_state.step = 2
                    st.rerun()

                st.markdown("</div>", unsafe_allow_html=True)

    # Étape 2: Sélection des algorithmes
    elif st.session_state.step == 2 and st.session_state.problem_type:
        st.markdown(f"## Algorithm Selection - {st.session_state.problem_type.title()}")

        # Aperçu des données
        st.markdown("### Dataset Overview")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Data Type", st.session_state.data_type)
        with col2:
            if hasattr(st.session_state, 'uploaded_data'):
                if st.session_state.data_type in ['tabular', 'demo_tabular']:
                    st.metric("Rows", len(st.session_state.uploaded_data))
                elif st.session_state.data_type in ['images', 'camera']:
                    st.metric("Images", len(st.session_state.uploaded_data))
        with col3:
            st.metric("Problem Type", st.session_state.problem_type)

        # Afficher l'aperçu selon le type de données
        if st.session_state.data_type in ['tabular', 'demo_tabular']:
            st.markdown("#### Data Preview")
            st.dataframe(st.session_state.data_preview, use_container_width=True)
        elif st.session_state.data_type in ['images', 'camera']:
            st.markdown("#### Sample Images")
            if st.session_state.data_type == 'images':
                images_to_show = st.session_state.uploaded_data[:4]
            else:
                images_to_show = st.session_state.uploaded_data[:4] if st.session_state.camera_capture else []

            if images_to_show:
                cols = st.columns(4)
                for idx, img in enumerate(images_to_show):
                    with cols[idx]:
                        if isinstance(img, Image.Image):
                            st.image(img, use_column_width=True, caption=f"Image {idx + 1}")
                        else:
                            image = Image.open(img)
                            st.image(image, use_column_width=True, caption=f"Image {idx + 1}")

            st.markdown("#### Dataset Info")
            st.dataframe(st.session_state.data_preview, use_container_width=True)

        # Navigation
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("View Results →", type="primary", disabled=len(st.session_state.trained_models) == 0):
                st.session_state.step = 3
                st.rerun()

            if st.button("← Change Dataset", use_container_width=True):
                st.session_state.step = 1
                st.rerun()

        # Sélection des algorithmes
        st.markdown("### Available Algorithms")

        # Adapter les algorithmes selon le type de données
        if st.session_state.data_type in ['images', 'camera']:
            # Algorithmes spécialisés pour les images
            image_algorithms = {
                'classification': {
                    'CNN Architectures': ['Simple CNN', 'Deep CNN', 'ResNet', 'VGG16', 'MobileNet'],
                    'Transfer Learning': ['ResNet50 + Fine-tuning', 'VGG19 + Fine-tuning', 'EfficientNet'],
                    'Traditional + Features': ['SVM + HOG', 'Random Forest + HOG', 'KNN + HOG']
                }
            }
            algo_to_use = image_algorithms.get(st.session_state.problem_type, algorithms[st.session_state.problem_type])
        else:
            algo_to_use = algorithms[st.session_state.problem_type]

        for category, models in algo_to_use.items():
            st.markdown(f"#### {category}")

            cols = st.columns(3)
            for idx, model in enumerate(models):
                col_idx = idx % 3
                with cols[col_idx]:
                    st.markdown(f"""
                        <div class="model-card">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                                <strong>{model}</strong>
                    """, unsafe_allow_html=True)

                    if model in st.session_state.trained_models:
                        if st.session_state.trained_models[model]['status'] == 'trained':
                            st.markdown("✅", unsafe_allow_html=True)

                    st.markdown("</div>", unsafe_allow_html=True)

                    if model in st.session_state.trained_models:
                        status = st.session_state.trained_models[model]['status']
                        if status == 'training':
                            if st.button("⏳ Training...", key=f"train_{model}", disabled=True,
                                         use_container_width=True):
                                pass
                        else:
                            if st.button("🔄 Retrain", key=f"retrain_{model}", use_container_width=True):
                                train_model(model, st.session_state.problem_type, st.session_state.data_type)
                    else:
                        if st.button("🚀 Train Model", key=f"train_{model}", use_container_width=True):
                            train_model(model, st.session_state.problem_type, st.session_state.data_type)

                    st.markdown("</div>", unsafe_allow_html=True)

    # Étape 3: Résultats (reste identique avec quelques adaptations)
    elif st.session_state.step == 3:
        st.markdown("## Training Results")

        if st.button("← Back to Training"):
            st.session_state.step = 2
            st.rerun()

        # Afficher le type de données utilisé
        st.info(f"**Data Type:** {st.session_state.data_type} | **Problem Type:** {st.session_state.problem_type}")

        # Comparaison des modèles
        trained_models_list = [name for name, model in st.session_state.trained_models.items()
                               if model['status'] == 'trained']

        if trained_models_list:
            st.markdown("### Model Performance Comparison")

            # Préparation des données pour le graphique
            comparison_data = []
            for model_name in trained_models_list:
                model_data = st.session_state.trained_models[model_name]
                if st.session_state.problem_type == 'classification':
                    score = float(model_data['metrics']['accuracy']) * 100
                else:
                    score = float(model_data['metrics']['r2Score']) * 100

                comparison_data.append({
                    'Model': model_name[:15] + '...' if len(model_name) > 15 else model_name,
                    'Score': score,
                    'Training Time': float(model_data['metrics']['trainTime'])
                })

            df_comparison = pd.DataFrame(comparison_data)

            # Graphique de comparaison
            fig = px.bar(df_comparison, x='Model', y='Score',
                         title=f"{'Accuracy' if st.session_state.problem_type == 'classification' else 'R² Score'} Comparison",
                         color='Score', color_continuous_scale='viridis')
            st.plotly_chart(fig, use_container_width=True)

            # Métriques détaillées
            st.markdown("### Detailed Model Metrics")

            for model_name in trained_models_list:
                model_data = st.session_state.trained_models[model_name]

                with st.expander(f"📊 {model_name} - Training Time: {model_data['metrics']['trainTime']}s",
                                 expanded=True):
                    # Métriques
                    metrics_cols = st.columns(4)
                    metric_items = list(model_data['metrics'].items())

                    for idx, (key, value) in enumerate(metric_items):
                        if key != 'trainTime':
                            col_idx = idx % 4
                            with metrics_cols[col_idx]:
                                st.markdown(f"""
                                    <div class="metric-card">
                                        <div style="font-size: 0.8rem; color: #64748b;">
                                            {key.replace('_', ' ').title()}
                                        </div>
                                        <div style="font-size: 1.5rem; font-weight: bold; color: #6366f1;">
                                            {value}
                                        </div>
                                    </div>
                                """, unsafe_allow_html=True)

                    # Graphiques supplémentaires pour la régression
                    if st.session_state.problem_type == 'regression' and 'predictions' in model_data:
                        st.markdown("#### Predictions vs Actual")

                        predictions_df = pd.DataFrame(model_data['predictions'])
                        fig_scatter = px.scatter(predictions_df, x='actual', y='predicted',
                                                 title="Actual vs Predicted Values")
                        fig_scatter.add_trace(go.Scatter(x=predictions_df['actual'],
                                                         y=predictions_df['actual'],
                                                         mode='lines', name='Perfect Prediction',
                                                         line=dict(dash='dash', color='green')))
                        st.plotly_chart(fig_scatter, use_container_width=True)

            # Matrice de confusion pour la classification
            if st.session_state.problem_type == 'classification':
                st.markdown("### Confusion Matrix (Sample)")

                confusion_data = {
                    'True Class': ['A', 'B', 'C'],
                    'Predicted A': [45, 2, 1],
                    'Predicted B': [3, 48, 4],
                    'Predicted C': [2, 5, 50]
                }

                confusion_df = pd.DataFrame(confusion_data)
                st.dataframe(confusion_df, use_container_width=True)

                # Heatmap de la matrice de confusion
                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=[[45, 3, 2], [2, 48, 5], [1, 4, 50]],
                    x=['Predicted A', 'Predicted B', 'Predicted C'],
                    y=['True A', 'True B', 'True C'],
                    colorscale='Blues',
                    showscale=True
                ))
                fig_heatmap.update_layout(title="Confusion Matrix Heatmap")
                st.plotly_chart(fig_heatmap, use_container_width=True)


def train_model(model_name, problem_type, data_type):
    """Simule l'entraînement d'un modèle"""
    # Marquer le modèle comme en cours d'entraînement
    st.session_state.trained_models[model_name] = {'status': 'training'}

    # Simulation de l'entraînement avec une barre de progression
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Message différent selon le type de données
    if data_type in ['images', 'camera']:
        status_message = f"Training {model_name} on images..."
    else:
        status_message = f"Training {model_name}..."

    for i in range(100):
        progress_bar.progress(i + 1)
        status_text.text(f"{status_message} {i + 1}%")
        time.sleep(0.02)

    status_text.text("")
    progress_bar.empty()

    # Génération de métriques simulées
    if problem_type == 'classification':
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

    # Génération de prédictions simulées
    predictions = []
    for i in range(20):
        base_value = 10 + i * 2
        actual = base_value + (np.random.random() - 0.5) * 2
        predicted = base_value + (np.random.random() - 0.5) * 3
        predictions.append({
            'actual': actual,
            'predicted': predicted,
            'index': i
        })

    st.session_state.trained_models[model_name] = {
        'status': 'trained',
        'metrics': metrics,
        'predictions': predictions
    }

    st.rerun()


if __name__ == "__main__":
    main()
