# app.py — Streamlit Vision ML/DL (Images: Classification & Régression) + Grad‑CAM + Rapport HTML/PDF
# ---------------------------------------------------------------------------------
# Nouveautés dans cette version :
# 1) **Grad‑CAM** pour visualiser où le CNN "regarde" (après entraînement et en prédiction)
# 2) **Rapport automatique HTML + PDF** (métriques, figures, gaps d'overfitting) téléchargeables
# ---------------------------------------------------------------------------------

import os
import io
import time
import zipfile
import tempfile
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import pandas as pd
from PIL import Image

import streamlit as st
import matplotlib.pyplot as plt

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Rapports
from jinja2 import Template
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader

st.set_page_config(page_title="Vision ML/DL — Images (Classif & Régression)", layout="wide")
ARTIFACT_DIR = "artifacts"; os.makedirs(ARTIFACT_DIR, exist_ok=True)
REPORT_DIR = os.path.join(ARTIFACT_DIR, "reports"); os.makedirs(REPORT_DIR, exist_ok=True)
FIG_DIR = os.path.join(ARTIFACT_DIR, "figs"); os.makedirs(FIG_DIR, exist_ok=True)

# ===========================
# Utilitaires d'affichage (matplotlib, sans seaborn)
# ===========================

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], title: str = "Matrice de confusion"):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='Vrai', xlabel='Prédit', title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig


def plot_history_curves(history: keras.callbacks.History, metric_pairs: List[Tuple[str,str]]):
    figs = []
    for m, m_val in metric_pairs:
        if m in history.history and m_val in history.history:
            fig, ax = plt.subplots()
            ax.plot(history.history[m]); ax.plot(history.history[m_val])
            ax.set_title(f"{m} / {m_val}")
            ax.set_xlabel("Epoch"); ax.set_ylabel(m)
            figs.append((f"{m}", fig))
    return figs


def plot_regression_scatter(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Vrai vs Prédit"):
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred)
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx])
    ax.set_title(title); ax.set_xlabel("y"); ax.set_ylabel("ŷ")
    return fig


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Résidus"):
    resid = y_true - y_pred
    fig, ax = plt.subplots()
    ax.scatter(y_pred, resid)
    ax.axhline(0)
    ax.set_title(title); ax.set_xlabel("ŷ"); ax.set_ylabel("y - ŷ")
    return fig

# ===========================
# Ingestion / Datasets
# ===========================

def extract_zip_to_temp(uploaded_zip) -> str:
    tmpdir = tempfile.mkdtemp()
    with zipfile.ZipFile(uploaded_zip) as z:
        z.extractall(tmpdir)
    return tmpdir

# --- Classification : directory class/ with image files

def make_classification_datasets(root_dir: str, img_size=(224,224), batch_size=32, val_split=0.2, seed=42):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        root_dir, validation_split=val_split, subset="training", seed=seed,
        image_size=img_size, batch_size=batch_size
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        root_dir, validation_split=val_split, subset="validation", seed=seed,
        image_size=img_size, batch_size=batch_size
    )
    class_names = train_ds.class_names
    AUTOTUNE = tf.data.AUTOTUNE
    return train_ds.prefetch(AUTOTUNE), val_ds.prefetch(AUTOTUNE), class_names

# --- Régression : folder of images + CSV (filename,value)

def load_image_for_reg(path: str, target_size=(224,224)) -> np.ndarray:
    img = Image.open(path).convert('RGB').resize(target_size)
    arr = np.asarray(img, dtype=np.float32)
    return arr


def make_regression_dataset(images_dir: str, labels_csv: bytes, img_size=(224,224), batch_size=32, val_split=0.2, seed=42):
    df = pd.read_csv(io.BytesIO(labels_csv))
    if not {'filename','value'}.issubset(df.columns):
        raise ValueError("CSV doit contenir les colonnes: filename,value")
    paths = [os.path.join(images_dir, fn) for fn in df['filename']]
    y = df['value'].astype(np.float32).values
    X = np.stack([load_image_for_reg(p, img_size) for p in paths])
    X = X / 255.0
    # split
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(val_split*n)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val   = X[val_idx], y[val_idx]

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds   = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds

# ===========================
# Modèles
# ===========================

def build_mini_cnn_classifier(num_classes: int, img_size=(224,224), dropout=0.2, l2=0.0):
    reg = keras.regularizers.l2(l2) if l2>0 else None
    inputs = layers.Input(shape=(img_size[0], img_size[1], 3))
    x = layers.Conv2D(32, 3, activation='relu', kernel_regularizer=reg)(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu', kernel_regularizer=reg)(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=reg)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def build_mobilenet_classifier(num_classes: int, img_size=(224,224), dropout=0.2, l2=0.0, fine_tune=False, lr=1e-3):
    reg = keras.regularizers.l2(l2) if l2>0 else None
    base = tf.keras.applications.MobileNetV2(input_shape=(img_size[0], img_size[1], 3), include_top=False, weights='imagenet')
    base.trainable = bool(fine_tune)
    inputs = layers.Input(shape=(img_size[0], img_size[1], 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='softmax', kernel_regularizer=reg)(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def build_mobilenet_regressor(img_size=(224,224), dropout=0.2, l2=0.0, fine_tune=False, lr=1e-3):
    reg = keras.regularizers.l2(l2) if l2>0 else None
    base = tf.keras.applications.MobileNetV2(input_shape=(img_size[0], img_size[1], 3), include_top=False, weights='imagenet')
    base.trainable = bool(fine_tune)
    inputs = layers.Input(shape=(img_size[0], img_size[1], 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation='linear', kernel_regularizer=reg)(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(lr), loss='mse', metrics=['mae'])
    return model

# ===========================
# Évaluation num & plots + Grad‑CAM
# ===========================

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred)**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true))**2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float('nan')
    return {"rmse": rmse, "mae": mae, "r2": r2}

# --- Grad‑CAM helpers ---

def find_last_conv_layer(model: keras.Model) -> Optional[str]:
    for layer in reversed(model.layers):
        if isinstance(layer, (layers.Conv2D, layers.SeparableConv2D, layers.DepthwiseConv2D)):
            return layer.name
    for layer in reversed(model.layers):
        if hasattr(layer, 'layers'):
            for sub in reversed(layer.layers):
                if isinstance(sub, (layers.Conv2D, layers.SeparableConv2D, layers.DepthwiseConv2D)):
                    return sub.name
    return None


def make_gradcam_heatmap(img_array: np.ndarray, model: keras.Model, last_conv_layer_name: str, pred_index: Optional[int]=None) -> np.ndarray:
    grad_model = keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(pil_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.35) -> Image.Image:
    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap).resize(pil_img.size)
    import matplotlib.cm as cm
    col = cm.get_cmap('jet')(np.array(heatmap) / 255.0)
    col = (col * 255).astype('uint8')
    col_img = Image.fromarray(col[:, :, :3])
    col_img.putalpha(int(alpha * 255))
    out = pil_img.convert('RGBA')
    out.alpha_composite(col_img)
    return out.convert('RGB')

# ===========================
# UI
# ===========================

st.title("🧠 Application Vision — Classification & Régression d'Images (Streamlit)")
st.write("Créez, entraînez, évaluez et déployez des modèles CNN via une interface simple. **Grad‑CAM** et **Rapports** intégrés.")

with st.sidebar:
    st.header("Paramètres globaux")
    seed = st.number_input("Seed", 0, 99999, 42)
    img_w = st.number_input("Largeur image", 64, 640, 224, step=32)
    img_h = st.number_input("Hauteur image", 64, 640, 224, step=32)
    img_size = (int(img_h), int(img_w))
    task = st.radio("Tâche", ["Classification", "Régression"])

    model_family = st.radio("Modèle", ["Mini-CNN", "MobileNetV2"])
    batch_size = st.selectbox("Batch size", [16, 32, 64], index=1)
    epochs = st.slider("Epochs", 3, 50, 10)
    val_split = st.slider("Validation split", 0.1, 0.4, 0.2, 0.05)

    st.subheader("Anti-overfitting")
    dropout = st.slider("Dropout", 0.0, 0.7, 0.2, 0.05)
    l2 = st.slider("L2", 0.0, 0.02, 0.001, 0.001)
    fine_tune = st.checkbox("Fine-tuning base", False)
    lr = st.number_input("Learning rate", value=1e-3, format="%e")

# Mémoire runs
if 'runs' not in st.session_state: st.session_state['runs'] = {}

st.markdown("---")

# ===========================
# 1) Données
# ===========================

st.header("1) Données")
if task == "Classification":
    zip_cls = st.file_uploader("Charger un ZIP structuré (class_name/)", type=["zip"])
    if zip_cls is not None:
        root = extract_zip_to_temp(zip_cls)
        with st.spinner("Préparation du dataset..."):
            train_ds, val_ds, class_names = make_classification_datasets(root, img_size=img_size, batch_size=batch_size, val_split=val_split, seed=seed)
        st.success(f"Classes: {class_names}")
        st.write(f"Train batches: {len(train_ds)} — Val batches: {len(val_ds)}")
        st.session_state['data'] = {"kind":"cls", "train": train_ds, "val": val_ds, "classes": class_names}
else:
    img_folder = st.file_uploader("(Régression) Charger un ZIP d'images **non classées**", type=["zip"], key="reg_zip")
    labels_csv = st.file_uploader("(Régression) CSV 'filename,value'", type=["csv"], key="reg_csv")
    if img_folder is not None and labels_csv is not None:
        img_root = extract_zip_to_temp(img_folder)
        with st.spinner("Construction du dataset régression..."):
            train_ds, val_ds = make_regression_dataset(img_root, labels_csv.read(), img_size=img_size, batch_size=batch_size, val_split=val_split, seed=seed)
        st.success(f"Train batches: {len(train_ds)} — Val batches: {len(val_ds)}")
        st.session_state['data'] = {"kind":"reg", "train": train_ds, "val": val_ds}

# ===========================
# 2) Modèle & Entraînement
# ===========================

st.header("2) Modèle & Entraînement")
if 'data' not in st.session_state:
    st.info("Chargez les données d'abord.")
else:
    info = st.session_state['data']
    if task == "Classification":
        num_classes = len(info['classes'])
        if model_family == "Mini-CNN":
            model = build_mini_cnn_classifier(num_classes, img_size, dropout=dropout, l2=l2)
        else:
            model = build_mobilenet_classifier(num_classes, img_size, dropout=dropout, l2=l2, fine_tune=fine_tune, lr=lr)
        metric_pairs = [("loss","val_loss"),("accuracy","val_accuracy")]
    else:
        model = build_mobilenet_regressor(img_size, dropout=dropout, l2=l2, fine_tune=fine_tune, lr=lr)
        metric_pairs = [("loss","val_loss"),("mae","val_mae")]

    es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    if st.button("🚀 Entraîner le modèle"):
        with st.spinner("Training en cours..."):
            history = model.fit(info['train'], validation_data=info['val'], epochs=epochs, verbose=0, callbacks=[es])
        st.success("Entraînement terminé")

        # Plots apprentissage
        figs = plot_history_curves(history, metric_pairs)
        saved_fig_paths = []
        for name, fig in figs:
            st.pyplot(fig)
            fig_path = os.path.join(FIG_DIR, f"{name}_{int(time.time())}.png")
            fig.savefig(fig_path, bbox_inches='tight'); saved_fig_paths.append(fig_path)

        # Evaluation & extras
        report_ctx: Dict[str, Any] = {"task": task, "model": model_family, "figs": saved_fig_paths, "history": {k: list(map(float, v)) for k,v in history.history.items()}}

        if task == "Classification":
            # Confusion
            from sklearn.metrics import confusion_matrix, accuracy_score
            y_true, y_pred = [], []
            for xb, yb in info['val']:
                p = model.predict(xb, verbose=0)
                y_true.extend(yb.numpy().tolist())
                y_pred.extend(np.argmax(p, axis=1).tolist())
            y_true = np.array(y_true); y_pred = np.array(y_pred)
            cm = confusion_matrix(y_true, y_pred)
            fig_cm = plot_confusion_matrix(cm, info['classes'])
            st.pyplot(fig_cm)
            cm_path = os.path.join(FIG_DIR, f"cm_{int(time.time())}.png")
            fig_cm.savefig(cm_path, bbox_inches='tight'); saved_fig_paths.append(cm_path)
            acc = float(accuracy_score(y_true, y_pred))
            st.write({"accuracy_val": acc})
            report_ctx.update({"accuracy_val": acc, "classes": info['classes']})

            # Grad‑CAM sur quelques images de validation
            last_conv = find_last_conv_layer(model)
            if last_conv is not None:
                st.subheader("Grad‑CAM (échantillon de validation)")
                shown = 0
                for xb, yb in info['val']:
                    preds = model.predict(xb, verbose=0)
                    for i in range(min(len(yb), 4)):
                        img = (xb[i].numpy()*255).astype(np.uint8)
                        pil = Image.fromarray(img)
                        arr = np.expand_dims(xb[i].numpy(), 0)
                        heat = make_gradcam_heatmap(arr, model, last_conv)
                        over = overlay_gradcam(pil, heat)
                        c1, c2 = st.columns(2)
                        with c1: st.image(pil, caption="Image")
                        with c2: st.image(over, caption="Grad‑CAM")
                        # Sauvegarder visuels
                        raw_path = os.path.join(FIG_DIR, f"raw_{time.time_ns()}.png"); pil.save(raw_path)
                        cam_path = os.path.join(FIG_DIR, f"cam_{time.time_ns()}.png"); over.save(cam_path)
                        saved_fig_paths += [raw_path, cam_path]
                        shown += 1
                    if shown >= 4: break
        else:
            # Régression : métriques et plots
            y_true, y_hat = [], []
            for xb, yb in info['val']:
                preds = model.predict(xb, verbose=0).squeeze()
                y_true.extend(yb.numpy().tolist())
                y_hat.extend(preds.tolist())
            y_true = np.array(y_true, dtype=np.float32)
            y_hat = np.array(y_hat, dtype=np.float32)
            mets = regression_metrics(y_true, y_hat)
            st.json(mets)
            fig_sc = plot_regression_scatter(y_true, y_hat); st.pyplot(fig_sc)
            fig_res = plot_residuals(y_true, y_hat); st.pyplot(fig_res)
            p1 = os.path.join(FIG_DIR, f"scatter_{int(time.time())}.png"); fig_sc.savefig(p1, bbox_inches='tight')
            p2 = os.path.join(FIG_DIR, f"resid_{int(time.time())}.png"); fig_res.savefig(p2, bbox_inches='tight')
            saved_fig_paths += [p1, p2]
            report_ctx.update(mets)

        # Gaps d'overfitting
        def last(h, k): return float(h[k][-1]) if k in h else float('nan')
        tr_loss, va_loss = last(history.history, 'loss'), last(history.history, 'val_loss')
        gap_loss = va_loss - tr_loss
        report_ctx.update({"gap_loss": gap_loss, "train_loss": tr_loss, "val_loss": va_loss})
        if 'accuracy' in history.history:
            tr_acc, va_acc = last(history.history, 'accuracy'), last(history.history, 'val_accuracy')
            report_ctx.update({"gap_acc": tr_acc - va_acc, "train_acc": tr_acc, "val_acc": va_acc})

        # Sauvegarde modèle
        run_id = f"{('CLS' if task=='Classification' else 'REG')}_{int(time.time())}"
        model_path = os.path.join(ARTIFACT_DIR, f"{run_id}.h5")
        model.save(model_path)
        st.session_state['runs'][run_id] = {"path": model_path, "task": task, "model": model_family, "classes": info.get('classes'), "report_ctx": report_ctx}
        with open(model_path, 'rb') as f:
            st.download_button("💾 Télécharger le modèle (.h5)", f, file_name=os.path.basename(model_path))

        # ====== Génération rapport HTML ======
        st.subheader("📄 Rapport automatique (HTML & PDF)")
        html_tpl = Template(
            """
            <html><head><meta charset='utf-8'><title>Rapport {{ run_id }}</title></head>
            <body>
            <h1>Rapport — {{ task }} ({{ model }})</h1>
            <h3>Overfitting</h3>
            <p>train loss: {{ train_loss | round(4) }} | val loss: {{ val_loss | round(4) }} | gap (val - train): <b>{{ gap_loss | round(4) }}</b></p>
            {% if gap_acc is defined %}
            <p>train acc: {{ train_acc | round(4) }} | val acc: {{ val_acc | round(4) }} | gap (train - val): <b>{{ gap_acc | round(4) }}</b></p>
            {% endif %}
            {% if accuracy_val is defined %}<p><b>Accuracy (val): {{ accuracy_val | round(4) }}</b></p>{% endif %}
            {% if rmse is defined %}<p><b>RMSE:</b> {{ rmse | round(4) }} | <b>MAE:</b> {{ mae | round(4) }} | <b>R²:</b> {{ r2 | round(4) }}</p>{% endif %}
            <h3>Figures</h3>
            {% for p in figs %}<div><img src='{{ p }}' width='480'></div>{% endfor %}
            </body></html>
            """
        )
        html_content = html_tpl.render(run_id=run_id, **report_ctx)
        html_path = os.path.join(REPORT_DIR, f"{run_id}.html")
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        with open(html_path, 'rb') as f:
            st.download_button("⬇️ Télécharger le rapport HTML", f, file_name=os.path.basename(html_path), mime="text/html")

        # ====== Génération rapport PDF minimal (ReportLab) ======
        pdf_path = os.path.join(REPORT_DIR, f"{run_id}.pdf")
        c = canvas.Canvas(pdf_path, pagesize=A4)
        W, H = A4
        y = H - 2*cm
        c.setFont("Helvetica-Bold", 14); c.drawString(2*cm, y, f"Rapport — {task} ({model_family})"); y -= 1.0*cm
        c.setFont("Helvetica", 10)
        c.drawString(2*cm, y, f"train loss: {tr_loss:.4f} | val loss: {va_loss:.4f} | gap: {gap_loss:.4f}"); y -= 0.7*cm
        if 'gap_acc' in report_ctx:
            c.drawString(2*cm, y, f"train acc: {report_ctx['train_acc']:.4f} | val acc: {report_ctx['val_acc']:.4f} | gap: {report_ctx['gap_acc']:.4f}"); y -= 0.7*cm
        if 'accuracy_val' in report_ctx:
            c.drawString(2*cm, y, f"accuracy(val): {report_ctx['accuracy_val']:.4f}"); y -= 0.7*cm
        if 'rmse' in report_ctx:
            c.drawString(2*cm, y, f"RMSE: {report_ctx['rmse']:.4f} | MAE: {report_ctx['mae']:.4f} | R²: {report_ctx['r2']:.4f}"); y -= 0.7*cm
        # Insérer quelques figures (max 3 par page)
        for i, p in enumerate(saved_fig_paths[:6]):
            if y < 6*cm:
                c.showPage(); y = H - 2*cm
            try:
                img = ImageReader(p)
                c.drawImage(img, 2*cm, y-6*cm, width=12*cm, height=6*cm, preserveAspectRatio=True, mask='auto')
                y -= 6.5*cm
            except Exception:
                continue
        c.save()
        with open(pdf_path, 'rb') as f:
            st.download_button("⬇️ Télécharger le rapport PDF", f, file_name=os.path.basename(pdf_path), mime="application/pdf")

st.markdown("---")

# ===========================
# 3) Prédiction en temps réel + Grad‑CAM
# ===========================

st.header("3) Prédiction en temps réel & Grad‑CAM")
if not st.session_state['runs']:
    st.info("Entraînez un modèle pour activer la prédiction.")
else:
    chosen = st.selectbox("Choisir un modèle entraîné", list(st.session_state['runs'].keys()))
    meta = st.session_state['runs'][chosen]

    if st.button("Charger le modèle choisi"):
        st.session_state['loaded'] = keras.models.load_model(meta['path'])
        st.success("Modèle chargé.")

    if 'loaded' in st.session_state:
        model_loaded = st.session_state['loaded']
        st.subheader("Source d'image")
        c1, c2 = st.columns(2)
        with c1:
            cam = st.camera_input("Prendre une photo")
        with c2:
            up = st.file_uploader("… ou uploader une image", type=["jpg","jpeg","png"], key="pred_img")

        def prepare(img: Image.Image, target=(224,224)):
            x = img.convert('RGB').resize(target)
            x = np.asarray(x, dtype=np.float32)
            x = np.expand_dims(x, 0)
            return x

        src = None
        if cam is not None:
            src = Image.open(cam)
        elif up is not None:
            src = Image.open(up)

        if src is not None:
            st.image(src, caption="Image saisie", width=256)
            x = prepare(src, (img_size[1], img_size[0]))
            preds = model_loaded.predict(x, verbose=0)

            if meta['task'] == "Classification":
                probs = preds[0]
                classes = meta.get('classes', [str(i) for i in range(len(probs))])
                top = int(np.argmax(probs))
                st.write(f"**Classe prédite**: {classes[top]} — confiance: {float(probs[top]):.3f}")
                order = np.argsort(probs)[::-1][:3]
                st.write({classes[i]: float(probs[i]) for i in order})

                # Grad‑CAM sur l'image choisie
                last_conv = find_last_conv_layer(model_loaded)
                if last_conv:
                    heat = make_gradcam_heatmap(x, model_loaded, last_conv, pred_index=top)
                    over = overlay_gradcam(src, heat)
                    st.image(over, caption="Grad‑CAM (overlay)", width=256)
            else:
                value = float(preds.squeeze())
                st.write(f"**Valeur prédite**: {value:.4f}")

st.caption("© Vision App — Grad‑CAM + Rapports HTML/PDF. Export .h5 prêt pour déploiement.")
