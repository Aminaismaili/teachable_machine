"""
classical_ml.py
Module pour les algorithmes de Machine Learning classiques (scikit-learn)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, roc_curve
)
import pickle
import logging

# Classification
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

# Regression
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClassicalMLTrainer:
    """
    Classe pour entraîner et évaluer des modèles ML classiques
    """
    
    def __init__(self, problem_type='classification'):
        """
        Initialise le trainer
        
        Args:
            problem_type: 'classification' ou 'regression'
        """
        self.problem_type = problem_type.lower()
        self.models = {}
        self.results = {}
        self.best_model_name = None
        self.best_model = None
        
        if self.problem_type not in ['classification', 'regression']:
            raise ValueError("problem_type doit être 'classification' ou 'regression'")
    
    def get_available_models(self):
        """
        Retourne la liste des modèles disponibles selon le type de problème
        
        Returns:
            dict: Dictionnaire {nom: modèle}
        """
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
        else:  # regression
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
        """
        Entraîne un seul modèle
        
        Args:
            model_name: Nom du modèle
            X_train, y_train: Données d'entraînement
            X_test, y_test: Données de test
            
        Returns:
            dict: Résultats du modèle
        """
        available_models = self.get_available_models()
        
        if model_name not in available_models:
            raise ValueError(f"Modèle '{model_name}' non disponible")
        
        try:
            logger.info(f"Entraînement de {model_name}...")
            
            model = available_models[model_name]
            model.fit(X_train, y_train)
            
            # Prédictions
            y_pred = model.predict(X_test)
            y_train_pred = model.predict(X_train)
            
            # Calcul des métriques
            metrics = self._calculate_metrics(y_test, y_pred, y_train, y_train_pred, model, X_test)
            
            # Stocker le modèle et les résultats
            self.models[model_name] = model
            result = {
                'model': model,
                'predictions': y_pred,
                'train_predictions': y_train_pred,
                'metrics': metrics
            }
            self.results[model_name] = result
            
            logger.info(f"{model_name} entraîné avec succès")
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement de {model_name}: {str(e)}")
            return None
    
    def train_all_models(self, X_train, y_train, X_test, y_test, selected_models=None):
        """
        Entraîne plusieurs modèles
        
        Args:
            X_train, y_train: Données d'entraînement
            X_test, y_test: Données de test
            selected_models: Liste des noms de modèles (None = tous)
            
        Returns:
            dict: Résultats de tous les modèles
        """
        available_models = self.get_available_models()
        
        if selected_models is None:
            selected_models = list(available_models.keys())
        
        logger.info(f"Entraînement de {len(selected_models)} modèles...")
        
        for model_name in selected_models:
            self.train_single_model(model_name, X_train, y_train, X_test, y_test)
        
        # Identifier le meilleur modèle
        self._identify_best_model()
        
        return self.results
    
    def _calculate_metrics(self, y_test, y_pred, y_train, y_train_pred, model, X_test):
        """
        Calcule les métriques selon le type de problème
        
        Args:
            y_test: Vraies valeurs test
            y_pred: Prédictions test
            y_train: Vraies valeurs train
            y_train_pred: Prédictions train
            model: Modèle entraîné
            X_test: Features test
            
        Returns:
            dict: Métriques calculées
        """
        metrics = {}
        
        if self.problem_type == 'classification':
            # Métriques de base
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            metrics['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics['f1_score'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Scores train (pour détecter overfitting)
            metrics['train_accuracy'] = accuracy_score(y_train, y_train_pred)
            
            # Matrice de confusion
            metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)
            
            # ROC AUC pour binaire ou multiclass
            try:
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)
                    if len(np.unique(y_test)) == 2:  # Binaire
                        metrics['roc_auc'] = roc_auc_score(y_test, y_proba[:, 1])
                    else:  # Multiclass
                        metrics['roc_auc'] = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
            except:
                metrics['roc_auc'] = None
            
        else:  # regression
            metrics['r2_score'] = r2_score(y_test, y_pred)
            metrics['mse'] = mean_squared_error(y_test, y_pred)
            metrics['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
            metrics['mae'] = mean_absolute_error(y_test, y_pred)
            
            # Scores train
            metrics['train_r2'] = r2_score(y_train, y_train_pred)
            
            # MAPE (Mean Absolute Percentage Error)
            try:
                metrics['mape'] = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            except:
                metrics['mape'] = None
        
        return metrics
    
    def _identify_best_model(self):
        """
        Identifie le meilleur modèle basé sur la métrique principale
        """
        if not self.results:
            return
        
        if self.problem_type == 'classification':
            metric_key = 'accuracy'
        else:
            metric_key = 'r2_score'
        
        best_score = -float('inf')
        
        for model_name, result in self.results.items():
            score = result['metrics'].get(metric_key, -float('inf'))
            if score > best_score:
                best_score = score
                self.best_model_name = model_name
                self.best_model = result['model']
        
        logger.info(f"Meilleur modèle: {self.best_model_name} ({metric_key}={best_score:.4f})")
    
    def get_comparison_dataframe(self):
        """
        Retourne un DataFrame comparatif des modèles
        
        Returns:
            DataFrame: Comparaison des métriques
        """
        if not self.results:
            return None
        
        comparison_data = {}
        
        for model_name, result in self.results.items():
            comparison_data[model_name] = result['metrics']
        
        df = pd.DataFrame(comparison_data).T
        
        # Trier par métrique principale
        if self.problem_type == 'classification':
            df = df.sort_values('accuracy', ascending=False)
        else:
            df = df.sort_values('r2_score', ascending=False)
        
        return df
    
    def get_feature_importance(self, model_name=None, feature_names=None):
        """
        Retourne l'importance des features si disponible
        
        Args:
            model_name: Nom du modèle (None = meilleur modèle)
            feature_names: Noms des features
            
        Returns:
            DataFrame ou None
        """
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
    
    def predict(self, X, model_name=None):
        """
        Fait des prédictions avec un modèle
        
        Args:
            X: Features
            model_name: Nom du modèle (None = meilleur modèle)
            
        Returns:
            Prédictions
        """
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.models:
            raise ValueError(f"Modèle '{model_name}' non trouvé")
        
        return self.models[model_name].predict(X)
    
    def predict_proba(self, X, model_name=None):
        """
        Retourne les probabilités (classification uniquement)
        
        Args:
            X: Features
            model_name: Nom du modèle (None = meilleur modèle)
            
        Returns:
            Probabilités ou None
        """
        if self.problem_type != 'classification':
            return None
        
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.models:
            raise ValueError(f"Modèle '{model_name}' non trouvé")
        
        model = self.models[model_name]
        
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)
        
        return None
    
    def save_model(self, model_name=None, path=None):
        """
        Sauvegarde un modèle
        
        Args:
            model_name: Nom du modèle (None = meilleur modèle)
            path: Chemin de sauvegarde
        """
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.models:
            raise ValueError(f"Modèle '{model_name}' non trouvé")
        
        if path is None:
            path = f"{model_name.replace(' ', '_').lower()}_model.pkl"
        
        with open(path, 'wb') as f:
            pickle.dump(self.models[model_name], f)
        
        logger.info(f"Modèle sauvegardé: {path}")
        return path
    
    def load_model(self, path, model_name='loaded_model'):
        """
        Charge un modèle sauvegardé
        
        Args:
            path: Chemin du modèle
            model_name: Nom à donner au modèle
        """
        with open(path, 'rb') as f:
            model = pickle.load(f)
        
        self.models[model_name] = model
        logger.info(f"Modèle chargé: {path}")
    
    def get_model_summary(self, model_name=None):
        """
        Retourne un résumé du modèle
        
        Args:
            model_name: Nom du modèle (None = meilleur modèle)
            
        Returns:
            dict: Résumé du modèle
        """
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.results:
            return None
        
        result = self.results[model_name]
        model = self.models[model_name]
        
        summary = {
            'model_name': model_name,
            'model_type': type(model).__name__,
            'parameters': model.get_params(),
            'metrics': result['metrics'],
            'is_best': model_name == self.best_model_name
        }
        
        return summary


if __name__ == "__main__":
    # Test du module
    print("Module ClassicalMLTrainer chargé avec succès!")
    print(f"\nModèles de Classification disponibles: {len(ClassicalMLTrainer('classification').get_available_models())}")
    print(f"Modèles de Régression disponibles: {len(ClassicalMLTra
