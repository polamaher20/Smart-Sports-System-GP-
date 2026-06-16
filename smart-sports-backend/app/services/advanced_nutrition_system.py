# app/services/advanced_nutrition_system.py
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import numpy as np
import joblib
import math
import random
import tensorflow as tf
from typing import Dict, List
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.losses import MeanSquaredError
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ML_DIR = Path(r"E:\Graduation Project\Graduation Project Final\smart-sports-backend\app\infrastructure\ml_models")


def patch_legacy_h5_input_layer(model_path):
    try:
        import h5py
        import json
    except ImportError:
        return False

    try:
        with h5py.File(model_path, 'r+') as f:
            raw = f.attrs.get('model_config')
            if isinstance(raw, bytes):
                raw = raw.decode('utf-8')
            config = json.loads(raw)
            model_config = config.get('config', {})
            build_input_shape = model_config.get('build_input_shape')
            input_layer_name = None
            output_layer_name = None

            if build_input_shape and isinstance(build_input_shape, list) and len(build_input_shape) > 1:
                input_shape = build_input_shape[1:]
                batch_shape = build_input_shape
            else:
                input_shape = None
                batch_shape = None

            layers = model_config.get('layers', [])
            for idx, layer in enumerate(layers):
                if layer.get('class_name') == 'InputLayer':
                    cfg = layer.setdefault('config', {})
                    if input_shape is not None and cfg.get('input_shape') is None:
                        cfg['input_shape'] = input_shape
                    if batch_shape is not None and cfg.get('batch_shape') is None:
                        cfg['batch_shape'] = batch_shape
                    cfg.setdefault('dtype', 'float32')
                    input_layer_name = cfg.get('name', input_layer_name)

                if layer.get('class_name') == 'Dense':
                    cfg = layer.setdefault('config', {})
                    cfg.pop('quantization_config', None)
                    cfg.pop('batch_input_shape', None)
                    if cfg.get('dtype') and isinstance(cfg['dtype'], dict) and cfg['dtype'].get('class_name') == 'DTypePolicy':
                        cfg['dtype'] = cfg['dtype'].get('config', {}).get('name', 'float32')

                if 'quantization_config' in layer:
                    layer.pop('quantization_config', None)
                if 'batch_input_shape' in layer:
                    layer.pop('batch_input_shape', None)

                if idx == len(layers) - 1:
                    output_layer_name = layer.get('config', {}).get('name')

            if model_config.get('input_layers') is None and input_layer_name is not None:
                model_config['input_layers'] = [[input_layer_name, 0, 0]]
            if model_config.get('output_layers') is None and output_layer_name is not None:
                model_config['output_layers'] = [[output_layer_name, 0, 0]]

            config['config'] = model_config
            f.attrs['model_config'] = json.dumps(config)
        return True
    except Exception:
        return False


tf.get_logger().setLevel('ERROR')


class LegacyInputLayer(InputLayer):
    def __init__(self, *args, batch_shape=None, optional=False, **kwargs):
        if batch_shape is not None and 'input_shape' not in kwargs:
            if isinstance(batch_shape, (list, tuple)) and len(batch_shape) > 1:
                kwargs['input_shape'] = tuple(batch_shape[1:])
            elif isinstance(batch_shape, (list, tuple)) and len(batch_shape) == 1:
                kwargs['input_shape'] = (batch_shape[0],)
        super().__init__(*args, **kwargs)


class AdvancedNutritionSystem:
    def __init__(self, model_path=None, scaler_path=None, features_path=None):
        # Defaults to workspace ML_DIR if not provided
        model_path = model_path or str(ML_DIR / "nutrition_model.h5")
        scaler_path = scaler_path or str(ML_DIR / "scaler.pkl")
        features_path = features_path or str(ML_DIR / "feature_columns.pkl")

        self.model = None
        self.scaler = None
        self.feature_columns = None

        if str(model_path).lower().endswith('.h5'):
            patched = patch_legacy_h5_input_layer(model_path)
            if patched:
                print("✅ Patched legacy H5 config before loading the model.")

        try:
            self.model = load_model(
                model_path,
                compile=False,
                custom_objects={
                    "InputLayer": LegacyInputLayer,
                    "LegacyInputLayer": LegacyInputLayer
                }
            )
        except Exception as e:
            print(f"⚠️ Neural Network loading failed after patch attempt: {e}")
            if str(model_path).lower().endswith('.h5'):
                try:
                    self.model = load_model(
                        model_path,
                        compile=False,
                        custom_objects={
                            "InputLayer": LegacyInputLayer,
                            "LegacyInputLayer": LegacyInputLayer
                        }
                    )
                    print("✅ Loaded model successfully after retry.")
                except Exception as retry_error:
                    print(f"⚠️ Retry after patching legacy config also failed: {retry_error}")

        try:
            try:
                from sklearn.exceptions import InconsistentVersionWarning
                warnings.filterwarnings('ignore', category=InconsistentVersionWarning)
            except Exception:
                pass
            self.scaler = joblib.load(scaler_path)
        except Exception as e:
            print(f"⚠️ Scaler not found or failed to load: {e}")

        try:
            self.feature_columns = joblib.load(features_path)
        except Exception as e:
            print(f"⚠️ Feature columns not found or failed to load: {e}")

        # BMI categories
        self.BMI_CATEGORIES = {
            (0, 18.5): 'Underweight',
            (18.5, 25): 'Normal',
            (25, 30): 'Overweight',
            (30, 40): 'Obese',
            (40, float('inf')): 'Extremely_Obese'
        }

        # Initialize food DB
        self.create_healthy_food_database()


    def create_healthy_food_database(self):
        self.food_db = pd.DataFrame([
           # Protein
            {'ingr_name': 'chicken breast', 'category': 'Protein', 'cal/g': 1.65, 'protein(g)': 0.31, 'carb(g)': 0.00, 'fat(g)': 0.036},
            {'ingr_name': 'grilled chicken', 'category': 'Protein', 'cal/g': 1.65, 'protein(g)': 0.31, 'carb(g)': 0.00, 'fat(g)': 0.036},
            {'ingr_name': 'turkey breast', 'category': 'Protein', 'cal/g': 1.35, 'protein(g)': 0.30, 'carb(g)': 0.00, 'fat(g)': 0.02},
            {'ingr_name': 'lean beef', 'category': 'Protein', 'cal/g': 2.0, 'protein(g)': 0.26, 'carb(g)': 0.00, 'fat(g)': 0.10},
            {'ingr_name': 'salmon', 'category': 'Protein', 'cal/g': 2.08, 'protein(g)': 0.20, 'carb(g)': 0.00, 'fat(g)': 0.13},
            {'ingr_name': 'tuna', 'category': 'Protein', 'cal/g': 1.3, 'protein(g)': 0.29, 'carb(g)': 0.00, 'fat(g)': 0.01},
            {'ingr_name': 'tilapia', 'category': 'Protein', 'cal/g': 1.28, 'protein(g)': 0.26, 'carb(g)': 0.00, 'fat(g)': 0.027},
            {'ingr_name': 'cod', 'category': 'Protein', 'cal/g': 0.85, 'protein(g)': 0.19, 'carb(g)': 0.00, 'fat(g)': 0.005},
            {'ingr_name': 'eggs', 'category': 'Protein', 'cal/g': 1.55, 'protein(g)': 0.125, 'carb(g)': 0.012, 'fat(g)': 0.105},
            {'ingr_name': 'egg whites', 'category': 'Protein', 'cal/g': 0.52, 'protein(g)': 0.11, 'carb(g)': 0.007, 'fat(g)': 0.002},
            {'ingr_name': 'greek yogurt', 'category': 'Protein', 'cal/g': 0.59, 'protein(g)': 0.10, 'carb(g)': 0.036, 'fat(g)': 0.004},
            {'ingr_name': 'cottage cheese', 'category': 'Protein', 'cal/g': 0.98, 'protein(g)': 0.11, 'carb(g)': 0.034, 'fat(g)': 0.043},
            {'ingr_name': 'tofu', 'category': 'Protein', 'cal/g': 0.76, 'protein(g)': 0.08, 'carb(g)': 0.019, 'fat(g)': 0.048},

            # Carbs
            {'ingr_name': 'brown rice', 'category': 'Carb', 'cal/g': 1.11, 'protein(g)': 0.026, 'carb(g)': 0.23, 'fat(g)': 0.009},
            {'ingr_name': 'quinoa', 'category': 'Carb', 'cal/g': 1.20, 'protein(g)': 0.044, 'carb(g)': 0.21, 'fat(g)': 0.019},
            {'ingr_name': 'oats', 'category': 'Carb', 'cal/g': 3.89, 'protein(g)': 0.169, 'carb(g)': 0.66, 'fat(g)': 0.069},
            {'ingr_name': 'whole wheat bread', 'category': 'Carb', 'cal/g': 2.47, 'protein(g)': 0.13, 'carb(g)': 0.42, 'fat(g)': 0.033},
            {'ingr_name': 'sweet potato', 'category': 'Carb', 'cal/g': 0.86, 'protein(g)': 0.016, 'carb(g)': 0.20, 'fat(g)': 0.001},
            {'ingr_name': 'potato', 'category': 'Carb', 'cal/g': 0.77, 'protein(g)': 0.02, 'carb(g)': 0.17, 'fat(g)': 0.001},
            {'ingr_name': 'whole wheat pasta', 'category': 'Carb', 'cal/g': 1.24, 'protein(g)': 0.05, 'carb(g)': 0.25, 'fat(g)': 0.01},
            {'ingr_name': 'lentils', 'category': 'Carb', 'cal/g': 1.16, 'protein(g)': 0.09, 'carb(g)': 0.20, 'fat(g)': 0.004},
            {'ingr_name': 'chickpeas', 'category': 'Carb', 'cal/g': 1.39, 'protein(g)': 0.08, 'carb(g)': 0.23, 'fat(g)': 0.026},
            {'ingr_name': 'black beans', 'category': 'Carb', 'cal/g': 1.32, 'protein(g)': 0.09, 'carb(g)': 0.24, 'fat(g)': 0.005},

             #fruite
            {'ingr_name': 'banana', 'category': 'Carb', 'cal/g': 0.89, 'protein(g)': 0.011, 'carb(g)': 0.23, 'fat(g)': 0.003},
            {'ingr_name': 'apple', 'category': 'Carb', 'cal/g': 0.52, 'protein(g)': 0.003, 'carb(g)': 0.14, 'fat(g)': 0.002},
            {'ingr_name': 'berries', 'category': 'Carb', 'cal/g': 0.57, 'protein(g)': 0.007, 'carb(g)': 0.14, 'fat(g)': 0.003},
            {'ingr_name': 'orange', 'category': 'Carb', 'cal/g': 0.47, 'protein(g)': 0.009, 'carb(g)': 0.12, 'fat(g)': 0.001},


            # Vegetables
            {'ingr_name': 'broccoli', 'category': 'Vegetable', 'cal/g': 0.35, 'protein(g)': 0.024, 'carb(g)': 0.07, 'fat(g)': 0.004},
            {'ingr_name': 'spinach', 'category': 'Vegetable', 'cal/g': 0.23, 'protein(g)': 0.029, 'carb(g)': 0.036, 'fat(g)': 0.004},
            {'ingr_name': 'kale', 'category': 'Vegetable', 'cal/g': 0.49, 'protein(g)': 0.043, 'carb(g)': 0.09, 'fat(g)': 0.007},
            {'ingr_name': 'asparagus', 'category': 'Vegetable', 'cal/g': 0.20, 'protein(g)': 0.022, 'carb(g)': 0.039, 'fat(g)': 0.001},
            {'ingr_name': 'cauliflower', 'category': 'Vegetable', 'cal/g': 0.25, 'protein(g)': 0.019, 'carb(g)': 0.05, 'fat(g)': 0.003},
            {'ingr_name': 'zucchini', 'category': 'Vegetable', 'cal/g': 0.17, 'protein(g)': 0.012, 'carb(g)': 0.031, 'fat(g)': 0.004},
            {'ingr_name': 'bell peppers', 'category': 'Vegetable', 'cal/g': 0.26, 'protein(g)': 0.01, 'carb(g)': 0.06, 'fat(g)': 0.003},
            {'ingr_name': 'tomatoes', 'category': 'Vegetable', 'cal/g': 0.18, 'protein(g)': 0.009, 'carb(g)': 0.039, 'fat(g)': 0.002},
            {'ingr_name': 'cucumber', 'category': 'Vegetable', 'cal/g': 0.15, 'protein(g)': 0.007, 'carb(g)': 0.036, 'fat(g)': 0.001},
            {'ingr_name': 'carrots', 'category': 'Vegetable', 'cal/g': 0.41, 'protein(g)': 0.009, 'carb(g)': 0.10, 'fat(g)': 0.002},

            # Fats
            {'ingr_name': 'avocado', 'category': 'Fat', 'cal/g': 1.60, 'protein(g)': 0.02, 'carb(g)': 0.09, 'fat(g)': 0.15},
            {'ingr_name': 'olive oil', 'category': 'Fat', 'cal/g': 8.84, 'protein(g)': 0.00, 'carb(g)': 0.00, 'fat(g)': 1.00},
            {'ingr_name': 'almonds', 'category': 'Fat', 'cal/g': 5.78, 'protein(g)': 0.21, 'carb(g)': 0.22, 'fat(g)': 0.50},
            {'ingr_name': 'walnuts', 'category': 'Fat', 'cal/g': 6.54, 'protein(g)': 0.15, 'carb(g)': 0.14, 'fat(g)': 0.65},
            

            # drinks
            {'ingr_name': 'water', 'category': 'Drink', 'cal/g': 0.0, 'protein(g)': 0.0, 'carb(g)': 0.0, 'fat(g)': 0.0},
            {'ingr_name': 'green tea', 'category': 'Drink', 'cal/g': 0.01, 'protein(g)': 0.0, 'carb(g)': 0.0, 'fat(g)': 0.0},
 
  ])
        self.protein_foods = self.food_db[self.food_db['category'] == 'Protein']
        self.carb_foods = self.food_db[self.food_db['category'] == 'Carb']
        self.veg_foods = self.food_db[self.food_db['category'] == 'Vegetable']
        self.fat_foods = self.food_db[self.food_db['category'] == 'Fat']
        self.drink_foods = self.food_db[self.food_db['category'] == 'Drink']
        self.balanced_foods = self.food_db

    def prepare_features(self, user_data):
        """Prepare input features for the Neural Network model"""
        
        # Start with empty dictionary
        features = {}
        
        # Initialize all feature columns with 0
        for col in self.feature_columns:
            features[col] = 0
        
        # Map user data to feature columns
        # Base numeric features
        numeric_features = ['age', 'BMI', 'diabetes', 'hypertension', 
                           'kidney_disease', 'heart_disease', 'pcos', 'anemia', 'gout']
        for col in numeric_features:
            if col in self.feature_columns:
                features[col] = user_data.get(col, 0)
        
        # Gender mapping
        if 'gender' in self.feature_columns:
            features['gender'] = 0 if user_data.get('gender', 'Male') == 'Male' else 1
        
        # Athlete mapping
        if 'athlete' in self.feature_columns:
            sport = user_data.get('sport_type', 'None')
            features['athlete'] = 0 if sport in ['None', None, ''] else 1
        
        # Goal one-hot encoding
        goal = user_data.get('goal', 'Maintenance')
        goal_col = f'goal_{goal}'
        if goal_col in self.feature_columns:
            features[goal_col] = 1
        
        # Activity level one-hot encoding
        activity = user_data.get('activity_level', 'Moderate')
        activity_col = f'act_{activity}'
        if activity_col in self.feature_columns:
            features[activity_col] = 1
        
        # Surgery one-hot encoding
        surgery = user_data.get('surgery_type', 'None')
        surgery_col = f'surg_{surgery}'
        if surgery_col in self.feature_columns:
            features[surgery_col] = 1
        
        # Convert to DataFrame with correct column order
        input_df = pd.DataFrame([features])[self.feature_columns]
        
        return input_df

    # removed NN-only predict_macros here; combined implementation below uses NN when available

    def categorize_foods_correctly(self):
        # نفس الكود الأصلي من الـ notebook
        self.food_db['protein_per_100g'] = self.food_db['protein(g)'] * 100
        self.food_db['carb_per_100g'] = self.food_db['carb(g)'] * 100
        self.food_db['fat_per_100g'] = self.food_db['fat(g)'] * 100

        self.food_db['cal_from_protein'] = self.food_db['protein_per_100g'] * 4
        self.food_db['cal_from_carb'] = self.food_db['carb_per_100g'] * 4
        self.food_db['cal_from_fat'] = self.food_db['fat_per_100g'] * 9
        self.food_db['total_calc_cal'] = self.food_db['cal_from_protein'] + self.food_db['cal_from_carb'] + self.food_db['cal_from_fat']
        self.food_db['total_calc_cal'] = self.food_db['total_calc_cal'].replace(0, 1)

        self.food_db['protein_pct'] = (self.food_db['cal_from_protein'] / self.food_db['total_calc_cal'] * 100).round(1)
        self.food_db['carb_pct'] = (self.food_db['cal_from_carb'] / self.food_db['total_calc_cal'] * 100).round(1)
        self.food_db['fat_pct'] = (self.food_db['cal_from_fat'] / self.food_db['total_calc_cal'] * 100).round(1)

        conditions = [
            (self.food_db['protein_pct'] >= 30),
            (self.food_db['carb_pct'] >= 40),
            (self.food_db['fat_pct'] >= 35),
        ]
        choices = ['High_Protein', 'High_Carb', 'High_Fat']
        self.food_db['macro_type'] = np.select(conditions, choices, default='Balanced')

        self.protein_foods = self.food_db[self.food_db['macro_type'] == 'High_Protein']
        self.carb_foods = self.food_db[self.food_db['macro_type'] == 'High_Carb']
        self.fat_foods = self.food_db[self.food_db['macro_type'] == 'High_Fat']
        self.balanced_foods = self.food_db[self.food_db['macro_type'] == 'Balanced']
        self.veg_foods = self.food_db[self.food_db['category'] == 'Vegetable']
                

    def weighted_macros(self, row: Dict):
        """نفس دالة weighted_macros الموجودة في الـ Notebook"""
        macros_dict = {
            "Bariatric": {"Protein":35,"Carb":40,"Fat":25},
            "Gallbladder": {"Protein":25,"Carb":50,"Fat":25},
            "Kidney": {"Protein":16,"Carb":53,"Fat":31},
            "Diabetes": {"Protein":25,"Carb":42,"Fat":33},
            "Hypertension": {"Protein":22,"Carb":53,"Fat":25},
            "Heart_Disease": {"Protein":20,"Carb":50,"Fat":30},
            "PCOS": {"Protein":20,"Carb":45,"Fat":35},
            "Anemia": {"Protein":22,"Carb":50,"Fat":28},
            "Gout": {"Protein":18,"Carb":55,"Fat":27},
            "Football": {"Protein":25,"Carb":55,"Fat":20},
            "Bodybuilding": {"Protein":30,"Carb":45,"Fat":25},
            "Running": {"Protein":22,"Carb":58,"Fat":20},
            "Yoga": {"Protein":20,"Carb":50,"Fat":30},
            "Default": {"Protein":20,"Carb":50,"Fat":30},
        }

        weights_dict = {
            "Bariatric": random.uniform(0.8, 1.0),
            "Gallbladder": random.uniform(0.8, 1.0),
            "Kidney": random.uniform(0.8, 1.0),
            "Diabetes": random.uniform(0.8, 1.0),
            "Hypertension": random.uniform(0.4, 1.0),
            "Heart_Disease": random.uniform(0.4, 1.0),
            "PCOS": random.uniform(0.4, 1.0),
            "Anemia": random.uniform(0.4, 1.0),
            "Gout": random.uniform(0.4, 1.0),
            "Football": random.uniform(0.6, 1.0),
            "Bodybuilding": random.uniform(0.8, 1.0),
            "Running": random.uniform(0.6, 1.0),
            "Yoga": random.uniform(0.6, 1.0),
            "Default": random.uniform(0.3, 1.0)
        }

        active_macros = []
        active_weights = []
        active_conditions = []

        # Surgery Type(s)
        surgeries = row.get('surgery_type') or []
        if isinstance(surgeries, str):
            surgeries = [surgeries]
        for surgery in surgeries:
            if surgery == "Bariatric":
                active_macros.append(macros_dict["Bariatric"])
                active_weights.append(weights_dict["Bariatric"])
                active_conditions.append("Bariatric")
            elif surgery == "Gallbladder":
                active_macros.append(macros_dict["Gallbladder"])
                active_weights.append(weights_dict["Gallbladder"])
                active_conditions.append("Gallbladder")

        # Medical Conditions
        medical_map = {
            'diabetes': "Diabetes",
            'hypertension': "Hypertension",
            'kidney_disease': "Kidney",
            'heart_disease': "Heart_Disease",
            'pcos': "PCOS",
            'anemia': "Anemia",
            'gout': "Gout"
        }

        for col, condition in medical_map.items():
            if row.get(col) == 1:
                active_macros.append(macros_dict[condition])
                active_weights.append(weights_dict[condition])
                active_conditions.append(condition)

        # Sport Type
        sport = row.get('sport_type')
        if sport in macros_dict:
            active_macros.append(macros_dict[sport])
            active_weights.append(weights_dict[sport])
            active_conditions.append(sport)

        # Default Case
        if len(active_macros) == 0:
            active_macros.append(macros_dict["Default"])
            active_weights.append(weights_dict["Default"])
            active_conditions.append("Default")

        # Weighted Average
        total_weight = sum(active_weights)
        protein = sum(m["Protein"] * w for m, w in zip(active_macros, active_weights)) / total_weight
        carb = sum(m["Carb"] * w for m, w in zip(active_macros, active_weights)) / total_weight
        fat = sum(m["Fat"] * w for m, w in zip(active_macros, active_weights)) / total_weight

        # Goal Adjustment
        goal = row.get('goal', 'Maintenance')
        if goal == "Fat_Loss":
            protein += 5
            carb -= 5
        elif goal == "Muscle_Gain":
            protein += 5
            carb += 5
            fat -= 5

        # Normalize to 100%
        total = protein + carb + fat
        if total != 100:
            protein = round(protein / total * 100, 1)
            carb = round(carb / total * 100, 1)
            fat = round(fat / total * 100, 1)

        return {'protein': protein, 'carbs': carb, 'fat': fat}

    def predict_macros(self, user_data):
        """Predict macros: prefer NN when available, otherwise fallback to rule-based averages."""
        # Try NN path if model, scaler and feature_columns are present
        if self.model is not None and self.scaler is not None and self.feature_columns is not None:
            try:
                input_df = self.prepare_features(user_data)
                input_scaled = self.scaler.transform(input_df)
                prediction = self.model.predict(input_scaled, verbose=0)[0]
                protein_pct = prediction[0] * 100
                carbs_pct = prediction[1] * 100
                fat_pct = prediction[2] * 100
                total = protein_pct + carbs_pct + fat_pct
                if abs(total - 100) > 0.1 and total > 0:
                    protein_pct = (protein_pct / total) * 100
                    carbs_pct = (carbs_pct / total) * 100
                    fat_pct = (fat_pct / total) * 100
                return {'protein': round(protein_pct, 1), 'carbs': round(carbs_pct, 1), 'fat': round(fat_pct, 1)}
            except Exception as e:
                print(f"⚠️ NN prediction failed, falling back to rule-based: {e}")

        # Rule-based fallback (average of active conditions)
        active_conditions = []
        surgeries = user_data.get('surgery_type') or []
        if isinstance(surgeries, str):
            surgeries = [surgeries]
        if "Bariatric" in surgeries:
            active_conditions.append("Bariatric")
        if "Gallbladder" in surgeries:
            active_conditions.append("Gallbladder")
        if user_data.get('diabetes') == 1:
            active_conditions.append("Diabetes")
        if user_data.get('hypertension') == 1:
            active_conditions.append("Hypertension")
        if user_data.get('kidney_disease') == 1:
            active_conditions.append("Kidney")
        if user_data.get('heart_disease') == 1:
            active_conditions.append("Heart_Disease")
        if user_data.get('pcos') == 1:
            active_conditions.append("PCOS")
        if user_data.get('anemia') == 1:
            active_conditions.append("Anemia")
        if user_data.get('gout') == 1:
            active_conditions.append("Gout")
        if user_data.get('sport_type') in ["Football", "Bodybuilding", "Running", "Yoga"]:
            active_conditions.append(user_data.get('sport_type'))

        if not active_conditions:
            active_conditions = ["Default"]

        macros_dict = {
            "Bariatric": {"Protein":35,"Carb":40,"Fat":25},
            "Gallbladder": {"Protein":25,"Carb":50,"Fat":25},
            "Kidney": {"Protein":16,"Carb":53,"Fat":31},
            "Diabetes": {"Protein":25,"Carb":42,"Fat":33},
            "Hypertension": {"Protein":22,"Carb":53,"Fat":25},
            "Heart_Disease": {"Protein":20,"Carb":50,"Fat":30},
            "PCOS": {"Protein":20,"Carb":45,"Fat":35},
            "Anemia": {"Protein":22,"Carb":50,"Fat":28},
            "Gout": {"Protein":18,"Carb":55,"Fat":27},
            "Football": {"Protein":25,"Carb":55,"Fat":20},
            "Bodybuilding": {"Protein":30,"Carb":45,"Fat":25},
            "Running": {"Protein":22,"Carb":58,"Fat":20},
            "Yoga": {"Protein":20,"Carb":50,"Fat":30},
            "Default": {"Protein":20,"Carb":50,"Fat":30},
        }

        protein = sum(macros_dict[cond]["Protein"] for cond in active_conditions) / len(active_conditions)
        carb = sum(macros_dict[cond]["Carb"] for cond in active_conditions) / len(active_conditions)
        fat = sum(macros_dict[cond]["Fat"] for cond in active_conditions) / len(active_conditions)

        if user_data.get('goal') == "Fat_Loss":
            protein += 5
            carb -= 5
        elif user_data.get('goal') == "Muscle_Gain":
            protein += 5
            carb += 3
            fat -= 8

        total = protein + carb + fat
        protein = round(protein / total * 100, 1)
        carb = round(carb / total * 100, 1)
        fat = round(fat / total * 100, 1)

        return {'protein': protein, 'carbs': carb, 'fat': fat}

    def calculate_meal_needs(self, total_calories, macros_percent, num_meals=3):
        """Calculate each meal's requirements honoring number of meals.

        Returns a list of meal requirement dicts of length `num_meals`.
        """
        if num_meals <= 0:
            num_meals = 3

        # Simple even split across meals
        calories_per_meal = total_calories / num_meals

        meal_needs = []
        for i in range(num_meals):
            meal = {
                'calories': calories_per_meal,
                'protein_g': (calories_per_meal * (macros_percent['protein']/100)) / 4,
                'carbs_g': (calories_per_meal * (macros_percent['carbs']/100)) / 4,
                'fat_g': (calories_per_meal * (macros_percent['fat']/100)) / 9
            }
            meal_needs.append(meal)

        return meal_needs
    def recommend_meal_from_db(self, target_protein, target_carbs, target_fat,
                               target_calories, meal_type='main', user_data=None):
        """Compose a meal from the database"""

        meal_foods = []
        current_calories = 0
        current_protein = 0
        current_carbs = 0
        current_fat = 0

        # Filter food pools by allergies if provided
        def _filter_by_allergies(df, user_data):
            if not user_data:
                return df
            forbidden = []
            if user_data.get('nut_allergy') == 1:
                forbidden += ['almond', 'walnut', 'peanut', 'almonds', 'walnuts', 'peanuts']
            if user_data.get('dairy_allergy') == 1:
                forbidden += ['milk', 'yogurt', 'cheese', 'cottage', 'greek', 'butter']
            if user_data.get('gluten_intolerance') == 1:
                forbidden += ['bread', 'wheat', 'pasta']
            if user_data.get('shellfish_allergy') == 1:
                forbidden += ['shrimp', 'prawn', 'crab', 'lobster', 'shellfish']
            if user_data.get('egg_allergy') == 1:
                forbidden += ['egg', 'eggs']

            if not forbidden:
                return df

            mask = np.ones(len(df), dtype=bool)
            names = df['ingr_name'].str.lower()
            for kw in forbidden:
                mask = mask & (~names.str.contains(kw))
            return df[mask]

        protein_pool = _filter_by_allergies(self.protein_foods, user_data)
        carb_pool = _filter_by_allergies(self.carb_foods, user_data)
        veg_pool = _filter_by_allergies(self.veg_foods, user_data)
        fat_pool = _filter_by_allergies(self.fat_foods, user_data)

        # Select protein source
        if len(protein_pool) > 0:
            protein_source = protein_pool.sample(1).iloc[0]
            protein_amount = min(150, target_protein * 0.8)
            meal_foods.append({
                'name': protein_source['ingr_name'],
                'amount_g': round(protein_amount, 1),
                'calories': protein_amount * protein_source['cal/g'],
                'protein': protein_amount * protein_source['protein(g)'],
                'carbs': protein_amount * protein_source['carb(g)'],
                'fat': protein_amount * protein_source['fat(g)']
            })

            current_calories += meal_foods[-1]['calories']
            current_protein += meal_foods[-1]['protein']
            current_carbs += meal_foods[-1]['carbs']
            current_fat += meal_foods[-1]['fat']
        else:
            balanced = self.balanced_foods.sample(1).iloc[0]
            meal_foods.append({
                'name': balanced['ingr_name'],
                'amount_g': 150,
                'calories': 150 * balanced['cal/g'],
                'protein': 150 * balanced['protein(g)'],
                'carbs': 150 * balanced['carb(g)'],
                'fat': 150 * balanced['fat(g)']
            })
            current_calories += meal_foods[-1]['calories']
            current_protein += meal_foods[-1]['protein']
            current_carbs += meal_foods[-1]['carbs']
            current_fat += meal_foods[-1]['fat']

        # Select carb source
        if current_carbs < target_carbs * 0.5 and len(carb_pool) > 0:
            carb_source = carb_pool.sample(1).iloc[0]
            remaining_carbs = target_carbs - current_carbs
            carb_amount = min(100, remaining_carbs * 1.2)
            meal_foods.append({
                'name': carb_source['ingr_name'],
                'amount_g': round(carb_amount, 1),
                'calories': carb_amount * carb_source['cal/g'],
                'protein': carb_amount * carb_source['protein(g)'],
                'carbs': carb_amount * carb_source['carb(g)'],
                'fat': carb_amount * carb_source['fat(g)']
            })

            current_calories += meal_foods[-1]['calories']
            current_protein += meal_foods[-1]['protein']
            current_carbs += meal_foods[-1]['carbs']
            current_fat += meal_foods[-1]['fat']

        # Select vegetables
        if len(veg_pool) > 0:
            veg = veg_pool.sample(1).iloc[0]
            veg_amount = 100
            meal_foods.append({
                'name': veg['ingr_name'],
                'amount_g': veg_amount,
                'calories': veg_amount * veg['cal/g'],
                'protein': veg_amount * veg['protein(g)'],
                'carbs': veg_amount * veg['carb(g)'],
                'fat': veg_amount * veg['fat(g)']
            })

            current_calories += meal_foods[-1]['calories']
            current_protein += meal_foods[-1]['protein']
            current_carbs += meal_foods[-1]['carbs']
            current_fat += meal_foods[-1]['fat']

        return {
            'foods': meal_foods,
            'totals': {
                'calories': round(current_calories, 1),
                'protein': round(current_protein, 1),
                'carbs': round(current_carbs, 1),
                'fat': round(current_fat, 1)
            },
            'targets': {
                'calories': target_calories,
                'protein': target_protein,
                'carbs': target_carbs,
                'fat': target_fat
            }
        }
    def generate_daily_plan(self, user_data, total_calories):
        """Generate complete daily plan"""
        macros = self.predict_macros(user_data)
        meals_per_day = int(user_data.get('meals_per_day', 3) or 3)
        # ensure reasonable bounds
        meals_per_day = max(1, min(meals_per_day, 6))

        meal_needs = self.calculate_meal_needs(total_calories, macros, num_meals=meals_per_day)

        plan = {
            'user_info': user_data,
            'daily_macros': macros,
            'meals': [],
            'hydration': {
                'recommended_liters': float(user_data.get('water_intake_liters') or 0),
                'tip': 'Spread intake evenly across the day.'
            }
        }

        # meal name patterns depending on number of meals
        name_map = {
            1: ['Meal'],
            2: ['Breakfast', 'Dinner'],
            3: ['Breakfast', 'Lunch', 'Dinner'],
            4: ['Breakfast', 'Snack', 'Lunch', 'Dinner'],
            5: ['Breakfast', 'Snack', 'Lunch', 'Snack', 'Dinner'],
            6: ['Breakfast', 'Snack', 'Lunch', 'Snack', 'Dinner', 'Snack']
        }

        meal_types = name_map.get(meals_per_day, name_map[3])

        for i, needs in enumerate(meal_needs):
            meal_label = meal_types[i] if i < len(meal_types) else f'Meal {i+1}'
            meal = self.recommend_meal_from_db(
                needs['protein_g'],
                needs['carbs_g'],
                needs['fat_g'],
                needs['calories'],
                meal_label,
                user_data
            )
            plan['meals'].append({
                'name': meal_label,
                'foods': meal['foods'],
                'totals': meal['totals']
            })

        # Add hydration tip for output
        if plan['hydration']['recommended_liters'] and plan['hydration']['recommended_liters'] > 0:
            plan.setdefault('notes', []).append(f"Aim for ~{plan['hydration']['recommended_liters']}L water/day")

        return plan

    def generate_weekly_plan(self, user_data, total_calories):
        macros = self.predict_macros(user_data)
        daily_meal_needs = self.calculate_meal_needs(total_calories, macros, num_meals=3)
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        meal_types = ['Breakfast', 'Lunch', 'Dinner']
        weekly_plan = {
            'daily_macros': macros,
            'days': []
            }
        for day in days:
            day_meals = []
            for i, needs in enumerate(daily_meal_needs):
                random.seed(hash(day + meal_types[i]) % 2**32)  # ✅ تنويع الوجبات
                meal = self.recommend_meal_from_db(
                    needs['protein_g'],
                    needs['carbs_g'],
                    needs['fat_g'],
                    needs['calories'],
                    meal_types[i],
                    user_data
                )
                day_meals.append({          # ✅ جوا الloop
                    'name': meal_types[i],
                    'foods': meal['foods'],
                    'totals': meal['totals']
                    })
            weekly_plan['days'].append({
                'day': day,
                'meals': day_meals
            })

        return weekly_plan

def print_weekly_plan(self, weekly_plan):
        """Print formatted weekly plan"""

        print("\n" + "="*80)
        print("📅 Your Weekly Meal Plan:")
        print("="*80)

        m = weekly_plan['daily_macros']
        print(f"\n📊 Daily macro percentages: Protein {m['protein']:.1f}% | Carbs {m['carbs']:.1f}% | Fat {m['fat']:.1f}%")

        # Calculate approximate daily calories
        if weekly_plan['days'][0]['meals']:
            daily_cals = sum(meal['totals']['calories'] for meal in weekly_plan['days'][0]['meals'])
            print(f"🔥 Target daily calories: ~{daily_cals:.0f} calories")

        for day_data in weekly_plan['days']:
            print(f"\n{'='*60}")
            print(f"📆 {day_data['day']}:")
            print(f"{'='*60}")

            total_day_calories = 0
            total_day_protein = 0
            total_day_carbs = 0
            total_day_fat = 0

            for meal in day_data['meals']:
                print(f"\n  🍽️ {meal['name']}:")
                for food in meal['foods']:
                    print(f"      • {food['name']}: {food['amount_g']:.0f} grams")
                print(f"      📊 Calories: {meal['totals']['calories']:.0f}")
                print(f"      💪 Protein: {meal['totals']['protein']:.0f}g")
                print(f"      🍚 Carbs: {meal['totals']['carbs']:.0f}g")
                print(f"      🥑 Fat: {meal['totals']['fat']:.0f}g")

                total_day_calories += meal['totals']['calories']
                total_day_protein += meal['totals']['protein']
                total_day_carbs += meal['totals']['carbs']
                total_day_fat += meal['totals']['fat']

            print(f"\n  📊 Daily total: {total_day_calories:.0f} calories | Protein {total_day_protein:.0f}g | Carbs {total_day_carbs:.0f}g | Fat {total_day_fat:.0f}g")

        print("\n" + "="*80)
        print("✅ Healthy and varied week! Different meals every day 🥗")
        print("="*80)
    # باقي الدوال (generate_daily_plan, recommend_meal_from_db, ...) يمكن إضافتها لاحقاً




class ExercisePlanner:
    def __init__(self):
        # Define exercises
        self.exercise_rules = {
            "gym": {
                "bench_press": {"joints": ["elbow_left", "elbow_right"], "up_threshold": 150, "down_threshold": 80, "min_angle": 70, "max_angle": 160},
                "shoulder_press": {"joints": ["elbow_left", "elbow_right"], "up_threshold": 160, "down_threshold": 90, "min_angle": 80, "max_angle": 170},
                "lat_pulldown": {"joints": ["elbow_left", "elbow_right"], "up_threshold": 150, "down_threshold": 60, "min_angle": 45, "max_angle": 160},
                "bicep_curl": {"joints": ["elbow_left", "elbow_right"], "up_threshold": 40, "down_threshold": 150, "min_angle": 30, "max_angle": 160},
                "triceps_pushdown": {"joints": ["elbow_left", "elbow_right"], "up_threshold": 160, "down_threshold": 40, "min_angle": 30, "max_angle": 170},
                "leg_extension": {"joints": ["knee_left", "knee_right"], "up_threshold": 160, "down_threshold": 100, "min_angle": 90, "max_angle": 175}
            },
            "rehab": {
                "arm_abduction": {"joints": ["shoulder_abduction_left", "shoulder_abduction_right"], "up_threshold": 80, "down_threshold": 20, "min_angle": 15, "max_angle": 95},
                "shoulder_flexion": {"joints": ["shoulder_flexion_left", "shoulder_flexion_right"], "up_threshold": 90, "down_threshold": 10, "min_angle": 0, "max_angle": 180},
                "arm_vw": {"joints": ["elbow_left", "elbow_right"], "up_threshold": 120, "down_threshold": 60, "min_angle": 45, "max_angle": 135},
                "pushups": {"joints": ["elbow_left", "elbow_right"], "up_threshold": 160, "down_threshold": 80, "min_angle": 70, "max_angle": 170},
                "leg_abduction": {"joints": ["hip_left", "hip_right"], "up_threshold": 40, "down_threshold": 10, "min_angle": 0, "max_angle": 45},
                "squats": {"joints": ["knee_left", "knee_right"], "up_threshold": 160, "down_threshold": 80, "min_angle": 70, "max_angle": 170},
                "leg_lunge": {"joints": ["knee_left", "knee_right"], "up_threshold": 160, "down_threshold": 95, "min_angle": 85, "max_angle": 170}
            },
            "fitness": {
                "bodyweight_squats": {"joints": ["knee_left", "knee_right"], "up_threshold": 120, "down_threshold": 60, "min_angle": 40, "max_angle": 130},
                "jumping_jacks": {"joints": ["shoulder_abduction_left", "shoulder_abduction_right"], "up_threshold": 140, "down_threshold": 30, "min_angle": 20, "max_angle": 150},
                "lunge": {"joints": ["knee_left", "knee_right"], "up_threshold": 160, "down_threshold": 95, "min_angle": 85, "max_angle": 170},
                "leg_swing": {"joints": ["hip_left", "hip_right"], "up_threshold": 80, "down_threshold": 10, "min_angle": 0, "max_angle": 90},
                "butt_kicks": {"joints": ["knee_left", "knee_right"], "up_threshold": 60, "down_threshold": 150, "min_angle": 45, "max_angle": 160},
                "high_knee": {"joints": ["hip_left", "hip_right"], "up_threshold": 80, "down_threshold": 160, "min_angle": 70, "max_angle": 170},
                "arm_circles": {
                    "joints": ["shoulder_left", "shoulder_right"],
                    "up_threshold": 90,
                    "down_threshold": 180,
                    "min_angle": 30,
                    "max_angle": 180,
                    "special_function": "arm_circle"
                },
                "arm_half_circles": {
                    "joints": ["shoulder_left", "shoulder_right"],
                    "up_threshold": 80,
                    "down_threshold": 120,
                    "min_angle": 30,
                    "max_angle": 120,
                    "special_function": "arm_circle"
                }
            }
        }

        # Classify exercises
        self.exercise_categories = self.categorize_exercises()

    def categorize_exercises(self):
        """Classify exercises by type and target muscles"""
        categories = {
            'Upper_Body': [],
            'Lower_Body': [],
            'Core': [],
            'Cardio': [],
            'Full_Body': []
        }

        for category, exercises in self.exercise_rules.items():
            for ex_name, ex_data in exercises.items():
                # Determine exercise type based on joints
                joints = ex_data['joints']
                if any('knee' in j or 'hip' in j for j in joints):
                    if any('shoulder' in j or 'elbow' in j for j in joints):
                        categories['Full_Body'].append({
                            'name': ex_name,
                            'category': category,
                            'rules': ex_data
                        })
                    else:
                        categories['Lower_Body'].append({
                            'name': ex_name,
                            'category': category,
                            'rules': ex_data
                        })
                elif any('shoulder' in j or 'elbow' in j for j in joints):
                    categories['Upper_Body'].append({
                        'name': ex_name,
                        'category': category,
                        'rules': ex_data
                    })
                else:
                    categories['Cardio'].append({
                        'name': ex_name,
                        'category': category,
                        'rules': ex_data
                    })

        return categories

    def select_exercises_by_goal(self, goal: str, available_days: int = 4, injuries: Dict = None) -> Dict:
        """Select appropriate exercises based on goal considering injuries"""

        exercise_plan = {
            'weekly_schedule': [],
            'exercise_details': {},
            'total_weekly_volume': 0,
            'tips': []
        }

        # Determine exercise distribution based on goal
        if goal == 'Muscle_Gain':
            distribution = {
                'Upper_Body': 0.4,
                'Lower_Body': 0.3,
                'Full_Body': 0.2,
                'Core': 0.1
            }
            sets_per_exercise = 4
            reps_range = "8-12"
            rest_time = "90 sec"
            exercise_plan['tips'] = [
                'Focus on gradually increasing weights',
                'Take adequate rest between sets',
                'Eat enough protein within 2 hours of workout'
            ]

        elif goal == 'Fat_Loss':
            distribution = {
                'Cardio': 0.4,
                'Full_Body': 0.3,
                'Upper_Body': 0.15,
                'Lower_Body': 0.15
            }
            sets_per_exercise = 3
            reps_range = "15-20"
            rest_time = "45 sec"
            exercise_plan['tips'] = [
                'Maintain high heart rate',
                'Reduce rest periods between exercises',
                'Drink enough water before and during workout'
            ]

        else:  # Maintenance
            distribution = {
                'Full_Body': 0.3,
                'Upper_Body': 0.25,
                'Lower_Body': 0.25,
                'Core': 0.1,
                'Cardio': 0.1
            }
            sets_per_exercise = 3
            reps_range = "12-15"
            rest_time = "60 sec"
            exercise_plan['tips'] = [
                'Mix between strength and cardio exercises',
                'Listen to your body and rest when needed',
                'Maintain consistent performance'
            ]

        # Add injury tips
        if injuries:
            if injuries.get('back_pain', 0) == 1:
                exercise_plan['tips'].append('Avoid exercises that strain the back')
                # Reduce Lower_Body exercises
                distribution['Lower_Body'] = distribution.get('Lower_Body', 0) * 0.5
                distribution['Upper_Body'] = distribution.get('Upper_Body', 0) + 0.1
            if injuries.get('ankle_injury', 0) == 1:
                exercise_plan['tips'].append('Use low-impact exercises for ankles')
                # Reduce high-impact cardio
                distribution['Cardio'] = distribution.get('Cardio', 0) * 0.7

        # Select exercises for each day
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

        selected_days = random.sample(days, min(available_days, len(days)))
        selected_days.sort(key=lambda x: days.index(x))

        for day in selected_days:
            day_exercises = []

            # Select 4-6 exercises per day
            num_exercises = min(random.randint(4, 6), available_days * 2)

            for _ in range(num_exercises):
                # Select exercise type based on distribution
                exercise_type = random.choices(
                    list(distribution.keys()),
                    weights=list(distribution.values())
                )[0]

                # Select specific exercise from category
                if self.exercise_categories[exercise_type]:
                    exercise = random.choice(self.exercise_categories[exercise_type])

                    # Avoid repeating the same exercise on the same day
                    if any(ex['name'] == exercise['name'] for ex in day_exercises):
                        continue

                    # Add exercise details
                    exercise_detail = {
                        'name': exercise['name'].replace('_', ' ').title(),
                        'type': exercise_type,
                        'category': exercise['category'],
                        'sets': sets_per_exercise,
                        'reps': reps_range,
                        'rest': rest_time,
                        'joints': exercise['rules']['joints']
                    }

                    day_exercises.append(exercise_detail)

                    # Store complete details
                    if exercise['name'] not in exercise_plan['exercise_details']:
                        exercise_plan['exercise_details'][exercise['name']] = {
                            'rules': exercise['rules'],
                            'total_volume_sets': 0
                        }
                    exercise_plan['exercise_details'][exercise['name']]['total_volume_sets'] += sets_per_exercise

            exercise_plan['weekly_schedule'].append({
                'day': day,
                'exercises': day_exercises[:min(6, len(day_exercises))],  # Maximum 6 exercises
                'total_exercises': min(6, len(day_exercises))
            })

        # Calculate total volume
        exercise_plan['total_weekly_volume'] = sum(
            len(day['exercises']) for day in exercise_plan['weekly_schedule']
        )

        return exercise_plan

    def print_workout_plan(self, plan: Dict):
        """Print formatted workout plan"""
        print("\n" + "="*70)
        print("💪 Weekly Workout Plan:")
        print("="*70)

        for workout in plan['weekly_schedule']:
            print(f"\n📅 {workout['day']}:")
            print("-" * 40)
            for ex in workout['exercises']:
                print(f"  • {ex['name']}")
                print(f"    {ex['sets']} sets × {ex['reps']} (rest {ex['rest']})")
                print(f"    Joints: {', '.join(set([j.replace('_', ' ').title() for j in ex['joints']]))}")

        print("\n💡 Exercise Tips:")
        for tip in plan['tips']:
            print(f"  • {tip}")




























# # app/services/advanced_nutrition_system.py
# import pandas as pd
# import numpy as np
# import joblib
# import math
# import random
# import tensorflow as tf
# from pathlib import Path
# from typing import Dict, List

# BASE_DIR = Path(__file__).resolve().parent.parent
# ML_DIR = BASE_DIR / "infrastructure" / "ml_models"


# class AdvancedNutritionSystem:
#     def __init__(self):
#         self.model = None
#         self.scaler = None
#         self.feature_columns = None

#         # تحميل Neural Network Model (نفس ما في الـ Notebook)
#         try:
#             self.model = tf.keras.models.load_model(ML_DIR / "nutrition_model.h5", compile=False)
#             print("✅ Neural Network Model loaded successfully (from Notebook)")
#         except Exception as e:
#             print(f"⚠️ Neural Network loading failed: {e}")
#             print("→ Using rule-based weighted macros as fallback")

#         # تحميل Scaler و Feature Columns
#         try:
#             self.scaler = joblib.load(ML_DIR / "scaler.pkl")
#             self.feature_columns = joblib.load(ML_DIR / "feature_columns.pkl")
#         except:
#             print("⚠️ Scaler or feature columns not found")

#         self.create_healthy_food_database()

#     # ====================== MACRO CALCULATION (مطابق للـ Notebook) ======================
#     def weighted_macros(self, row: Dict):
#         """نفس دالة weighted_macros الموجودة في الـ Notebook"""
#         macros_dict = {
#             "Bariatric": {"Protein":35,"Carb":40,"Fat":25},
#             "Gallbladder": {"Protein":25,"Carb":50,"Fat":25},
#             "Kidney": {"Protein":16,"Carb":53,"Fat":31},
#             "Diabetes": {"Protein":25,"Carb":42,"Fat":33},
#             "Hypertension": {"Protein":22,"Carb":53,"Fat":25},
#             "Heart_Disease": {"Protein":20,"Carb":50,"Fat":30},
#             "PCOS": {"Protein":20,"Carb":45,"Fat":35},
#             "Anemia": {"Protein":22,"Carb":50,"Fat":28},
#             "Gout": {"Protein":18,"Carb":55,"Fat":27},
#             "Football": {"Protein":25,"Carb":55,"Fat":20},
#             "Bodybuilding": {"Protein":30,"Carb":45,"Fat":25},
#             "Running": {"Protein":22,"Carb":58,"Fat":20},
#             "Yoga": {"Protein":20,"Carb":50,"Fat":30},
#             "Default": {"Protein":20,"Carb":50,"Fat":30},
#         }

#         weights_dict = {
#             "Bariatric": random.uniform(0.8, 1.0),
#             "Gallbladder": random.uniform(0.8, 1.0),
#             "Kidney": random.uniform(0.8, 1.0),
#             "Diabetes": random.uniform(0.8, 1.0),
#             "Hypertension": random.uniform(0.4, 1.0),
#             "Heart_Disease": random.uniform(0.4, 1.0),
#             "PCOS": random.uniform(0.4, 1.0),
#             "Anemia": random.uniform(0.4, 1.0),
#             "Gout": random.uniform(0.4, 1.0),
#             "Football": random.uniform(0.6, 1.0),
#             "Bodybuilding": random.uniform(0.8, 1.0),
#             "Running": random.uniform(0.6, 1.0),
#             "Yoga": random.uniform(0.6, 1.0),
#             "Default": random.uniform(0.3, 1.0)
#         }

#         active_macros = []
#         active_weights = []
#         active_conditions = []

#         # Surgery Type
#         surgery = row.get('surgery_type')
#         if surgery == "Bariatric":
#             active_macros.append(macros_dict["Bariatric"])
#             active_weights.append(weights_dict["Bariatric"])
#             active_conditions.append("Bariatric")
#         elif surgery == "Gallbladder":
#             active_macros.append(macros_dict["Gallbladder"])
#             active_weights.append(weights_dict["Gallbladder"])
#             active_conditions.append("Gallbladder")

#         # Medical Conditions
#         medical_map = {
#             'diabetes': "Diabetes",
#             'hypertension': "Hypertension",
#             'kidney_disease': "Kidney",
#             'heart_disease': "Heart_Disease",
#             'pcos': "PCOS",
#             'anemia': "Anemia",
#             'gout': "Gout"
#         }

#         for col, condition in medical_map.items():
#             if row.get(col) == 1:
#                 active_macros.append(macros_dict[condition])
#                 active_weights.append(weights_dict[condition])
#                 active_conditions.append(condition)

#         # Sport Type
#         sport = row.get('sport_type')
#         if sport in macros_dict:
#             active_macros.append(macros_dict[sport])
#             active_weights.append(weights_dict[sport])
#             active_conditions.append(sport)

#         # Default Case
#         if len(active_macros) == 0:
#             active_macros.append(macros_dict["Default"])
#             active_weights.append(weights_dict["Default"])
#             active_conditions.append("Default")

#         # Weighted Average
#         total_weight = sum(active_weights)
#         protein = sum(m["Protein"] * w for m, w in zip(active_macros, active_weights)) / total_weight
#         carb = sum(m["Carb"] * w for m, w in zip(active_macros, active_weights)) / total_weight
#         fat = sum(m["Fat"] * w for m, w in zip(active_macros, active_weights)) / total_weight

#         # Goal Adjustment
#         goal = row.get('goal', 'Maintenance')
#         if goal == "Fat_Loss":
#             protein += 5
#             carb -= 5
#         elif goal == "Muscle_Gain":
#             protein += 5
#             carb += 5
#             fat -= 5

#         # Normalize to 100%
#         total = protein + carb + fat
#         if total != 100:
#             protein = round(protein / total * 100, 1)
#             carb = round(carb / total * 100, 1)
#             fat = round(fat / total * 100, 1)

#         return {'protein': protein, 'carbs': carb, 'fat': fat}

#     def predict_macros(self, user_data: Dict):
#         """الدالة العامة المستخدمة من باقي النظام"""
#         return self.weighted_macros(user_data)

#     # ====================== FOOD DATABASE ======================
#     def create_healthy_food_database(self):
#         """قاعدة بيانات أكلات غنية (محدثة)"""
#         self.food_db = pd.DataFrame([
#             # Protein
#             {'ingr_name': 'chicken breast', 'category': 'Protein', 'cal/g': 1.65, 'protein(g)': 0.31, 'carb(g)': 0.00, 'fat(g)': 0.036},
#             {'ingr_name': 'grilled chicken', 'category': 'Protein', 'cal/g': 1.65, 'protein(g)': 0.31, 'carb(g)': 0.00, 'fat(g)': 0.036},
#             {'ingr_name': 'turkey breast', 'category': 'Protein', 'cal/g': 1.35, 'protein(g)': 0.30, 'carb(g)': 0.00, 'fat(g)': 0.02},
#             {'ingr_name': 'lean beef', 'category': 'Protein', 'cal/g': 2.0, 'protein(g)': 0.26, 'carb(g)': 0.00, 'fat(g)': 0.10},
#             {'ingr_name': 'salmon', 'category': 'Protein', 'cal/g': 2.08, 'protein(g)': 0.20, 'carb(g)': 0.00, 'fat(g)': 0.13},
#             {'ingr_name': 'tuna', 'category': 'Protein', 'cal/g': 1.3, 'protein(g)': 0.29, 'carb(g)': 0.00, 'fat(g)': 0.01},
#             {'ingr_name': 'tilapia', 'category': 'Protein', 'cal/g': 1.28, 'protein(g)': 0.26, 'carb(g)': 0.00, 'fat(g)': 0.027},
#             {'ingr_name': 'cod', 'category': 'Protein', 'cal/g': 0.85, 'protein(g)': 0.19, 'carb(g)': 0.00, 'fat(g)': 0.005},
#             {'ingr_name': 'eggs', 'category': 'Protein', 'cal/g': 1.55, 'protein(g)': 0.125, 'carb(g)': 0.012, 'fat(g)': 0.105},
#             {'ingr_name': 'egg whites', 'category': 'Protein', 'cal/g': 0.52, 'protein(g)': 0.11, 'carb(g)': 0.007, 'fat(g)': 0.002},
#             {'ingr_name': 'greek yogurt', 'category': 'Protein', 'cal/g': 0.59, 'protein(g)': 0.10, 'carb(g)': 0.036, 'fat(g)': 0.004},
#             {'ingr_name': 'cottage cheese', 'category': 'Protein', 'cal/g': 0.98, 'protein(g)': 0.11, 'carb(g)': 0.034, 'fat(g)': 0.043},
#             {'ingr_name': 'tofu', 'category': 'Protein', 'cal/g': 0.76, 'protein(g)': 0.08, 'carb(g)': 0.019, 'fat(g)': 0.048},

#             # Carbs
#             {'ingr_name': 'brown rice', 'category': 'Carb', 'cal/g': 1.11, 'protein(g)': 0.026, 'carb(g)': 0.23, 'fat(g)': 0.009},
#             {'ingr_name': 'quinoa', 'category': 'Carb', 'cal/g': 1.20, 'protein(g)': 0.044, 'carb(g)': 0.21, 'fat(g)': 0.019},
#             {'ingr_name': 'oats', 'category': 'Carb', 'cal/g': 3.89, 'protein(g)': 0.169, 'carb(g)': 0.66, 'fat(g)': 0.069},
#             {'ingr_name': 'whole wheat bread', 'category': 'Carb', 'cal/g': 2.47, 'protein(g)': 0.13, 'carb(g)': 0.42, 'fat(g)': 0.033},
#             {'ingr_name': 'sweet potato', 'category': 'Carb', 'cal/g': 0.86, 'protein(g)': 0.016, 'carb(g)': 0.20, 'fat(g)': 0.001},
#             {'ingr_name': 'potato', 'category': 'Carb', 'cal/g': 0.77, 'protein(g)': 0.02, 'carb(g)': 0.17, 'fat(g)': 0.001},
#             {'ingr_name': 'whole wheat pasta', 'category': 'Carb', 'cal/g': 1.24, 'protein(g)': 0.05, 'carb(g)': 0.25, 'fat(g)': 0.01},
#             {'ingr_name': 'lentils', 'category': 'Carb', 'cal/g': 1.16, 'protein(g)': 0.09, 'carb(g)': 0.20, 'fat(g)': 0.004},
#             {'ingr_name': 'chickpeas', 'category': 'Carb', 'cal/g': 1.39, 'protein(g)': 0.08, 'carb(g)': 0.23, 'fat(g)': 0.026},
#             {'ingr_name': 'black beans', 'category': 'Carb', 'cal/g': 1.32, 'protein(g)': 0.09, 'carb(g)': 0.24, 'fat(g)': 0.005},

#              #fruite
#             {'ingr_name': 'banana', 'category': 'Carb', 'cal/g': 0.89, 'protein(g)': 0.011, 'carb(g)': 0.23, 'fat(g)': 0.003},
#             {'ingr_name': 'apple', 'category': 'Carb', 'cal/g': 0.52, 'protein(g)': 0.003, 'carb(g)': 0.14, 'fat(g)': 0.002},
#             {'ingr_name': 'berries', 'category': 'Carb', 'cal/g': 0.57, 'protein(g)': 0.007, 'carb(g)': 0.14, 'fat(g)': 0.003},
#             {'ingr_name': 'orange', 'category': 'Carb', 'cal/g': 0.47, 'protein(g)': 0.009, 'carb(g)': 0.12, 'fat(g)': 0.001},


#             # Vegetables
#             {'ingr_name': 'broccoli', 'category': 'Vegetable', 'cal/g': 0.35, 'protein(g)': 0.024, 'carb(g)': 0.07, 'fat(g)': 0.004},
#             {'ingr_name': 'spinach', 'category': 'Vegetable', 'cal/g': 0.23, 'protein(g)': 0.029, 'carb(g)': 0.036, 'fat(g)': 0.004},
#             {'ingr_name': 'kale', 'category': 'Vegetable', 'cal/g': 0.49, 'protein(g)': 0.043, 'carb(g)': 0.09, 'fat(g)': 0.007},
#             {'ingr_name': 'asparagus', 'category': 'Vegetable', 'cal/g': 0.20, 'protein(g)': 0.022, 'carb(g)': 0.039, 'fat(g)': 0.001},
#             {'ingr_name': 'cauliflower', 'category': 'Vegetable', 'cal/g': 0.25, 'protein(g)': 0.019, 'carb(g)': 0.05, 'fat(g)': 0.003},
#             {'ingr_name': 'zucchini', 'category': 'Vegetable', 'cal/g': 0.17, 'protein(g)': 0.012, 'carb(g)': 0.031, 'fat(g)': 0.004},
#             {'ingr_name': 'bell peppers', 'category': 'Vegetable', 'cal/g': 0.26, 'protein(g)': 0.01, 'carb(g)': 0.06, 'fat(g)': 0.003},
#             {'ingr_name': 'tomatoes', 'category': 'Vegetable', 'cal/g': 0.18, 'protein(g)': 0.009, 'carb(g)': 0.039, 'fat(g)': 0.002},
#             {'ingr_name': 'cucumber', 'category': 'Vegetable', 'cal/g': 0.15, 'protein(g)': 0.007, 'carb(g)': 0.036, 'fat(g)': 0.001},
#             {'ingr_name': 'carrots', 'category': 'Vegetable', 'cal/g': 0.41, 'protein(g)': 0.009, 'carb(g)': 0.10, 'fat(g)': 0.002},

#             # Fats
#             {'ingr_name': 'avocado', 'category': 'Fat', 'cal/g': 1.60, 'protein(g)': 0.02, 'carb(g)': 0.09, 'fat(g)': 0.15},
#             {'ingr_name': 'olive oil', 'category': 'Fat', 'cal/g': 8.84, 'protein(g)': 0.00, 'carb(g)': 0.00, 'fat(g)': 1.00},
#             {'ingr_name': 'almonds', 'category': 'Fat', 'cal/g': 5.78, 'protein(g)': 0.21, 'carb(g)': 0.22, 'fat(g)': 0.50},
#             {'ingr_name': 'walnuts', 'category': 'Fat', 'cal/g': 6.54, 'protein(g)': 0.15, 'carb(g)': 0.14, 'fat(g)': 0.65},
            

#             # drinks
#             {'ingr_name': 'water', 'category': 'Drink', 'cal/g': 0.0, 'protein(g)': 0.0, 'carb(g)': 0.0, 'fat(g)': 0.0},
#             {'ingr_name': 'green tea', 'category': 'Drink', 'cal/g': 0.01, 'protein(g)': 0.0, 'carb(g)': 0.0, 'fat(g)': 0.0},
 
#             ])

#         self.categorize_foods_correctly()
#         print(f"✅ Food database loaded with {len(self.food_db)} items")

#     def categorize_foods_correctly(self):
#         """تصنيف الأكلات (من الـ Notebook)"""
#         self.food_db['protein_per_100g'] = self.food_db['protein(g)'] * 100
#         self.food_db['carb_per_100g'] = self.food_db['carb(g)'] * 100
#         self.food_db['fat_per_100g'] = self.food_db['fat(g)'] * 100

#         self.food_db['cal_from_protein'] = self.food_db['protein_per_100g'] * 4
#         self.food_db['cal_from_carb'] = self.food_db['carb_per_100g'] * 4
#         self.food_db['cal_from_fat'] = self.food_db['fat_per_100g'] * 9
#         self.food_db['total_calc_cal'] = self.food_db[['cal_from_protein', 'cal_from_carb', 'cal_from_fat']].sum(axis=1)
#         self.food_db['total_calc_cal'] = self.food_db['total_calc_cal'].replace(0, 1)

#         self.food_db['protein_pct'] = (self.food_db['cal_from_protein'] / self.food_db['total_calc_cal'] * 100).round(1)
#         self.food_db['carb_pct'] = (self.food_db['cal_from_carb'] / self.food_db['total_calc_cal'] * 100).round(1)
#         self.food_db['fat_pct'] = (self.food_db['cal_from_fat'] / self.food_db['total_calc_cal'] * 100).round(1)

#         conditions = [
#             (self.food_db['protein_pct'] >= 30),
#             (self.food_db['carb_pct'] >= 40),
#             (self.food_db['fat_pct'] >= 35),
#         ]
#         choices = ['High_Protein', 'High_Carb', 'High_Fat']
#         self.food_db['macro_type'] = np.select(conditions, choices, default='Balanced')

#         self.protein_foods = self.food_db[self.food_db['macro_type'] == 'High_Protein']
#         self.carb_foods = self.food_db[self.food_db['macro_type'] == 'High_Carb']
#         self.fat_foods = self.food_db[self.food_db['macro_type'] == 'High_Fat']
#         self.veg_foods = self.food_db[self.food_db['category'] == 'Vegetable']

#     # ====================== PLAN GENERATION ======================
#     def generate_daily_plan(self, user_data: Dict, total_calories: int):
#         macros = self.predict_macros(user_data)
#         meal_needs = self.calculate_meal_needs(total_calories, macros, num_meals=3)

#         plan = {
#             'user_info': user_data,
#             'daily_macros': macros,
#             'meals': []
#         }

#         meal_types = ['Breakfast', 'Lunch', 'Dinner']
#         for i, needs in enumerate(meal_needs):
#             meal = self.recommend_meal_from_db(
#                 needs['protein_g'],
#                 needs['carbs_g'],
#                 needs['fat_g'],
#                 needs['calories'],
#                 meal_types[i],
#                 user_data
#             )
#             plan['meals'].append({
#                 'name': meal_types[i],
#                 'foods': meal['foods'],
#                 'totals': meal['totals']
#             })

#         return {
#             'daily_macros': macros,
#             'total_calories': total_calories,
#             'meals': []  # ستكملينها حسب احتياجك
#         }

#     def generate_weekly_plan(self, user_data: Dict, total_calories: int):
#         """توليد خطة أسبوعية (7 أيام)"""
#         days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
#         weekly_plan = {
#             'daily_macros': self.predict_macros(user_data),
#             'days': []
#         }

#         for day in days:
#             daily = self.generate_daily_plan(user_data, total_calories)
#             daily['day'] = day
#             weekly_plan['days'].append(daily)

#         return weekly_plan


# # ====================== Exercise Planner (متروك كما هو) ======================
# class ExercisePlanner:
#     def __init__(self):
#         # Define exercises
#         self.exercise_rules = {
#             "gym": {
#                 "bench_press": {"joints": ["elbow_left", "elbow_right"], "up_threshold": 150, "down_threshold": 80, "min_angle": 70, "max_angle": 160},
#                 "shoulder_press": {"joints": ["elbow_left", "elbow_right"], "up_threshold": 160, "down_threshold": 90, "min_angle": 80, "max_angle": 170},
#                 "lat_pulldown": {"joints": ["elbow_left", "elbow_right"], "up_threshold": 150, "down_threshold": 60, "min_angle": 45, "max_angle": 160},
#                 "bicep_curl": {"joints": ["elbow_left", "elbow_right"], "up_threshold": 40, "down_threshold": 150, "min_angle": 30, "max_angle": 160},
#                 "triceps_pushdown": {"joints": ["elbow_left", "elbow_right"], "up_threshold": 160, "down_threshold": 40, "min_angle": 30, "max_angle": 170},
#                 "leg_extension": {"joints": ["knee_left", "knee_right"], "up_threshold": 160, "down_threshold": 100, "min_angle": 90, "max_angle": 175}
#             },
#             "rehab": {
#                 "arm_abduction": {"joints": ["shoulder_abduction_left", "shoulder_abduction_right"], "up_threshold": 80, "down_threshold": 20, "min_angle": 15, "max_angle": 95},
#                 "shoulder_flexion": {"joints": ["shoulder_flexion_left", "shoulder_flexion_right"], "up_threshold": 90, "down_threshold": 10, "min_angle": 0, "max_angle": 180},
#                 "arm_vw": {"joints": ["elbow_left", "elbow_right"], "up_threshold": 120, "down_threshold": 60, "min_angle": 45, "max_angle": 135},
#                 "pushups": {"joints": ["elbow_left", "elbow_right"], "up_threshold": 160, "down_threshold": 80, "min_angle": 70, "max_angle": 170},
#                 "leg_abduction": {"joints": ["hip_left", "hip_right"], "up_threshold": 40, "down_threshold": 10, "min_angle": 0, "max_angle": 45},
#                 "squats": {"joints": ["knee_left", "knee_right"], "up_threshold": 160, "down_threshold": 80, "min_angle": 70, "max_angle": 170},
#                 "leg_lunge": {"joints": ["knee_left", "knee_right"], "up_threshold": 160, "down_threshold": 95, "min_angle": 85, "max_angle": 170}
#             },
#             "fitness": {
#                 "bodyweight_squats": {"joints": ["knee_left", "knee_right"], "up_threshold": 120, "down_threshold": 60, "min_angle": 40, "max_angle": 130},
#                 "jumping_jacks": {"joints": ["shoulder_abduction_left", "shoulder_abduction_right"], "up_threshold": 140, "down_threshold": 30, "min_angle": 20, "max_angle": 150},
#                 "lunge": {"joints": ["knee_left", "knee_right"], "up_threshold": 160, "down_threshold": 95, "min_angle": 85, "max_angle": 170},
#                 "leg_swing": {"joints": ["hip_left", "hip_right"], "up_threshold": 80, "down_threshold": 10, "min_angle": 0, "max_angle": 90},
#                 "butt_kicks": {"joints": ["knee_left", "knee_right"], "up_threshold": 60, "down_threshold": 150, "min_angle": 45, "max_angle": 160},
#                 "high_knee": {"joints": ["hip_left", "hip_right"], "up_threshold": 80, "down_threshold": 160, "min_angle": 70, "max_angle": 170},
#                 "arm_circles": {
#                     "joints": ["shoulder_left", "shoulder_right"],
#                     "up_threshold": 90,
#                     "down_threshold": 180,
#                     "min_angle": 30,
#                     "max_angle": 180,
#                     "special_function": "arm_circle"
#                 },
#                 "arm_half_circles": {
#                     "joints": ["shoulder_left", "shoulder_right"],
#                     "up_threshold": 80,
#                     "down_threshold": 120,
#                     "min_angle": 30,
#                     "max_angle": 120,
#                     "special_function": "arm_circle"
#                 }
#             }
#         }

#         # Classify exercises
#         self.exercise_categories = self.categorize_exercises()

#     def categorize_exercises(self):
#         """Classify exercises by type and target muscles"""
#         categories = {
#             'Upper_Body': [],
#             'Lower_Body': [],
#             'Core': [],
#             'Cardio': [],
#             'Full_Body': []
#         }

#         for category, exercises in self.exercise_rules.items():
#             for ex_name, ex_data in exercises.items():
#                 # Determine exercise type based on joints
#                 joints = ex_data['joints']
#                 if any('knee' in j or 'hip' in j for j in joints):
#                     if any('shoulder' in j or 'elbow' in j for j in joints):
#                         categories['Full_Body'].append({
#                             'name': ex_name,
#                             'category': category,
#                             'rules': ex_data
#                         })
#                     else:
#                         categories['Lower_Body'].append({
#                             'name': ex_name,
#                             'category': category,
#                             'rules': ex_data
#                         })
#                 elif any('shoulder' in j or 'elbow' in j for j in joints):
#                     categories['Upper_Body'].append({
#                         'name': ex_name,
#                         'category': category,
#                         'rules': ex_data
#                     })
#                 else:
#                     categories['Cardio'].append({
#                         'name': ex_name,
#                         'category': category,
#                         'rules': ex_data
#                     })

#         return categories

#     def select_exercises_by_goal(self, goal: str, available_days: int = 4, injuries: Dict = None) -> Dict:
#         """Select appropriate exercises based on goal considering injuries"""

#         exercise_plan = {
#             'weekly_schedule': [],
#             'exercise_details': {},
#             'total_weekly_volume': 0,
#             'tips': []
#         }

#         # Determine exercise distribution based on goal
#         if goal == 'Muscle_Gain':
#             distribution = {
#                 'Upper_Body': 0.4,
#                 'Lower_Body': 0.3,
#                 'Full_Body': 0.2,
#                 'Core': 0.1
#             }
#             sets_per_exercise = 4
#             reps_range = "8-12"
#             rest_time = "90 sec"
#             exercise_plan['tips'] = [
#                 'Focus on gradually increasing weights',
#                 'Take adequate rest between sets',
#                 'Eat enough protein within 2 hours of workout'
#             ]

#         elif goal == 'Fat_Loss':
#             distribution = {
#                 'Cardio': 0.4,
#                 'Full_Body': 0.3,
#                 'Upper_Body': 0.15,
#                 'Lower_Body': 0.15
#             }
#             sets_per_exercise = 3
#             reps_range = "15-20"
#             rest_time = "45 sec"
#             exercise_plan['tips'] = [
#                 'Maintain high heart rate',
#                 'Reduce rest periods between exercises',
#                 'Drink enough water before and during workout'
#             ]

#         else:  # Maintenance
#             distribution = {
#                 'Full_Body': 0.3,
#                 'Upper_Body': 0.25,
#                 'Lower_Body': 0.25,
#                 'Core': 0.1,
#                 'Cardio': 0.1
#             }
#             sets_per_exercise = 3
#             reps_range = "12-15"
#             rest_time = "60 sec"
#             exercise_plan['tips'] = [
#                 'Mix between strength and cardio exercises',
#                 'Listen to your body and rest when needed',
#                 'Maintain consistent performance'
#             ]

#         # Add injury tips
#         if injuries:
#             if injuries.get('back_pain', 0) == 1:
#                 exercise_plan['tips'].append('Avoid exercises that strain the back')
#                 # Reduce Lower_Body exercises
#                 distribution['Lower_Body'] = distribution.get('Lower_Body', 0) * 0.5
#                 distribution['Upper_Body'] = distribution.get('Upper_Body', 0) + 0.1
#             if injuries.get('ankle_injury', 0) == 1:
#                 exercise_plan['tips'].append('Use low-impact exercises for ankles')
#                 # Reduce high-impact cardio
#                 distribution['Cardio'] = distribution.get('Cardio', 0) * 0.7

#         # Select exercises for each day
#         days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

#         selected_days = random.sample(days, min(available_days, len(days)))
#         selected_days.sort(key=lambda x: days.index(x))

#         for day in selected_days:
#             day_exercises = []

#             # Select 4-6 exercises per day
#             num_exercises = min(random.randint(4, 6), available_days * 2)

#             for _ in range(num_exercises):
#                 # Select exercise type based on distribution
#                 exercise_type = random.choices(
#                     list(distribution.keys()),
#                     weights=list(distribution.values())
#                 )[0]

#                 # Select specific exercise from category
#                 if self.exercise_categories[exercise_type]:
#                     exercise = random.choice(self.exercise_categories[exercise_type])

#                     # Avoid repeating the same exercise on the same day
#                     if any(ex['name'] == exercise['name'] for ex in day_exercises):
#                         continue

#                     # Add exercise details
#                     exercise_detail = {
#                         'name': exercise['name'].replace('_', ' ').title(),
#                         'type': exercise_type,
#                         'category': exercise['category'],
#                         'sets': sets_per_exercise,
#                         'reps': reps_range,
#                         'rest': rest_time,
#                         'joints': exercise['rules']['joints']
#                     }

#                     day_exercises.append(exercise_detail)

#                     # Store complete details
#                     if exercise['name'] not in exercise_plan['exercise_details']:
#                         exercise_plan['exercise_details'][exercise['name']] = {
#                             'rules': exercise['rules'],
#                             'total_volume_sets': 0
#                         }
#                     exercise_plan['exercise_details'][exercise['name']]['total_volume_sets'] += sets_per_exercise

#             exercise_plan['weekly_schedule'].append({
#                 'day': day,
#                 'exercises': day_exercises[:min(6, len(day_exercises))],  # Maximum 6 exercises
#                 'total_exercises': min(6, len(day_exercises))
#             })

#         # Calculate total volume
#         exercise_plan['total_weekly_volume'] = sum(
#             len(day['exercises']) for day in exercise_plan['weekly_schedule']
#         )

#         return exercise_plan

#     def print_workout_plan(self, plan: Dict):
#         """Print formatted workout plan"""
#         print("\n" + "="*70)
#         print("💪 Weekly Workout Plan:")
#         print("="*70)

#         for workout in plan['weekly_schedule']:
#             print(f"\n📅 {workout['day']}:")
#             print("-" * 40)
#             for ex in workout['exercises']:
#                 print(f"  • {ex['name']}")
#                 print(f"    {ex['sets']} sets × {ex['reps']} (rest {ex['rest']})")
#                 print(f"    Joints: {', '.join(set([j.replace('_', ' ').title() for j in ex['joints']]))}")

#         print("\n💡 Exercise Tips:")
#         for tip in plan['tips']:
#             print(f"  • {tip}")
