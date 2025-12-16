#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KAGGLE INFLUENZA PREDICTION - STRATÉGIE OPTIMISÉE
Objectif: Maximiser le score sur le leaderboard Kaggle
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("KAGGLE INFLUENZA PREDICTION - STRATÉGIE OPTIMISÉE")
print("="*80)

#%% 1. CHARGEMENT ET ANALYSE DES DONNÉES
print("\n" + "="*80)
print("1. CHARGEMENT DES DONNÉES")
print("="*80)

# Charger les datasets nettoyés
df_train_full = pd.read_csv('data_plus/train_synop_cleaned_complet.csv')
print(f"✓ Train data loaded: {df_train_full.shape}")

# Convertir date
df_train_full['date'] = pd.to_datetime(df_train_full['date'])

# Analyser la structure temporelle
print(f"\nPériode TRAIN: {df_train_full['date'].min()} à {df_train_full['date'].max()}")
print(f"Régions: {df_train_full['region_code'].nunique()}")
print(f"Semaines: {df_train_full['week_year'].nunique()}")

#%% 2. FEATURE ENGINEERING STRATÉGIQUE
print("\n" + "="*80)
print("2. FEATURE ENGINEERING POUR MAXIMISER LE SCORE KAGGLE")
print("="*80)

def create_temporal_features(df):
    """
    Créer les features temporelles optimales pour prédire 2 ans dans le futur
    """
    df = df.copy()

    # Extraire composantes temporelles
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['day_of_year'] = df['date'].dt.dayofyear

    # Features cycliques (IMPORTANT pour la saisonnalité)
    df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Encoder saison
    saison_map = {'Hiver': 1, 'Printemps': 2, 'Ete': 3, 'Automne': 4}
    df['saison_encoded'] = df['saison'].map(saison_map)

    return df

def create_historical_features(df):
    """
    FEATURE SECRET WEAPON: Moyennes historiques par région/semaine/mois
    Cette feature capture les patterns saisonniers de chaque région
    """
    df = df.copy()

    # 1. Moyenne par région + semaine de l'année
    # Ex: Alsace semaine 10 → moyenne de toutes les semaines 10 historiques
    historical_week_mean = df.groupby(['region_code', 'week_of_year'])['TauxGrippe'].transform('mean')
    df['TauxGrippe_hist_week_mean'] = historical_week_mean

    # 2. Moyenne par région + mois
    historical_month_mean = df.groupby(['region_code', 'month'])['TauxGrippe'].transform('mean')
    df['TauxGrippe_hist_month_mean'] = historical_month_mean

    # 3. Moyenne par région + saison
    historical_season_mean = df.groupby(['region_code', 'saison'])['TauxGrippe'].transform('mean')
    df['TauxGrippe_hist_season_mean'] = historical_season_mean

    # 4. Stats globales par région
    regional_mean = df.groupby('region_code')['TauxGrippe'].transform('mean')
    regional_std = df.groupby('region_code')['TauxGrippe'].transform('std')
    df['TauxGrippe_region_mean'] = regional_mean
    df['TauxGrippe_region_std'] = regional_std

    # 5. Stats globales par semaine (toutes régions)
    week_mean = df.groupby('week_of_year')['TauxGrippe'].transform('mean')
    df['TauxGrippe_week_global_mean'] = week_mean

    return df

# Appliquer le feature engineering
print("\n📊 Création des features temporelles...")
df_train_full = create_temporal_features(df_train_full)

print("🎯 Création des features historiques (SECRET WEAPON)...")
df_train_full = create_historical_features(df_train_full)

print(f"✓ Features créées. Shape: {df_train_full.shape}")

#%% 3. SÉLECTION DES FEATURES OPTIMALES
print("\n" + "="*80)
print("3. SÉLECTION DES FEATURES POUR MAXIMISER LA PERFORMANCE")
print("="*80)

# Features météo (disponibles dans le test)
meteo_features = ['t', 'u', 'td', 'ff', 'vv', 'tminsol', 'pres',
                  'rr3', 'rr6', 'rr12', 'rr24', 'n']

# Features temporelles
temporal_features = ['week_of_year', 'month', 'week_sin', 'week_cos',
                     'month_sin', 'month_cos', 'saison_encoded']

# Features historiques (SECRET WEAPON)
historical_features = ['TauxGrippe_hist_week_mean', 'TauxGrippe_hist_month_mean',
                       'TauxGrippe_hist_season_mean', 'TauxGrippe_region_mean',
                       'TauxGrippe_region_std', 'TauxGrippe_week_global_mean']

# Région (catégorielle)
region_features = ['region_code']

# Combiner toutes les features
all_features = meteo_features + temporal_features + historical_features + region_features

# Vérifier que toutes les features existent
available_features = [f for f in all_features if f in df_train_full.columns]
print(f"\n✓ Features sélectionnées: {len(available_features)}")
print(f"  - Météo: {len([f for f in meteo_features if f in available_features])}")
print(f"  - Temporelles: {len([f for f in temporal_features if f in available_features])}")
print(f"  - Historiques: {len([f for f in historical_features if f in available_features])}")

#%% 4. CRÉATION DU SPLIT DE VALIDATION (MIMIQUE LE TEST!)
print("\n" + "="*80)
print("4. SPLIT DE VALIDATION STRATÉGIQUE")
print("="*80)

# STRATÉGIE:
# - Train: 2004-2010 (7 ans)
# - Validation: 2011 (1 an) → Mimique le test qui est 2012-2013
# Cela simule parfaitement la situation réelle!

df_train = df_train_full[df_train_full['year'] <= 2010].copy()
df_val = df_train_full[df_train_full['year'] == 2011].copy()

print(f"\n✓ Train set: {df_train.shape[0]} obs ({df_train['date'].min()} à {df_train['date'].max()})")
print(f"✓ Validation set: {df_val.shape[0]} obs ({df_val['date'].min()} à {df_val['date'].max()})")

# Préparer X, y
X_train = df_train[available_features].copy()
y_train = df_train['TauxGrippe'].copy()

X_val = df_val[available_features].copy()
y_val = df_val['TauxGrippe'].copy()

# Gestion des NaN
print(f"\n📋 Gestion des valeurs manquantes...")
print(f"NaN dans X_train: {X_train.isnull().sum().sum()}")
print(f"NaN dans X_val: {X_val.isnull().sum().sum()}")

# Imputer les NaN avec la médiane
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
X_train_imputed = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)
X_val_imputed = pd.DataFrame(
    imputer.transform(X_val),
    columns=X_val.columns,
    index=X_val.index
)

print(f"✓ Imputation terminée")

#%% 5. ENTRAÎNEMENT DES MEILLEURS MODÈLES
print("\n" + "="*80)
print("5. ENTRAÎNEMENT DES MODÈLES TOP PERFORMANCE")
print("="*80)

results = {}

# ===== MODÈLE 1: XGBoost (PRIORITÉ #1) =====
print("\n🚀 [1/3] XGBoost...")
xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(
    X_train_imputed, y_train,
    eval_set=[(X_val_imputed, y_val)],
    early_stopping_rounds=50,
    verbose=False
)

y_val_pred_xgb = xgb_model.predict(X_val_imputed)
rmse_xgb = np.sqrt(mean_squared_error(y_val, y_val_pred_xgb))
mae_xgb = mean_absolute_error(y_val, y_val_pred_xgb)
r2_xgb = r2_score(y_val, y_val_pred_xgb)

results['XGBoost'] = {'RMSE': rmse_xgb, 'MAE': mae_xgb, 'R²': r2_xgb}
print(f"✓ XGBoost - RMSE: {rmse_xgb:.2f} | MAE: {mae_xgb:.2f} | R²: {r2_xgb:.4f}")

# ===== MODÈLE 2: LightGBM =====
print("\n⚡ [2/3] LightGBM...")
lgb_model = lgb.LGBMRegressor(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

lgb_model.fit(
    X_train_imputed, y_train,
    eval_set=[(X_val_imputed, y_val)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
)

y_val_pred_lgb = lgb_model.predict(X_val_imputed)
rmse_lgb = np.sqrt(mean_squared_error(y_val, y_val_pred_lgb))
mae_lgb = mean_absolute_error(y_val, y_val_pred_lgb)
r2_lgb = r2_score(y_val, y_val_pred_lgb)

results['LightGBM'] = {'RMSE': rmse_lgb, 'MAE': mae_lgb, 'R²': r2_lgb}
print(f"✓ LightGBM - RMSE: {rmse_lgb:.2f} | MAE: {mae_lgb:.2f} | R²: {r2_lgb:.4f}")

# ===== MODÈLE 3: CatBoost =====
print("\n🐱 [3/3] CatBoost...")
cat_model = CatBoostRegressor(
    iterations=500,
    depth=7,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42,
    verbose=False
)

cat_model.fit(
    X_train_imputed, y_train,
    eval_set=(X_val_imputed, y_val),
    early_stopping_rounds=50,
    verbose=False
)

y_val_pred_cat = cat_model.predict(X_val_imputed)
rmse_cat = np.sqrt(mean_squared_error(y_val, y_val_pred_cat))
mae_cat = mean_absolute_error(y_val, y_val_pred_cat)
r2_cat = r2_score(y_val, y_val_pred_cat)

results['CatBoost'] = {'RMSE': rmse_cat, 'MAE': mae_cat, 'R²': r2_cat}
print(f"✓ CatBoost - RMSE: {rmse_cat:.2f} | MAE: {mae_cat:.2f} | R²: {r2_cat:.4f}")

# ===== MODÈLE 4: ENSEMBLE (moyenne pondérée) =====
print("\n🎯 [4/4] Ensemble (moyenne pondérée)...")

# Pondération basée sur les performances de validation
weights = {
    'XGBoost': 1/rmse_xgb,
    'LightGBM': 1/rmse_lgb,
    'CatBoost': 1/rmse_cat
}
total_weight = sum(weights.values())
weights = {k: v/total_weight for k, v in weights.items()}

print(f"Poids: XGBoost={weights['XGBoost']:.3f}, LightGBM={weights['LightGBM']:.3f}, CatBoost={weights['CatBoost']:.3f}")

y_val_pred_ensemble = (
    weights['XGBoost'] * y_val_pred_xgb +
    weights['LightGBM'] * y_val_pred_lgb +
    weights['CatBoost'] * y_val_pred_cat
)

rmse_ensemble = np.sqrt(mean_squared_error(y_val, y_val_pred_ensemble))
mae_ensemble = mean_absolute_error(y_val, y_val_pred_ensemble)
r2_ensemble = r2_score(y_val, y_val_pred_ensemble)

results['Ensemble'] = {'RMSE': rmse_ensemble, 'MAE': mae_ensemble, 'R²': r2_ensemble}
print(f"✓ Ensemble - RMSE: {rmse_ensemble:.2f} | MAE: {mae_ensemble:.2f} | R²: {r2_ensemble:.4f}")

#%% 6. COMPARAISON DES RÉSULTATS
print("\n" + "="*80)
print("6. COMPARAISON DES PERFORMANCES")
print("="*80)

df_results = pd.DataFrame(results).T.sort_values('RMSE')
print("\n" + df_results.to_string())

best_model_name = df_results['RMSE'].idxmin()
best_rmse = df_results.loc[best_model_name, 'RMSE']

print(f"\n🏆 MEILLEUR MODÈLE: {best_model_name}")
print(f"   RMSE validation: {best_rmse:.2f}")
print(f"   Cela représente {100*best_rmse/y_train.mean():.1f}% du taux moyen")

#%% 7. FEATURE IMPORTANCE
print("\n" + "="*80)
print("7. ANALYSE DES FEATURES IMPORTANTES")
print("="*80)

# Feature importance XGBoost
feature_importance = pd.DataFrame({
    'feature': X_train_imputed.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n📊 Top 15 features (XGBoost):")
print(feature_importance.head(15).to_string(index=False))

# Sauvegarder
feature_importance.to_csv('feature_importance.csv', index=False)
print("\n✓ Feature importance sauvegardée: feature_importance.csv")

#%% 8. RÉENTRAÎNEMENT SUR TOUTES LES DONNÉES (2004-2011)
print("\n" + "="*80)
print("8. RÉENTRAÎNEMENT FINAL SUR TOUTES LES DONNÉES")
print("="*80)

# Utiliser TOUTES les données train (2004-2011)
X_full = df_train_full[available_features].copy()
y_full = df_train_full['TauxGrippe'].copy()

X_full_imputed = pd.DataFrame(
    imputer.fit_transform(X_full),
    columns=X_full.columns,
    index=X_full.index
)

print(f"Réentraînement sur {len(X_full)} observations (2004-2011)...")

# Réentraîner les 3 modèles
print("\n🚀 XGBoost...")
xgb_final = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    random_state=42,
    n_jobs=-1
)
xgb_final.fit(X_full_imputed, y_full, verbose=False)

print("⚡ LightGBM...")
lgb_final = lgb.LGBMRegressor(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgb_final.fit(X_full_imputed, y_full)

print("🐱 CatBoost...")
cat_final = CatBoostRegressor(
    iterations=500,
    depth=7,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42,
    verbose=False
)
cat_final.fit(X_full_imputed, y_full, verbose=False)

print("✓ Tous les modèles réentraînés!")

#%% 9. SAUVEGARDE DES MODÈLES
import pickle

print("\n💾 Sauvegarde des modèles...")
with open('xgb_final_model.pkl', 'wb') as f:
    pickle.dump(xgb_final, f)
with open('lgb_final_model.pkl', 'wb') as f:
    pickle.dump(lgb_final, f)
with open('cat_final_model.pkl', 'wb') as f:
    pickle.dump(cat_final, f)
with open('imputer.pkl', 'wb') as f:
    pickle.dump(imputer, f)

print("✓ Modèles sauvegardés: xgb_final_model.pkl, lgb_final_model.pkl, cat_final_model.pkl")

#%% 10. RÉSUMÉ FINAL
print("\n" + "="*80)
print("10. RÉSUMÉ FINAL")
print("="*80)

print(f"\n📊 STATISTIQUES:")
print(f"   Train: {len(df_train)} obs (2004-2010)")
print(f"   Validation: {len(df_val)} obs (2011)")
print(f"   Features: {len(available_features)}")
print(f"   Régions: {df_train_full['region_code'].nunique()}")

print(f"\n🎯 PERFORMANCES (validation 2011):")
for model_name, metrics in results.items():
    print(f"   {model_name:12s}: RMSE={metrics['RMSE']:6.2f} | R²={metrics['R²']:.4f}")

print(f"\n🏆 MEILLEUR MODÈLE: {best_model_name} (RMSE={best_rmse:.2f})")

print(f"\n💡 PROCHAINES ÉTAPES:")
print(f"   1. Préparer le test set (2012-2013) avec les mêmes features")
print(f"   2. Générer les prédictions avec les modèles finaux")
print(f"   3. Créer le fichier submission.csv")
print(f"   4. Soumettre sur Kaggle!")

print("\n" + "="*80)
print("✅ ENTRAÎNEMENT TERMINÉ - Prêt pour la prédiction!")
print("="*80)
