#!/usr/bin/env python3
"""Test d'un modèle simple pour prédire le TauxGrippe"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("MODÈLE SIMPLE - PRÉDICTION TAUX GRIPPE")
print("="*80)

# ============================================================================
# 1. CHARGER LES DONNÉES
# ============================================================================
print("\n[1/5] Chargement des données...")
df = pd.read_csv('data_plus/train_weather_merged_complete.csv')
print(f"✓ {df.shape[0]} lignes, {df.shape[1]} colonnes")

# ============================================================================
# 2. PRÉPARATION DES DONNÉES
# ============================================================================
print("\n[2/5] Préparation des features...")

# Variables à utiliser comme features
feature_cols = [
    # Variables météo principales
    't', 'td', 'u', 'ff', 'vv', 'dd',           # Température, humidité, vent
    'tminsol', 'nbas', 'n',                     # Sol, nébulosité
    'rr1', 'rr3', 'rr6', 'rr12', 'rr24',       # Précipitations
    'pres', 'tn12', 'tx12',                     # Pression, temp min/max
    'rafper', 'per', 'ht_neige',                # Vent, neige
    # Variables temporelles
    'week_year',
    'region_code'
]

# Garder seulement les colonnes qui existent
feature_cols = [c for c in feature_cols if c in df.columns]
print(f"  Features utilisées : {len(feature_cols)}")

# Target
target = 'TauxGrippe'

# Créer X et y
X = df[feature_cols].copy()
y = df[target].copy()

# Gérer les NaN (imputation simple par la médiane)
print(f"  NaN avant imputation : {X.isnull().sum().sum()}")
for col in X.columns:
    if X[col].isnull().sum() > 0:
        X[col].fillna(X[col].median(), inplace=True)
print(f"  NaN après imputation : {X.isnull().sum().sum()}")

# ============================================================================
# 3. SPLIT TRAIN/TEST
# ============================================================================
print("\n[3/5] Split train/test...")

# Split temporel : 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"  Train : {X_train.shape[0]} lignes")
print(f"  Test  : {X_test.shape[0]} lignes")

# ============================================================================
# 4. ENTRAÎNEMENT DU MODÈLE
# ============================================================================
print("\n[4/5] Entraînement Random Forest...")

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
print("✓ Modèle entraîné")

# ============================================================================
# 5. ÉVALUATION
# ============================================================================
print("\n[5/5] Évaluation...")

# Prédictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Métriques Train
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
mae_train = mean_absolute_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)

# Métriques Test
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print("\n" + "="*80)
print("RÉSULTATS")
print("="*80)

print("\n📊 MÉTRIQUES TRAIN:")
print(f"  RMSE : {rmse_train:.2f}")
print(f"  MAE  : {mae_train:.2f}")
print(f"  R²   : {r2_train:.4f}")

print("\n📊 MÉTRIQUES TEST:")
print(f"  RMSE : {rmse_test:.2f}")
print(f"  MAE  : {mae_test:.2f}")
print(f"  R²   : {r2_test:.4f}")

# Feature importance
print("\n📈 TOP 10 FEATURES IMPORTANTES:")
importances = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for i, row in importances.head(10).iterrows():
    print(f"  {row['feature']:15s} : {row['importance']:.4f}")

# Exemples de prédictions
print("\n🔍 EXEMPLES DE PRÉDICTIONS (TEST):")
print(f"{'Réel':>8s} {'Prédit':>8s} {'Erreur':>8s}")
print("-" * 26)
for i in range(min(10, len(y_test))):
    real = y_test.iloc[i]
    pred = y_pred_test[i]
    err = abs(real - pred)
    print(f"{real:8.1f} {pred:8.1f} {err:8.1f}")

print("\n" + "="*80)
print("✓ TERMINÉ!")
print("="*80)

# Sauvegarder les prédictions
results = pd.DataFrame({
    'TauxGrippe_reel': y_test,
    'TauxGrippe_predit': y_pred_test
})
results.to_csv('predictions_test.csv', index=False)
print("\n💾 Prédictions sauvegardées : predictions_test.csv")
