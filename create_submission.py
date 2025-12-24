#!/usr/bin/env python3
"""Génère le fichier submission.csv pour Kaggle"""

import pandas as pd
import numpy as np
import glob
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("GÉNÉRATION SUBMISSION KAGGLE")
print("="*80)

# ============================================================================
# 1. CHARGER ET ENTRAÎNER LE MODÈLE SUR TRAIN
# ============================================================================
print("\n[1/5] Entraînement du modèle sur train.csv...")

df_train = pd.read_csv('data_plus/train_weather_merged_complete.csv')
print(f"✓ Train chargé : {df_train.shape}")

# Features
feature_cols = ['t', 'td', 'u', 'ff', 'vv', 'tminsol', 'nbas', 'n',
                'rr24', 'rr12', 'rr6', 'pres', 'tn12', 'tx12',
                'week_year', 'region_code']
feature_cols = [c for c in feature_cols if c in df_train.columns]

X_train = df_train[feature_cols].copy()
y_train = df_train['TauxGrippe'].copy()

# Imputer NaN
for col in X_train.columns:
    if X_train[col].isnull().sum() > 0:
        X_train[col].fillna(X_train[col].median(), inplace=True)

# Entraîner Random Forest
print("🌲 Entraînement Random Forest...")
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=3,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("✓ Modèle entraîné")

# ============================================================================
# 2. CHARGER TEST.CSV
# ============================================================================
print("\n[2/5] Chargement test.csv...")
df_test = pd.read_csv('data_origin/test.csv')
print(f"✓ Test chargé : {df_test.shape}")
print(f"  Semaines : {df_test['week'].min()} -> {df_test['week'].max()}")

# ============================================================================
# 3. MERGER TEST AVEC DONNÉES MÉTÉO 2013
# ============================================================================
print("\n[3/5] Merge test.csv avec météo 2013...")

# Mapping régions -> stations
MAPPING = {
    'ALSACE': [7190, 7280],
    'AQUITAINE': [7510, 7630],
    'AUVERGNE': [7460, 7380],
    'BASSE-NORMANDIE': [7027, 7139],
    'BOURGOGNE': [7280, 7255],
    'BRETAGNE': [7110, 7117, 7130],
    'CENTRE': [7255, 7149],
    'CHAMPAGNE-ARDENNE': [7072, 7168],
    'CORSE': [7761, 7790],
    'FRANCHE-COMTE': [7299, 7280],
    'HAUTE-NORMANDIE': [7037, 7020],
    'ILE-DE-FRANCE': [7150, 7149],
    'LANGUEDOC-ROUSSILLON': [7630, 7643],
    'LIMOUSIN': [7434, 7335],
    'LORRAINE': [7090, 7180],
    'MIDI-PYRENEES': [7630, 7627],
    'NORD-PAS-DE-CALAIS': [7005, 7015],
    'PAYS DE LA LOIRE': [7222, 7130],
    'PICARDIE': [7005, 7015],
    'POITOU-CHARENTES': [7335, 7255],
    "PROVENCE-ALPES-COTE D'AZUR": [7650, 7690],
    'RHONE-ALPES': [7481, 7482],
}

rows = []
for region, stations in MAPPING.items():
    for station in stations:
        rows.append({'numer_sta': station, 'region_name': region})
df_mapping = pd.DataFrame(rows)

# Charger synop 2013
synop_files = sorted(glob.glob('DonneesMeteorologiques/DonneesMeteorologiques/synop.2013*.csv'))
print(f"  {len(synop_files)} fichiers synop 2013 trouvés")

stations_list = df_mapping['numer_sta'].unique().tolist()
all_data = []

for fpath in synop_files:
    try:
        df = pd.read_csv(fpath, sep=';', low_memory=False)
        df = df[df['numer_sta'].isin(stations_list)]
        if len(df) > 0:
            all_data.append(df)
    except Exception as e:
        print(f"  ⚠ Erreur {fpath}: {e}")

df_synop = pd.concat(all_data, ignore_index=True)
print(f"  ✓ {len(df_synop)} observations météo chargées")

# Convertir dates en semaines
df_synop['date'] = pd.to_datetime(df_synop['date'], format='%Y%m%d%H%M%S', errors='coerce')
df_synop['year'] = df_synop['date'].dt.isocalendar().year
df_synop['week'] = df_synop['date'].dt.isocalendar().week
df_synop['week_year'] = df_synop['year'] * 100 + df_synop['week']

# Merger avec mapping
df_synop = df_synop.merge(df_mapping, on='numer_sta', how='inner')

# Variables météo
meteo_vars = ['tend', 'dd', 'ff', 't', 'td', 'u', 'vv', 'n', 'nbas', 'hbas',
              'pres', 'tn12', 'tx12', 'tminsol', 'rafper', 'per',
              'ht_neige', 'ssfrai', 'perssfrai', 'rr1', 'rr3', 'rr6', 'rr12', 'rr24']

for var in meteo_vars:
    if var in df_synop.columns:
        df_synop[var] = pd.to_numeric(df_synop[var], errors='coerce')

# Agréger par région et semaine
vars_disponibles = [v for v in meteo_vars if v in df_synop.columns]
agg_dict = {v: 'mean' for v in vars_disponibles}
df_meteo = df_synop.groupby(['region_name', 'week_year'], as_index=False).agg(agg_dict)

print(f"  ✓ {len(df_meteo)} observations agrégées")

# Merger avec test
df_test['region_clean'] = df_test['region_name'].str.upper().str.strip()
df_meteo['region_clean'] = df_meteo['region_name'].str.upper().str.strip()

df_test_merged = df_test.merge(
    df_meteo,
    left_on=['region_clean', 'week'],
    right_on=['region_clean', 'week_year'],
    how='left'
)

print(f"  ✓ Test mergé : {df_test_merged.shape}")

# ============================================================================
# 4. PRÉDIRE
# ============================================================================
print("\n[4/5] Prédictions...")

# Préparer X_test
X_test = df_test_merged[feature_cols].copy()

# Imputer NaN avec les valeurs du train
for col in X_test.columns:
    if X_test[col].isnull().sum() > 0:
        median_val = X_train[col].median()
        X_test[col].fillna(median_val, inplace=True)

# Prédire
predictions = model.predict(X_test)

# Arrondir et s'assurer >= 0
predictions = np.maximum(0, np.round(predictions))

print(f"✓ {len(predictions)} prédictions générées")
print(f"  Min : {predictions.min():.0f}")
print(f"  Max : {predictions.max():.0f}")
print(f"  Moyenne : {predictions.mean():.1f}")

# ============================================================================
# 5. CRÉER SUBMISSION
# ============================================================================
print("\n[5/5] Création du fichier submission...")

submission = pd.DataFrame({
    'Id': df_test['Id'],
    'TauxGrippe': predictions.astype(int)
})

# Sauvegarder
submission.to_csv('submission.csv', index=False)

print("\n" + "="*80)
print("✓ TERMINÉ!")
print("="*80)
print(f"\n📁 Fichier créé : submission.csv")
print(f"   {len(submission)} prédictions")
print(f"\n🔍 Aperçu :")
print(submission.head(10))
print(f"\n📊 Statistiques :")
print(submission['TauxGrippe'].describe())
print("\n✅ Prêt à soumettre sur Kaggle!")
