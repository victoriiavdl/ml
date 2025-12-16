#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRÉPARATION DU TEST SET
Merge test.csv avec données météo 2012-2013
"""

import pandas as pd
import numpy as np
import glob
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("PRÉPARATION DU TEST SET (2012-2013)")
print("="*80)

#%% 1. CHARGEMENT DU TEST
print("\n📥 Chargement du test.csv...")
df_test = pd.read_csv('data_origin/test.csv')
print(f"✓ Test chargé: {df_test.shape}")
print(f"Colonnes: {df_test.columns.tolist()}")
print(f"Périodes: {df_test['week'].min()} à {df_test['week'].max()}")

#%% 2. CHARGEMENT DES DONNÉES MÉTÉO 2012-2013
print("\n📥 Chargement des données météo 2012-2013...")

# Liste des fichiers météo 2012-2013
meteo_files = sorted(glob.glob('DonneesMeteorologiques/DonneesMeteorologiques/synop.201[23]*.csv'))
print(f"✓ {len(meteo_files)} fichiers météo trouvés")

# Charger et concaténer
meteo_list = []
for file in meteo_files:
    try:
        df_meteo_month = pd.read_csv(file, sep=';')
        meteo_list.append(df_meteo_month)
    except Exception as e:
        print(f"⚠️ Erreur lecture {file}: {e}")

df_meteo = pd.concat(meteo_list, ignore_index=True)
print(f"✓ Données météo chargées: {df_meteo.shape}")

#%% 3. NETTOYAGE DES DONNÉES MÉTÉO
print("\n🧹 Nettoyage des données météo...")

# Colonnes nécessaires (même traitement que le train)
colonnes_a_garder = [
    'numer_sta', 'date',
    'tend', 'dd', 'ff', 't', 'td', 'u', 'vv', 'n',
    'nbas', 'hbas', 'pres', 'tn12', 'tx12', 'tminsol',
    'rafper', 'per', 'ht_neige', 'ssfrai', 'perssfrai',
    'rr1', 'rr3', 'rr6', 'rr12', 'rr24'
]

colonnes_disponibles = [col for col in colonnes_a_garder if col in df_meteo.columns]
df_meteo = df_meteo[colonnes_disponibles].copy()

# Convertir date
df_meteo['date'] = pd.to_datetime(df_meteo['date'], format='%Y%m%d%H%M%S', errors='coerce')

# Créer week_year
df_meteo['year'] = df_meteo['date'].dt.year
df_meteo['week'] = df_meteo['date'].dt.isocalendar().week
df_meteo['week_year'] = df_meteo['year'] * 100 + df_meteo['week']

print(f"✓ Nettoyage terminé: {df_meteo.shape}")

#%% 4. CONVERSION DES COLONNES EN NUMÉRIQUE
print("\n🔧 Conversion des colonnes en numérique...")

# Convertir toutes les colonnes (sauf les identifiants) en numérique
numeric_cols = [col for col in df_meteo.columns if col not in ['numer_sta', 'date', 'year', 'week', 'week_year']]
for col in numeric_cols:
    df_meteo[col] = pd.to_numeric(df_meteo[col], errors='coerce')

# Sélectionner uniquement les colonnes numériques pour l'agrégation
numeric_cols_for_agg = df_meteo.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols_for_agg = [col for col in numeric_cols_for_agg if col not in ['year', 'week']]

print(f"✓ {len(numeric_cols_for_agg)} colonnes numériques à agréger")

#%% 5. AGRÉGATION PAR SEMAINE ET STATION
print("\n📊 Agrégation par semaine et station...")

# Grouper par station et semaine (moyenne)
agg_dict = {col: 'mean' for col in numeric_cols_for_agg if col not in ['numer_sta', 'week_year']}

df_meteo_agg = df_meteo.groupby(['numer_sta', 'week_year']).agg(agg_dict).reset_index()
print(f"✓ Agrégation terminée: {df_meteo_agg.shape}")

#%% 6. MAPPING STATION → RÉGION
print("\n🗺️ Mapping station → région...")

# Charger la liste des stations avec région
df_stations = pd.read_csv('data_origin/ListedesStationsMeteo.csv', sep=';')
print(f"✓ {len(df_stations)} stations chargées")

# Mapping station → region_code
station_region_map = df_stations.set_index('ID')['Region'].to_dict()

# Appliquer au météo
df_meteo_agg['region_code'] = df_meteo_agg['numer_sta'].map(station_region_map)

# Supprimer les lignes sans région
df_meteo_agg = df_meteo_agg[df_meteo_agg['region_code'].notna()].copy()
df_meteo_agg['region_code'] = df_meteo_agg['region_code'].astype(int)

print(f"✓ Mapping appliqué: {df_meteo_agg.shape}")

#%% 7. AGRÉGATION PAR RÉGION ET SEMAINE
print("\n📊 Agrégation par région et semaine...")

# Grouper par région et semaine (moyenne des stations)
agg_dict_region = {col: 'mean' for col in df_meteo_agg.columns
                   if col not in ['numer_sta', 'region_code', 'week_year']}

df_meteo_region = df_meteo_agg.groupby(['region_code', 'week_year']).agg(agg_dict_region).reset_index()
print(f"✓ Agrégation par région: {df_meteo_region.shape}")

#%% 8. MERGE TEST + MÉTÉO
print("\n🔗 Merge test + météo...")

# Renommer 'week' en 'week_year' dans le test si nécessaire
if 'week' in df_test.columns and 'week_year' not in df_test.columns:
    df_test['week_year'] = df_test['week']

# Merge
df_test_merged = df_test.merge(
    df_meteo_region,
    on=['region_code', 'week_year'],
    how='left'
)

print(f"✓ Merge terminé: {df_test_merged.shape}")

# Vérifier les valeurs manquantes
missing = df_test_merged.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing) > 0:
    print(f"\n⚠️ Colonnes avec valeurs manquantes:")
    print(missing.head(10))

#%% 9. TRAITEMENT DES VALEURS MANQUANTES
print("\n🔧 Traitement des valeurs manquantes...")

# Supprimer colonnes avec >50% NaN
cols_to_drop = missing[missing / len(df_test_merged) > 0.5].index.tolist()
if len(cols_to_drop) > 0:
    print(f"Suppression de {len(cols_to_drop)} colonnes: {cols_to_drop}")
    df_test_merged = df_test_merged.drop(columns=cols_to_drop)

# Imputer le reste avec la médiane
from sklearn.impute import SimpleImputer
numeric_cols = df_test_merged.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [col for col in numeric_cols if col not in ['Id', 'week_year', 'region_code']]

if df_test_merged[numeric_cols].isnull().sum().sum() > 0:
    imputer = SimpleImputer(strategy='median')
    df_test_merged[numeric_cols] = imputer.fit_transform(df_test_merged[numeric_cols])
    print(f"✓ Valeurs manquantes imputées")

#%% 10. SAUVEGARDE
print("\n💾 Sauvegarde du test set préparé...")

# Sauvegarder dans data_plus/
df_test_merged.to_csv('data_plus/test_synop_merged.csv', index=False)

print(f"\n✅ TEST SET PRÉPARÉ!")
print("="*80)
print(f"Fichier: data_plus/test_synop_merged.csv")
print(f"Shape: {df_test_merged.shape}")
print(f"Colonnes: {len(df_test_merged.columns)}")
print(f"Périodes: {df_test_merged['week_year'].min()} à {df_test_merged['week_year'].max()}")
print(f"\n📋 Aperçu:")
print(df_test_merged.head())
print("\n🎯 Vous pouvez maintenant exécuter KAGGLE_PREDICT.ipynb!")
print("="*80)
