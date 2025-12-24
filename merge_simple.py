#!/usr/bin/env python3
"""Script simple pour merger train.csv avec les données météo"""

import pandas as pd
import glob
import sys

print("="*80)
print("MERGE DONNÉES MÉTÉO - VERSION SIMPLE")
print("="*80)

# ============================================================================
# 1. CHARGER TRAIN.CSV
# ============================================================================
print("\n[1/5] Chargement train.csv...")
df_train = pd.read_csv('data_origin/train.csv')
print(f"✓ {df_train.shape[0]} lignes, {df_train.shape[1]} colonnes")

# ============================================================================
# 2. MAPPING RÉGIONS -> STATIONS (SIMPLIFIÉ)
# ============================================================================
print("\n[2/5] Création du mapping régions->stations...")

# Mapping des régions aux stations principales
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

# Créer la table de correspondance
rows = []
for region, stations in MAPPING.items():
    for station in stations:
        rows.append({'numer_sta': station, 'region_name': region})
df_mapping = pd.DataFrame(rows)
print(f"✓ {len(df_mapping)} associations créées")

# ============================================================================
# 3. CHARGER LES FICHIERS SYNOP
# ============================================================================
print("\n[3/5] Chargement des fichiers synop...")

synop_files = sorted(glob.glob('DonneesMeteorologiques/DonneesMeteorologiques/synop.*.csv'))
if len(synop_files) == 0:
    print("❌ ERREUR: Aucun fichier synop trouvé!")
    sys.exit(1)

print(f"  {len(synop_files)} fichiers trouvés")

# Liste des stations à charger
stations_list = df_mapping['numer_sta'].unique().tolist()
print(f"  {len(stations_list)} stations à chercher")

# Charger tous les fichiers
all_data = []
for i, fpath in enumerate(synop_files):
    if i % 12 == 0:
        print(f"  Fichier {i+1}/{len(synop_files)}...")

    try:
        df = pd.read_csv(fpath, sep=';', low_memory=False)
        # Filtrer par stations
        df = df[df['numer_sta'].isin(stations_list)]
        if len(df) > 0:
            all_data.append(df)
    except Exception as e:
        print(f"  ⚠ Erreur {fpath}: {e}")

if len(all_data) == 0:
    print("❌ ERREUR: Aucune donnée chargée!")
    sys.exit(1)

df_synop = pd.concat(all_data, ignore_index=True)
print(f"✓ {len(df_synop)} observations chargées")

# ============================================================================
# 4. CONVERTIR DATES EN SEMAINES ET AGRÉGER
# ============================================================================
print("\n[4/5] Agrégation par région et semaine...")

# Convertir date
df_synop['date'] = pd.to_datetime(df_synop['date'], format='%Y%m%d%H%M%S', errors='coerce')
df_synop['year'] = df_synop['date'].dt.isocalendar().year
df_synop['week'] = df_synop['date'].dt.isocalendar().week
df_synop['week_year'] = df_synop['year'] * 100 + df_synop['week']

# Merger avec mapping
df_synop = df_synop.merge(df_mapping, on='numer_sta', how='inner')

# Variables météo à garder
meteo_vars = ['tend', 'dd', 'ff', 't', 'td', 'u', 'vv', 'n', 'nbas', 'hbas',
              'pres', 'tn12', 'tx12', 'tminsol', 'rafper', 'per',
              'ht_neige', 'ssfrai', 'perssfrai', 'rr1', 'rr3', 'rr6', 'rr12', 'rr24']

# Convertir en numérique
for var in meteo_vars:
    if var in df_synop.columns:
        df_synop[var] = pd.to_numeric(df_synop[var], errors='coerce')

# Agréger par région et semaine
vars_disponibles = [v for v in meteo_vars if v in df_synop.columns]
agg_dict = {v: 'mean' for v in vars_disponibles}
df_meteo = df_synop.groupby(['region_name', 'week_year'], as_index=False).agg(agg_dict)

print(f"✓ {len(df_meteo)} observations (région x semaine)")
print(f"  {len(vars_disponibles)} variables météo")

# ============================================================================
# 5. MERGE AVEC TRAIN.CSV
# ============================================================================
print("\n[5/5] Merge final...")

# Normaliser les noms
df_train['region_clean'] = df_train['region_name'].str.upper().str.strip()
df_meteo['region_clean'] = df_meteo['region_name'].str.upper().str.strip()

# Merger
df_final = df_train.merge(
    df_meteo,
    left_on=['region_clean', 'week'],
    right_on=['region_clean', 'week_year'],
    how='inner'
)

# Nettoyer
df_final = df_final.drop(columns=['region_clean', 'week_year', 'region_name_y'], errors='ignore')
if 'region_name_x' in df_final.columns:
    df_final = df_final.rename(columns={'region_name_x': 'region_name'})

print(f"✓ {len(df_final)} lignes dans le résultat final")
print(f"  Couverture: {len(df_final)/len(df_train)*100:.1f}%")

# ============================================================================
# SAUVEGARDE
# ============================================================================
output = 'data_plus/train_weather_merged_complete.csv'
df_final.to_csv(output, index=False)

print(f"\n{'='*80}")
print("✓ TERMINÉ!")
print("="*80)
print(f"\nFichier créé: {output}")
print(f"Dimensions: {df_final.shape}")
print(f"Colonnes: {list(df_final.columns)}")
print(f"\nAperçu:")
print(df_final.head(3))
