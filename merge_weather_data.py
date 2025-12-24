"""
SCRIPT DE MERGE COMPLET : train.csv + Stations Météo + Données Météo

DÉMARCHE:
=========
1. Charger train.csv (données par région et semaine avec TauxGrippe)
2. Charger ListedesStationsMeteo.csv (liste des stations avec coordonnées)
3. Charger tous les fichiers synop (données météo par station et date)
4. Mapper chaque région à ses stations météo représentatives
5. Agréger les données météo par région et par semaine
6. Merger train.csv avec les données météo agrégées

PROBLÈMES À RÉSOUDRE:
=====================
- train.csv : données par RÉGION et SEMAINE
- synop : données par STATION et DATE (horaire/journalier)
- Il faut :
  a) Convertir les dates des fichiers synop en semaines
  b) Mapper les stations aux régions
  c) Agréger les données météo (moyenne par région/semaine)
"""

import pandas as pd
import numpy as np
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("MERGE COMPLET : TRAIN + STATIONS MÉTÉO + DONNÉES MÉTÉO")
print("="*80)

# ============================================================================
# ÉTAPE 1 : CHARGER LES DONNÉES DE BASE
# ============================================================================
print("\n[1/6] Chargement des données de base...")

# Train.csv
df_train = pd.read_csv('data_origin/train.csv')
print(f"✓ train.csv : {df_train.shape}")
print(f"  Colonnes : {df_train.columns.tolist()}")
print(f"  Exemple :\n{df_train.head(3)}")

# Liste des stations météo
df_stations = pd.read_csv('data_origin/ListedesStationsMeteo.csv', sep=';')
print(f"\n✓ ListedesStationsMeteo.csv : {df_stations.shape}")
print(f"  Colonnes : {df_stations.columns.tolist()}")
print(f"  Exemple :\n{df_stations.head(3)}")

# ============================================================================
# ÉTAPE 2 : MAPPER RÉGIONS <-> STATIONS MÉTÉO
# ============================================================================
print("\n[2/6] Mapping Régions -> Stations Météo...")

# APPROCHE : Chaque région a des stations météo principales
# On va créer un mapping manuel basé sur la géographie française
# Sources : connaissance des régions françaises (2011) et stations principales

REGION_STATION_MAPPING = {
    'ALSACE': ['07190', '07280'],  # Strasbourg
    'AQUITAINE': ['07510', '07630'],  # Bordeaux, Biarritz
    'AUVERGNE': ['07460', '07380'],  # Clermont-Ferrand, Vichy
    'BASSE-NORMANDIE': ['07027', '07139'],  # Caen
    'BOURGOGNE': ['07280', '07255'],  # Dijon
    'BRETAGNE': ['07110', '07117', '07130'],  # Brest, Rennes
    'CENTRE': ['07255', '07149'],  # Orléans, Tours
    'CHAMPAGNE-ARDENNE': ['07072', '07168'],  # Reims
    'CORSE': ['07761', '07790'],  # Ajaccio, Bastia
    'FRANCHE-COMTE': ['07299', '07280'],  # Besançon
    'HAUTE-NORMANDIE': ['07037', '07020'],  # Rouen, Le Havre
    'ILE-DE-FRANCE': ['07150', '07149'],  # Paris-Montsouris, Orly
    'LANGUEDOC-ROUSSILLON': ['07630', '07643'],  # Montpellier, Perpignan
    'LIMOUSIN': ['07434', '07335'],  # Limoges
    'LORRAINE': ['07090', '07180'],  # Nancy, Metz
    'MIDI-PYRENEES': ['07630', '07627'],  # Toulouse
    'NORD-PAS-DE-CALAIS': ['07005', '07015'],  # Lille
    'PAYS DE LA LOIRE': ['07222', '07130'],  # Nantes
    'PICARDIE': ['07005', '07015'],  # Amiens
    'POITOU-CHARENTES': ['07335', '07255'],  # Poitiers, La Rochelle
    'PROVENCE-ALPES-COTE D\'AZUR': ['07650', '07690'],  # Marseille, Nice
    'RHONE-ALPES': ['07481', '07482'],  # Lyon
}

# Créer une table de mapping station -> région
station_region_map = []
for region, stations in REGION_STATION_MAPPING.items():
    for station in stations:
        station_region_map.append({
            'numer_sta': station,
            'region_name': region
        })

df_station_region = pd.DataFrame(station_region_map)
print(f"✓ Mapping créé : {len(df_station_region)} associations station-région")
print(f"  Régions couvertes : {df_station_region['region_name'].nunique()}")
print(f"  Exemple :\n{df_station_region.head(10)}")

# ============================================================================
# ÉTAPE 3 : CHARGER LES DONNÉES MÉTÉO (SYNOP)
# ============================================================================
print("\n[3/6] Chargement des données météo (synop)...")

# Lister tous les fichiers synop
synop_files = sorted(glob.glob('DonneesMeteorologiques/DonneesMeteorologiques/synop.*.csv'))
print(f"✓ Fichiers synop trouvés : {len(synop_files)}")
print(f"  Premier : {synop_files[0]}")
print(f"  Dernier : {synop_files[-1]}")

# Charger tous les fichiers synop
# ATTENTION: C'est volumineux (~550MB), on va filtrer tout de suite par les stations d'intérêt
stations_of_interest = df_station_region['numer_sta'].unique().tolist()
print(f"  Stations d'intérêt : {len(stations_of_interest)}")

synop_data_list = []
for i, file in enumerate(synop_files):
    if i % 12 == 0:  # Afficher progression tous les 12 mois
        print(f"  Chargement : {file.split('/')[-1]}...")

    try:
        df_synop = pd.read_csv(file, sep=';', low_memory=False)
        # Filtrer uniquement les stations d'intérêt
        df_synop = df_synop[df_synop['numer_sta'].isin(stations_of_interest)]
        synop_data_list.append(df_synop)
    except Exception as e:
        print(f"  ⚠ Erreur lecture {file} : {e}")

# Concaténer tous les fichiers
df_synop_all = pd.concat(synop_data_list, ignore_index=True)
print(f"\n✓ Données synop chargées : {df_synop_all.shape}")
print(f"  Colonnes : {len(df_synop_all.columns)}")
print(f"  Période : {df_synop_all['date'].min()} -> {df_synop_all['date'].max()}")

# ============================================================================
# ÉTAPE 4 : CONVERTIR LES DATES EN SEMAINES
# ============================================================================
print("\n[4/6] Conversion des dates en semaines...")

# Convertir la date en format datetime
df_synop_all['date'] = pd.to_datetime(df_synop_all['date'], format='%Y%m%d%H%M%S', errors='coerce')

# Extraire année et semaine
df_synop_all['year'] = df_synop_all['date'].dt.isocalendar().year
df_synop_all['week'] = df_synop_all['date'].dt.isocalendar().week

# Créer le code semaine au format AAAASS (comme dans train.csv)
df_synop_all['week_year'] = df_synop_all['year'] * 100 + df_synop_all['week']

print(f"✓ Conversion effectuée")
print(f"  Exemple week_year : {df_synop_all['week_year'].head().tolist()}")
print(f"  Semaines uniques : {df_synop_all['week_year'].nunique()}")

# ============================================================================
# ÉTAPE 5 : AGRÉGER LES DONNÉES MÉTÉO PAR RÉGION ET SEMAINE
# ============================================================================
print("\n[5/6] Agrégation des données météo par région et semaine...")

# Merger avec le mapping station->région
df_synop_all = df_synop_all.merge(df_station_region, on='numer_sta', how='inner')
print(f"✓ Merge avec mapping : {df_synop_all.shape}")

# Sélectionner les variables météo importantes
# Basé sur l'analyse du notebook NETTOYAGE_DONNEES.ipynb
meteo_vars = [
    'tend', 'dd', 'ff', 't', 'td', 'u', 'vv', 'n', 'nbas', 'hbas',
    'pres', 'niv_bar', 'geop', 'tend24', 'tn12', 'tn24', 'tx12', 'tx24',
    'tminsol', 'tw', 'raf10', 'rafper', 'per', 'ht_neige', 'ssfrai',
    'perssfrai', 'rr1', 'rr3', 'rr6', 'rr12', 'rr24'
]

# Vérifier quelles variables existent
meteo_vars_available = [v for v in meteo_vars if v in df_synop_all.columns]
print(f"  Variables météo disponibles : {len(meteo_vars_available)}/{len(meteo_vars)}")

# Convertir les variables météo en numérique (remplacer 'mq' par NaN)
for var in meteo_vars_available:
    df_synop_all[var] = pd.to_numeric(df_synop_all[var], errors='coerce')

# Agréger par région et semaine (moyenne)
agg_dict = {var: 'mean' for var in meteo_vars_available}
df_meteo_agg = df_synop_all.groupby(['region_name', 'week_year'], as_index=False).agg(agg_dict)

print(f"✓ Agrégation effectuée : {df_meteo_agg.shape}")
print(f"  Exemple :\n{df_meteo_agg.head(3)}")

# ============================================================================
# ÉTAPE 6 : MERGER AVEC TRAIN.CSV
# ============================================================================
print("\n[6/6] Merge final avec train.csv...")

# Normaliser les noms de régions pour le merge
df_train['region_name_clean'] = df_train['region_name'].str.upper().str.strip()
df_meteo_agg['region_name_clean'] = df_meteo_agg['region_name'].str.upper().str.strip()

# Merger
df_final = df_train.merge(
    df_meteo_agg,
    left_on=['region_name_clean', 'week'],
    right_on=['region_name_clean', 'week_year'],
    how='inner'
)

print(f"✓ Merge effectué : {df_final.shape}")
print(f"  Colonnes : {len(df_final.columns)}")

# Nettoyer les colonnes dupliquées
cols_to_drop = ['region_name_clean', 'week_year']
df_final = df_final.drop(columns=[c for c in cols_to_drop if c in df_final.columns])

# ============================================================================
# SAUVEGARDE
# ============================================================================
output_file = 'data_plus/train_weather_merged_complete.csv'
df_final.to_csv(output_file, index=False)

print(f"\n{'='*80}")
print("MERGE TERMINÉ AVEC SUCCÈS!")
print("="*80)
print(f"\nFichier généré : {output_file}")
print(f"  Dimensions : {df_final.shape}")
print(f"  Observations : {len(df_final)}")
print(f"  Variables météo ajoutées : {len(meteo_vars_available)}")
print(f"\nAperçu final :")
print(df_final.head(3))

print(f"\nColonnes finales ({len(df_final.columns)}) :")
print(df_final.columns.tolist())

print(f"\n✓ Le fichier est prêt pour l'analyse et la modélisation!")
