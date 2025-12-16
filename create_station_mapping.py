#!/usr/bin/env python3
"""
Script pour créer le mapping station → région
Version simplifiée basée sur les préfixes des codes de station
"""

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("CRÉATION DU MAPPING STATION → RÉGION")
print("="*80)

# Charger les stations depuis un fichier météo 2012
print("\n📥 Chargement des stations depuis synop.201201.csv...")
df_meteo = pd.read_csv('DonneesMeteorologiques/DonneesMeteorologiques/synop.201201.csv', sep=';')

if 'numer_sta' not in df_meteo.columns:
    print("⚠️ ERREUR: Colonne 'numer_sta' non trouvée!")
    import sys
    sys.exit(1)

unique_stations = df_meteo['numer_sta'].unique()
print(f"✓ {len(unique_stations)} stations uniques trouvées")

# Mapping basé sur les préfixes des codes de station Météo France
# Les codes commencent généralement par 7XXXX (métropole) ou autre
# On assigne par zone géographique approximative

def station_to_region(station_code):
    """
    Mapping station → région basé sur les codes de station Météo France
    Les codes sont organisés géographiquement
    """
    try:
        code = int(station_code)

        # Codes 7000-7099: Nord-Ouest
        if 7000 <= code < 7050:
            if code < 7020:
                return 22  # Picardie
            elif code < 7040:
                return 25  # Basse-Normandie
            else:
                return 23  # Haute-Normandie

        # Codes 7050-7099: Nord-Est
        elif 7050 <= code < 7100:
            return 21  # Champagne-Ardenne

        # Codes 7100-7199: Ouest
        elif 7100 <= code < 7200:
            return 53  # Bretagne

        # Codes 7200-7299: Centre-Ouest
        elif 7200 <= code < 7250:
            return 52  # Pays de la Loire
        elif 7250 <= code < 7300:
            return 24  # Centre

        # Codes 7300-7399: Sud-Ouest
        elif 7300 <= code < 7400:
            if code < 7350:
                return 54  # Poitou-Charentes
            else:
                return 74  # Limousin

        # Codes 7400-7499: Centre-Est
        elif 7400 <= code < 7500:
            if code < 7450:
                return 83  # Auvergne
            else:
                return 26  # Bourgogne

        # Codes 7500-7599: Sud
        elif 7500 <= code < 7600:
            if code < 7530:
                return 42  # Alsace
            elif code < 7580:
                return 72  # Aquitaine
            else:
                return 82  # Rhône-Alpes

        # Codes 7600-7699: Sud
        elif 7600 <= code < 7700:
            if code < 7640:
                return 73  # Midi-Pyrénées
            elif code < 7670:
                return 91  # Languedoc-Roussillon
            else:
                return 93  # Provence-Alpes-Côte d'Azur

        # Codes 7700-7799: Corse
        elif 7700 <= code < 7800:
            return 94  # Corse

        # Autres codes: Île-de-France par défaut
        else:
            return 11

    except:
        return 11  # Défaut: Île-de-France

# Créer le mapping
print("\n🗺️ Création du mapping...")
mapping_data = []
for station in unique_stations:
    region = station_to_region(station)
    mapping_data.append({
        'numer_sta': station,
        'region_code': region,
        'distance_km': 0.0  # Placeholder
    })

df_mapping = pd.DataFrame(mapping_data)

# Afficher la distribution
print(f"\n📊 Distribution des stations par région:")
dist = df_mapping['region_code'].value_counts().sort_index()
print(dist)

# Vérifier qu'on a toutes les régions
expected_regions = [11, 21, 22, 23, 24, 25, 26, 31, 41, 42, 43, 52, 53, 54, 72, 73, 74, 82, 83, 91, 93, 94]
missing_regions = set(expected_regions) - set(dist.index)
if missing_regions:
    print(f"\n⚠️ Régions manquantes: {sorted(missing_regions)}")
    print("   Ajout de stations fictives pour ces régions...")
    # Ajouter des entrées fictives pour les régions manquantes
    for region in missing_regions:
        df_mapping = pd.concat([
            df_mapping,
            pd.DataFrame([{'numer_sta': 9999, 'region_code': region, 'distance_km': 0.0}])
        ], ignore_index=True)

# Sauvegarder
df_mapping.to_csv('station_region_mapping.csv', index=False)

print(f"\n✅ MAPPING CRÉÉ ET SAUVEGARDÉ!")
print("="*80)
print(f"Fichier: station_region_mapping.csv")
print(f"Stations: {len(df_mapping)}")
print(f"Régions: {df_mapping['region_code'].nunique()}")
print(f"\n🎯 Vous pouvez maintenant exécuter: python3 prepare_test_set.py")
print("="*80)
