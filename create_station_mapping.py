#!/usr/bin/env python3
"""
Script pour extraire le mapping station → région depuis les données historiques
et créer un fichier de mapping réutilisable
"""

import pandas as pd
import glob
import warnings
warnings.filterwarnings('ignore')

print("Extraction du mapping station → région...")

# Charger les données météo historiques (2004-2011)
meteo_files = sorted(glob.glob('DonneesMeteorologiques/DonneesMeteorologiques/synop.20[0-1][01]*.csv'))
print(f"✓ {len(meteo_files)} fichiers météo trouvés")

# Charger un échantillon
sample = []
for file in meteo_files[:12]:
    try:
        df = pd.read_csv(file, sep=';')
        if 'numer_sta' in df.columns and 'Latitude' in df.columns and 'Longitude' in df.columns:
            sample.append(df[['numer_sta', 'Latitude', 'Longitude']].drop_duplicates())
    except Exception as e:
        print(f"Erreur {file}: {e}")

df_stations = pd.concat(sample, ignore_index=True).drop_duplicates('numer_sta')
print(f"✓ {len(df_stations)} stations uniques trouvées")

# Charger le train merged pour avoir les centroides des régions
df_train = pd.read_csv('data_plus/train_synop_merged_inner.csv')
regions = df_train[['region_code', 'region_name']].drop_duplicates()

print(f"\n✓ {len(regions)} régions dans le train")
print(regions)

# Définition manuelle des régions par coordonnées approximatives (centroides)
region_centroids = {
    11: (48.8566, 2.3522),   # Île-de-France
    21: (49.0000, 4.0000),   # Champagne-Ardenne
    22: (49.6500, 2.3000),   # Picardie
    23: (49.4000, 1.0000),   # Haute-Normandie
    24: (47.5000, 1.5000),   # Centre
    25: (49.0000, -0.5000),  # Basse-Normandie
    26: (47.3000, 4.8000),   # Bourgogne
    31: (50.6300, 3.0600),   # Nord-Pas-de-Calais
    41: (48.7000, 6.2000),   # Lorraine
    42: (48.5800, 7.7500),   # Alsace
    43: (47.2500, 6.0000),   # Franche-Comté
    52: (47.4700, -0.5500),  # Pays de la Loire
    53: (48.2000, -2.9300),  # Bretagne
    54: (45.8300, 0.5000),   # Poitou-Charentes
    72: (44.8400, -0.5800),  # Aquitaine
    73: (43.6000, 1.4400),   # Midi-Pyrénées
    74: (45.8300, 1.2600),   # Limousin
    82: (45.7600, 4.8400),   # Rhône-Alpes
    83: (45.7800, 3.0800),   # Auvergne
    91: (43.6100, 3.8800),   # Languedoc-Roussillon
    93: (43.9400, 6.0700),   # Provence-Alpes-Côte d'Azur
    94: (42.0000, 9.0000),   # Corse
}

def haversine_distance(lat1, lon1, lat2, lon2):
    """Distance en km entre deux points GPS"""
    from math import radians, sin, cos, sqrt, atan2

    R = 6371  # Rayon de la Terre en km

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))

    return R * c

# Mapper chaque station à la région la plus proche
mapping = []
for _, row in df_stations.iterrows():
    station = row['numer_sta']
    lat, lon = row['Latitude'], row['Longitude']

    # Trouver la région la plus proche
    min_dist = float('inf')
    closest_region = None

    for region_code, (r_lat, r_lon) in region_centroids.items():
        dist = haversine_distance(lat, lon, r_lat, r_lon)
        if dist < min_dist:
            min_dist = dist
            closest_region = region_code

    mapping.append({
        'numer_sta': station,
        'region_code': closest_region,
        'distance_km': min_dist
    })

df_mapping = pd.DataFrame(mapping)

# Sauvegarder le mapping
df_mapping.to_csv('station_region_mapping.csv', index=False)

print(f"\n✅ Mapping créé et sauvegardé: station_region_mapping.csv")
print(f"   {len(df_mapping)} stations mappées")
print(f"\nAperçu:")
print(df_mapping.head(10))
print(f"\nDistribution par région:")
print(df_mapping['region_code'].value_counts().sort_index())
