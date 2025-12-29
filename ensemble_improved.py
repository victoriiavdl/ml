"""
================================================================================
🦠 ENSEMBLE AMÉLIORÉ: CatBoost + XGBoost + LightGBM
================================================================================
Améliorations majeures:
  1. VRAIS LAGS temporels par région (pas juste moyennes historiques)
  2. Rolling features (moyennes mobiles 2-4 semaines)
  3. Target encoding de région (mean/std historique)
  4. Features Google améliorées (diff, rolling, interactions)
  5. Split temporel propre (comme Kaggle: 2004-2010 train, 2011 val)
  6. Stacking au lieu d'un simple blend
================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor, Pool
from xgboost import XGBRegressor
import lightgbm as lgb

np.random.seed(42)

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# Load
# -----------------------------------------------------------------------------
print("=" * 80)
print("🦠 ENSEMBLE AMÉLIORÉ: CatBoost + XGBoost + LightGBM")
print("=" * 80)

train = pd.read_csv("data_plus/train_synop_cleaned_complet.csv")
print(f"✓ Train data loaded: {train.shape}")

# Convertir date
train['date'] = pd.to_datetime(train['date'])
train = train.sort_values(['region_code', 'date']).reset_index(drop=True)

# Extraire year/week
train["year"] = train["date"].dt.year
train["week_num"] = train["date"].dt.isocalendar().week

# -----------------------------------------------------------------------------
# Feature engineering AMÉLIORÉ
# -----------------------------------------------------------------------------
def create_features_improved(df: pd.DataFrame,
                             hist_data: pd.DataFrame,
                             is_train: bool = True) -> pd.DataFrame:
    """
    Feature engineering avec VRAIS lags et rolling features

    Args:
        df: données à transformer
        hist_data: données historiques (pour éviter le leakage)
        is_train: si True, on peut calculer des lags sur df lui-même
    """
    df = df.copy()
    hist = hist_data.copy()

    # ==========================================================================
    # 1. FEATURES TEMPORELLES (Fourier + phases)
    # ==========================================================================
    # Fourier (3 harmoniques pour capturer saisonnalité)
    for k in (1, 2, 3):
        df[f"sin_{k}"] = np.sin(2 * np.pi * k * df["week_num"] / 52)
        df[f"cos_{k}"] = np.cos(2 * np.pi * k * df["week_num"] / 52)

    # Phases épidémie
    df["is_peak"] = df["week_num"].isin([4, 5, 6, 7, 8]).astype(int)
    df["is_rise"] = df["week_num"].isin(list(range(48, 53)) + [1, 2, 3]).astype(int)
    df["is_low"]  = df["week_num"].isin(range(18, 45)).astype(int)

    # Distance au pic
    df["dist_peak"] = df["week_num"].apply(lambda w: min(abs(w - 5), abs(w - 57), abs(w + 47)))
    df["peak_intensity"] = np.exp(-df["dist_peak"] / 5)

    # ==========================================================================
    # 2. STATS HISTORIQUES PAR SEMAINE (nationales)
    # ==========================================================================
    agg_w = hist.groupby("week_num")["TauxGrippe"].agg(["mean", "std", "median"]).reset_index()
    agg_w.columns = ["week_num", "w_mean", "w_std", "w_median"]
    df = df.merge(agg_w, on="week_num", how="left")

    # ==========================================================================
    # 3. STATS HISTORIQUES PAR RÉGION (TARGET ENCODING)
    # ==========================================================================
    # Moyenne et std du TauxGrippe par région (historique)
    agg_region = hist.groupby("region_code")["TauxGrippe"].agg(["mean", "std"]).reset_index()
    agg_region.columns = ["region_code", "region_mean", "region_std"]
    df = df.merge(agg_region, on="region_code", how="left")

    # Moyenne par région x semaine (pattern saisonnier régional)
    agg_reg_week = hist.groupby(["region_code", "week_num"])["TauxGrippe"].mean().reset_index()
    agg_reg_week.columns = ["region_code", "week_num", "region_week_mean"]
    df = df.merge(agg_reg_week, on=["region_code", "week_num"], how="left")

    # Moyenne par région x mois
    hist_with_month = hist.copy()
    hist_with_month["month"] = pd.to_datetime(hist_with_month["date"]).dt.month
    agg_reg_month = hist_with_month.groupby(["region_code", "month"])["TauxGrippe"].mean().reset_index()
    agg_reg_month.columns = ["region_code", "month", "region_month_mean"]
    df["month"] = pd.to_datetime(df["date"]).dt.month
    df = df.merge(agg_reg_month, on=["region_code", "month"], how="left")

    # ==========================================================================
    # 4. LAG FEATURES SAISONNIERS (moyennes historiques)
    # ==========================================================================
    # lag1_seasonal: moyenne historique de la semaine précédente
    wmean_map = hist.groupby("week_num")["TauxGrippe"].mean()
    wmean_by_wn = wmean_map.reindex(range(1, 53))
    lag_wmean_by_wn = wmean_by_wn.shift(1)
    lag_wmean_by_wn.iloc[0] = wmean_by_wn.iloc[-1]  # wrap-around
    df["lag1_seasonal"] = df["week_num"].map(lag_wmean_by_wn)
    df["lag1_seasonal"] = df["lag1_seasonal"].fillna(df["w_mean"])

    # ==========================================================================
    # 5. VRAIS LAGS TEMPORELS PAR RÉGION (IMPORTANT!)
    # ==========================================================================
    if is_train:
        # Pour train/val: on peut calculer les lags sur df lui-même
        df = df.sort_values(['region_code', 'date']).reset_index(drop=True)
        for lag in [1, 2, 3, 4]:
            df[f'lag{lag}_real'] = df.groupby('region_code')['TauxGrippe'].shift(lag)
    else:
        # Pour test: on utilise les dernières valeurs du train
        # NOTE: il faudra les passer en argument pour le test
        # Pour l'instant on met NaN, à gérer plus tard
        for lag in [1, 2, 3, 4]:
            df[f'lag{lag}_real'] = np.nan

    # ==========================================================================
    # 6. ROLLING FEATURES (moyennes mobiles) - SI TRAIN
    # ==========================================================================
    if is_train:
        for window in [2, 3, 4]:
            df[f'rolling_mean_{window}w'] = df.groupby('region_code')['TauxGrippe'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )
    else:
        for window in [2, 3, 4]:
            df[f'rolling_mean_{window}w'] = np.nan

    # ==========================================================================
    # 7. GOOGLE TRENDS (si disponible)
    # ==========================================================================
    google_cols = [c for c in df.columns if 'google' in c.lower()]
    if len(google_cols) > 0:
        # Utiliser la meilleure colonne Google (filtre2 si dispo)
        google_col = 'google_grippe_filtre2' if 'google_grippe_filtre2' in df.columns else google_cols[0]
        df[google_col] = df[google_col].fillna(0)

        # Log transform
        df["google_log"] = np.log1p(df[google_col])

        # Anomaly (écart à la moyenne)
        google_mean = hist[google_col].mean() if google_col in hist.columns else df[google_col].mean()
        df["google_anomaly"] = df[google_col] / (google_mean + 1)

        # Diff (variation)
        if is_train:
            df["google_diff"] = df.groupby('region_code')[google_col].diff()
        else:
            df["google_diff"] = 0

        # Rolling Google (2-4 semaines)
        if is_train:
            for window in [2, 3, 4]:
                df[f'google_roll_{window}w'] = df.groupby('region_code')[google_col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
        else:
            for window in [2, 3, 4]:
                df[f'google_roll_{window}w'] = df[google_col]

        # Interactions Google
        df["google_x_w"] = df["google_log"] * df["w_mean"]
        df["google_x_region"] = df["google_log"] * df["region_mean"]
        df["google_x_lag1"] = df["google_log"] * df["lag1_seasonal"]
    else:
        # Pas de Google trends
        df["google_log"] = 0
        df["google_anomaly"] = 0
        df["google_diff"] = 0
        df["google_x_w"] = 0
        df["google_x_region"] = 0
        df["google_x_lag1"] = 0
        for window in [2, 3, 4]:
            df[f'google_roll_{window}w'] = 0

    # ==========================================================================
    # 8. MÉTÉO (si disponible)
    # ==========================================================================
    if "t" in df.columns:
        df["t"] = df["t"].fillna(df["t"].median() if df["t"].notna().any() else 0)
        df["cold"] = np.clip(8 - df["t"], 0, None)

        # Rolling température
        if is_train:
            for window in [2, 4]:
                df[f't_roll_{window}w'] = df.groupby('region_code')['t'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
        else:
            for window in [2, 4]:
                df[f't_roll_{window}w'] = df["t"]

        # Interaction température x semaine
        df["cold_x_w"] = df["cold"] * df["w_mean"]
    else:
        df["t"] = 0
        df["cold"] = 0
        df["cold_x_w"] = 0
        for window in [2, 4]:
            df[f't_roll_{window}w'] = 0

    if "rr1" in df.columns:
        df["rr1"] = df["rr1"].fillna(0)
    else:
        df["rr1"] = 0

    # Region encoding (pour XGB/LGB)
    if "region_code" in df.columns:
        df["region_encoded"] = pd.factorize(df["region_code"])[0]

    # ==========================================================================
    # 9. IMPUTATION FINALE
    # ==========================================================================
    for col in df.columns:
        if df[col].dtype in ["float64", "int64"] and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)

    return df


# -----------------------------------------------------------------------------
# Split TEMPOREL (comme Kaggle)
# -----------------------------------------------------------------------------
print("\n🔧 Split temporel (2004-2010 train, 2011 val)...")

# Train: 2004-2010, Val: 2011
train_data = train[train["year"] <= 2010].copy()
val_data   = train[train["year"] == 2011].copy()

print(f"   Train: {len(train_data)} obs ({train_data['date'].min()} → {train_data['date'].max()})")
print(f"   Val:   {len(val_data)} obs ({val_data['date'].min()} → {val_data['date'].max()})")

# Feature engineering
print("\n🔧 Feature engineering (avec VRAIS lags et rolling)...")
train_feat = create_features_improved(train_data, train_data, is_train=True)
val_feat   = create_features_improved(val_data, train_data, is_train=True)  # no leakage
full_feat  = create_features_improved(train, train, is_train=True)

# -----------------------------------------------------------------------------
# Feature list
# -----------------------------------------------------------------------------
exclude = [
    "Id", "week", "region_name", "TauxGrippe", "franche_comte_impute",
    "year", "annee", "mois", "region_code", "date", "saison", "week_year",
    "day_of_year", "month"
]
features = [c for c in train_feat.columns if c not in exclude and train_feat[c].dtype in ["float64", "int64"]]
print(f"\n   Features totales: {len(features)}")

# Vérifier qu'il n'y a pas de NaN
print(f"   NaN in train_feat: {train_feat[features].isna().sum().sum()}")
print(f"   NaN in val_feat: {val_feat[features].isna().sum().sum()}")

# CatBoost uses region_code as categorical
cat_features_list = ["region_code"] if "region_code" in train_feat.columns else []

X_train = train_feat[features].copy()
X_val   = val_feat[features].copy()
X_full  = full_feat[features].copy()

y_train = train_feat["TauxGrippe"]
y_val   = val_feat["TauxGrippe"]
y_full  = full_feat["TauxGrippe"]

# -----------------------------------------------------------------------------
# 1) CatBoost
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("CATBOOST")
print("=" * 80)

cat = CatBoostRegressor(
    iterations=3000,
    learning_rate=0.03,
    depth=4,
    l2_leaf_reg=10,
    min_data_in_leaf=50,
    random_strength=1.0,
    bagging_temperature=1.0,
    border_count=64,
    random_seed=42,
    verbose=0,
    early_stopping_rounds=200,
    use_best_model=True
)

cat.fit(X_train, y_train, eval_set=(X_val, y_val))
pred_val_cat = np.clip(cat.predict(X_val), 0, None)
print(f"🏁 CatBoost Val RMSE: {rmse(y_val, pred_val_cat):.2f}")

# Fit final (use best iteration from validation)
best_iter = cat.get_best_iteration()
cat_final = CatBoostRegressor(
    iterations=best_iter,
    learning_rate=0.03,
    depth=4,
    l2_leaf_reg=10,
    min_data_in_leaf=50,
    random_strength=1.0,
    bagging_temperature=1.0,
    border_count=64,
    random_seed=42,
    verbose=0
)
cat_final.fit(X_full, y_full)

# -----------------------------------------------------------------------------
# 2) XGBoost
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("XGBOOST")
print("=" * 80)

xgb = XGBRegressor(
    n_estimators=5000,
    learning_rate=0.01,
    max_depth=5,
    min_child_weight=30,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=10.0,
    gamma=0.0,
    objective="reg:squarederror",
    random_state=42,
    tree_method="hist"
)

xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
pred_val_xgb = np.clip(xgb.predict(X_val), 0, None)
print(f"🏁 XGBoost Val RMSE: {rmse(y_val, pred_val_xgb):.2f}")

# Fit final (use best iteration from validation)
best_iter_xgb = xgb.best_iteration if hasattr(xgb, 'best_iteration') else 5000
xgb_final = XGBRegressor(
    n_estimators=best_iter_xgb,
    learning_rate=0.01,
    max_depth=5,
    min_child_weight=30,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=10.0,
    gamma=0.0,
    objective="reg:squarederror",
    random_state=42,
    tree_method="hist"
)
xgb_final.fit(X_full, y_full, verbose=False)

# -----------------------------------------------------------------------------
# 3) LightGBM
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("LIGHTGBM")
print("=" * 80)

lgbm = lgb.LGBMRegressor(
    n_estimators=5000,
    learning_rate=0.01,
    max_depth=5,
    min_child_samples=30,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=10.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
pred_val_lgb = np.clip(lgbm.predict(X_val), 0, None)
print(f"🏁 LightGBM Val RMSE: {rmse(y_val, pred_val_lgb):.2f}")

# Fit final (use best iteration from validation)
best_iter_lgb = lgbm.best_iteration_ if hasattr(lgbm, 'best_iteration_') else 5000
lgbm_final = lgb.LGBMRegressor(
    n_estimators=best_iter_lgb,
    learning_rate=0.01,
    max_depth=5,
    min_child_samples=30,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=10.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgbm_final.fit(X_full, y_full)

# -----------------------------------------------------------------------------
# 4) STACKING (meta-model)
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STACKING (Ridge meta-model)")
print("=" * 80)

# Créer les features de niveau 1 (prédictions des modèles de base)
meta_train = np.column_stack([pred_val_cat, pred_val_xgb, pred_val_lgb])

# Entraîner un Ridge comme meta-model
meta_model = Ridge(alpha=1.0, random_state=42)
meta_model.fit(meta_train, y_val)

# Prédiction meta
pred_val_meta = np.clip(meta_model.predict(meta_train), 0, None)
print(f"🏁 Stacking Val RMSE: {rmse(y_val, pred_val_meta):.2f}")

# Coefficients
print(f"   Poids: Cat={meta_model.coef_[0]:.3f}, XGB={meta_model.coef_[1]:.3f}, LGB={meta_model.coef_[2]:.3f}")

# -----------------------------------------------------------------------------
# 5) Simple weighted ensemble (backup)
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("ENSEMBLE WEIGHTED (grid search)")
print("=" * 80)

best_w = (0.33, 0.33, 0.34)
best_rmse_ens = float("inf")

for w_cat in np.arange(0.0, 1.01, 0.1):
    for w_xgb in np.arange(0.0, 1.01 - w_cat, 0.1):
        w_lgb = 1.0 - w_cat - w_xgb
        if w_lgb < 0:
            continue

        blend = w_cat * pred_val_cat + w_xgb * pred_val_xgb + w_lgb * pred_val_lgb
        sc = rmse(y_val, blend)

        if sc < best_rmse_ens:
            best_rmse_ens = sc
            best_w = (w_cat, w_xgb, w_lgb)

print(f"🏆 Best: w_cat={best_w[0]:.2f}, w_xgb={best_w[1]:.2f}, w_lgb={best_w[2]:.2f} → RMSE={best_rmse_ens:.2f}")

# -----------------------------------------------------------------------------
# COMPARAISON
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("RÉSULTATS VALIDATION")
print("=" * 80)
print(f"   CatBoost:  {rmse(y_val, pred_val_cat):.2f}")
print(f"   XGBoost:   {rmse(y_val, pred_val_xgb):.2f}")
print(f"   LightGBM:  {rmse(y_val, pred_val_lgb):.2f}")
print(f"   Weighted:  {best_rmse_ens:.2f}")
print(f"   Stacking:  {rmse(y_val, pred_val_meta):.2f}")

print("\n💡 Meilleur modèle pour validation: ", end="")
best_model = min([
    ("CatBoost", rmse(y_val, pred_val_cat)),
    ("XGBoost", rmse(y_val, pred_val_xgb)),
    ("LightGBM", rmse(y_val, pred_val_lgb)),
    ("Weighted", best_rmse_ens),
    ("Stacking", rmse(y_val, pred_val_meta))
], key=lambda x: x[1])
print(f"{best_model[0]} (RMSE={best_model[1]:.2f})")

# -----------------------------------------------------------------------------
# TEST SET PREDICTIONS
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("PRÉDICTIONS TEST SET")
print("=" * 80)

# Check if test file exists
import os
test_file = "data_plus/test_synop_cleaned_complet.csv"
if not os.path.exists(test_file):
    print(f"⚠️  Fichier test non trouvé: {test_file}")
    print("   Création des submissions ignorée.")
    print("\n✅ Entraînement terminé!")
else:
    print(f"\n📂 Chargement du test set: {test_file}")
    test = pd.read_csv(test_file)
    test['date'] = pd.to_datetime(test['date'])
    test = test.sort_values(['region_code', 'date']).reset_index(drop=True)

    # Extraire year/week
    test["year"] = test["date"].dt.year
    test["week_num"] = test["date"].dt.isocalendar().week

    print(f"   Test: {len(test)} obs ({test['date'].min()} → {test['date'].max()})")

    # Feature engineering pour test
    # NOTE: is_train=False car on ne peut pas calculer les vrais lags sur le test
    # Les lags seront imputés (NaN -> 0 dans create_features_improved)
    print("\n🔧 Feature engineering test (⚠️  sans vrais lags - à améliorer)...")
    test_feat = create_features_improved(test, train, is_train=False)

    X_test = test_feat[features].copy()

    print(f"   NaN in X_test: {X_test.isna().sum().sum()}")

    # Prédictions avec les modèles finaux
    print("\n🔮 Génération des prédictions...")
    pred_test_cat = np.clip(cat_final.predict(X_test), 0, None)
    pred_test_xgb = np.clip(xgb_final.predict(X_test), 0, None)
    pred_test_lgb = np.clip(lgbm_final.predict(X_test), 0, None)

    # Weighted ensemble
    pred_test_weighted = (
        best_w[0] * pred_test_cat +
        best_w[1] * pred_test_xgb +
        best_w[2] * pred_test_lgb
    )

    # Stacking
    meta_test = np.column_stack([pred_test_cat, pred_test_xgb, pred_test_lgb])
    pred_test_stacking = np.clip(meta_model.predict(meta_test), 0, None)

    # Stats
    print("\n📊 Statistiques des prédictions test:")
    print(f"   CatBoost:  mean={pred_test_cat.mean():.2f}, std={pred_test_cat.std():.2f}")
    print(f"   XGBoost:   mean={pred_test_xgb.mean():.2f}, std={pred_test_xgb.std():.2f}")
    print(f"   LightGBM:  mean={pred_test_lgb.mean():.2f}, std={pred_test_lgb.std():.2f}")
    print(f"   Weighted:  mean={pred_test_weighted.mean():.2f}, std={pred_test_weighted.std():.2f}")
    print(f"   Stacking:  mean={pred_test_stacking.mean():.2f}, std={pred_test_stacking.std():.2f}")

    # -----------------------------------------------------------------------------
    # Save submissions
    # -----------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("CRÉATION DES SUBMISSIONS")
    print("=" * 80)

    # Vérifier que test a un champ Id
    if "Id" not in test.columns:
        print("⚠️  Colonne 'Id' non trouvée dans le test set.")
        # Créer un Id si absent
        test["Id"] = range(len(test))
        print("   → Id créé automatiquement (0 à {})".format(len(test)-1))

    sub_cat = pd.DataFrame({
        "Id": test["Id"].astype(int),
        "TauxGrippe": pred_test_cat
    }).sort_values("Id").reset_index(drop=True)

    sub_xgb = pd.DataFrame({
        "Id": test["Id"].astype(int),
        "TauxGrippe": pred_test_xgb
    }).sort_values("Id").reset_index(drop=True)

    sub_lgb = pd.DataFrame({
        "Id": test["Id"].astype(int),
        "TauxGrippe": pred_test_lgb
    }).sort_values("Id").reset_index(drop=True)

    sub_weighted = pd.DataFrame({
        "Id": test["Id"].astype(int),
        "TauxGrippe": pred_test_weighted
    }).sort_values("Id").reset_index(drop=True)

    sub_stacking = pd.DataFrame({
        "Id": test["Id"].astype(int),
        "TauxGrippe": pred_test_stacking
    }).sort_values("Id").reset_index(drop=True)

    p_cat = OUT_DIR / "submission_cat_improved.csv"
    p_xgb = OUT_DIR / "submission_xgb_improved.csv"
    p_lgb = OUT_DIR / "submission_lgb_improved.csv"
    p_weighted = OUT_DIR / "submission_weighted_improved.csv"
    p_stacking = OUT_DIR / "submission_stacking_improved.csv"

    sub_cat.to_csv(p_cat, index=False)
    sub_xgb.to_csv(p_xgb, index=False)
    sub_lgb.to_csv(p_lgb, index=False)
    sub_weighted.to_csv(p_weighted, index=False)
    sub_stacking.to_csv(p_stacking, index=False)

    print("\n✅ Submissions sauvegardées:")
    print(f"   {p_cat.name}")
    print(f"   {p_xgb.name}")
    print(f"   {p_lgb.name}")
    print(f"   {p_weighted.name}")
    print(f"   {p_stacking.name}")

    print(f"\n💡 À soumettre en priorité: {best_model[0].lower()}")
    print(f"   (Meilleur score validation: RMSE={best_model[1]:.2f})")

    print("\n⚠️  NOTE IMPORTANTE:")
    print("   Les vrais lags temporels ne sont pas utilisés pour le test (is_train=False)")
    print("   Pour améliorer: implémenter prédiction auto-régressive (itérative)")
    print("   Cela pourrait encore réduire le RMSE de 5-15%")

    print("\n✅ Tout terminé! Prêt pour Kaggle 🚀")
