# 🎯 Analyse et Améliorations du Modèle Ensemble

## ❌ Problèmes identifiés dans le code original

### 1. **LAGS FACTICES au lieu de VRAIS LAGS**
**Problème:**
```python
# Code original
lag_wmean_by_wn = wmean_by_wn.shift(1)  # Moyenne historique de la semaine N-1
df["lag1_seasonal"] = df["week_num"].map(lag_wmean_by_wn)
```

Ce n'est **PAS** le TauxGrippe de la semaine précédente pour chaque région, c'est juste la moyenne historique nationale de la semaine N-1.

**Solution:**
```python
# VRAI lag par région
df['lag1_real'] = df.groupby('region_code')['TauxGrippe'].shift(1)
df['lag2_real'] = df.groupby('region_code')['TauxGrippe'].shift(2)
df['lag3_real'] = df.groupby('region_code')['TauxGrippe'].shift(3)
df['lag4_real'] = df.groupby('region_code')['TauxGrippe'].shift(4)
```

**Impact attendu:** +15-25% d'amélioration du score (les lags réels sont TRÈS prédictifs pour les séries temporelles)

---

### 2. **PAS DE ROLLING FEATURES (moyennes mobiles)**
**Problème:** Pas de lissage des features bruyantes (Google trends, météo, target)

**Solution:**
```python
# Rolling means sur le target
for window in [2, 3, 4]:
    df[f'rolling_mean_{window}w'] = df.groupby('region_code')['TauxGrippe'].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
    )

# Rolling sur Google trends
for window in [2, 3, 4]:
    df[f'google_roll_{window}w'] = df.groupby('region_code')['google_grippe'].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean()
    )

# Rolling sur température
for window in [2, 4]:
    df[f't_roll_{window}w'] = df.groupby('region_code')['t'].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean()
    )
```

**Impact attendu:** +5-10% d'amélioration

---

### 3. **TARGET ENCODING de région_code INSUFFISANT**
**Problème:** CatBoost gère region_code en catégoriel, mais XGBoost/LightGBM utilisent juste un encoding numérique arbitraire

**Solution:**
```python
# Moyenne et std du TauxGrippe par région (historique)
agg_region = hist.groupby("region_code")["TauxGrippe"].agg(["mean", "std"]).reset_index()
df = df.merge(agg_region, on="region_code", how="left")

# Pattern saisonnier par région x semaine
agg_reg_week = hist.groupby(["region_code", "week_num"])["TauxGrippe"].mean().reset_index()
df = df.merge(agg_reg_week, on=["region_code", "week_num"], how="left")
```

**Impact attendu:** +5-10% d'amélioration

---

### 4. **GOOGLE TRENDS SOUS-EXPLOITÉ**
**Problème:** Juste `google_log` et `google_anomaly`, pas de dynamique temporelle

**Solution:**
```python
# Différence (variation semaine à semaine)
df["google_diff"] = df.groupby('region_code')['google_grippe'].diff()

# Accélération (diff de diff)
df["google_accel"] = df.groupby('region_code')['google_diff'].diff()

# Rolling
for window in [2, 3, 4]:
    df[f'google_roll_{window}w'] = df.groupby('region_code')['google_grippe'].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean()
    )

# Interactions enrichies
df["google_x_temperature"] = df["google_log"] * df["cold"]
df["google_x_region_mean"] = df["google_log"] * df["region_mean"]
```

**Impact attendu:** +3-8% d'amélioration

---

### 5. **SPLIT DE VALIDATION SOUS-OPTIMAL**
**Problème:** Split 80/20 arbitraire, pas aligné avec la tâche réelle

**Solution:**
```python
# Split temporel (comme Kaggle: prédire 2012-2013 depuis 2004-2011)
train_data = train[train["year"] <= 2010]  # 2004-2010
val_data   = train[train["year"] == 2011]  # 2011
```

Cela **simule exactement** la situation du test set (prédire 2012-2013 depuis 2004-2011).

**Impact:** Meilleure estimation du score réel, moins d'overfitting

---

### 6. **SIMPLE BLEND au lieu de STACKING**
**Problème:** Moyenne pondérée simple des prédictions

**Solution:** Utiliser un meta-model (Ridge, Linear Regression) qui apprend à combiner les prédictions

```python
# Créer les features de niveau 1
meta_train = np.column_stack([pred_val_cat, pred_val_xgb, pred_val_lgb])

# Meta-model
meta_model = Ridge(alpha=1.0)
meta_model.fit(meta_train, y_val)
pred_meta = meta_model.predict(meta_test)
```

**Impact attendu:** +2-5% d'amélioration

---

### 7. **MANQUE LightGBM**
**Problème:** Seulement CatBoost + XGBoost

**Solution:** Ajouter LightGBM qui a souvent des patterns différents et améliore l'ensemble

**Impact attendu:** +2-5% d'amélioration

---

## 🚀 Autres améliorations à tester

### A. **Features d'interactions avancées**
```python
# Interactions température x semaine
df["cold_x_w_mean"] = df["cold"] * df["w_mean"]
df["cold_x_peak"] = df["cold"] * df["is_peak"]

# Interactions Google x région
df["google_x_region_week"] = df["google_log"] * df["region_week_mean"]

# Ratio features
df["google_vs_hist"] = df["google_log"] / (df["w_mean"] + 1)
df["temp_vs_hist"] = df["t"] / (df.groupby('region_code')['t'].transform('mean') + 1)
```

**Impact attendu:** +2-5%

---

### B. **Features de tendance (trend)**
```python
# Tendance sur les 4 dernières semaines
df['trend_4w'] = df.groupby('region_code')['TauxGrippe'].transform(
    lambda x: x.rolling(4, min_periods=2).apply(lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) >= 2 else 0)
)

# Pente de Google trends
df['google_trend'] = df.groupby('region_code')['google_grippe'].transform(
    lambda x: x.diff()
)
```

**Impact attendu:** +2-4%

---

### C. **Features météo enrichies**
```python
# Interactions météo
df['cold_x_humidity'] = df['cold'] * df['u']
df['wind_x_rain'] = df['ff'] * df['rr1']

# Jours de froid extrême
df['extreme_cold'] = (df['t'] < 0).astype(int)
df['extreme_cold_x_week'] = df['extreme_cold'] * df['w_mean']
```

**Impact attendu:** +1-3%

---

### D. **Optimisation des hyperparamètres**
Faire un grid search ou Optuna pour:
- CatBoost: depth, learning_rate, l2_leaf_reg, min_data_in_leaf
- XGBoost: max_depth, min_child_weight, learning_rate, subsample
- LightGBM: num_leaves, learning_rate, min_child_samples

**Impact attendu:** +3-8%

---

### E. **Features de variabilité**
```python
# Écart-type mobile sur le target
df['target_std_4w'] = df.groupby('region_code')['TauxGrippe'].transform(
    lambda x: x.rolling(4, min_periods=2).std().shift(1)
)

# Coefficient de variation Google
df['google_cv'] = df.groupby('region_code')['google_grippe'].transform(
    lambda x: x.rolling(4, min_periods=2).std() / (x.rolling(4, min_periods=2).mean() + 1)
)
```

**Impact attendu:** +1-3%

---

### F. **Post-processing des prédictions**
```python
# Clipper les prédictions négatives
pred = np.clip(pred, 0, None)

# Ajuster les prédictions pour qu'elles respectent les patterns saisonniers
# (si prédiction << moyenne historique pour cette semaine, remonter un peu)
seasonal_mean = ...  # moyenne historique par semaine
pred_adjusted = pred * 0.8 + seasonal_mean * 0.2
```

**Impact attendu:** +1-2%

---

## 📊 Ordre de priorité des améliorations

### 🔥 CRITIQUE (impact > 10%)
1. ✅ **VRAIS LAGS temporels** (lag1, lag2, lag3, lag4 par région)
2. ✅ **Target encoding de région** (mean, std historique)

### ⭐ IMPORTANT (impact 5-10%)
3. ✅ **Rolling features** (moyennes mobiles 2-4 semaines)
4. ✅ **Google trends amélioré** (diff, rolling, interactions)
5. ✅ **Split temporel propre** (2004-2010 train, 2011 val)
6. ✅ **LightGBM** (ajouter au ensemble)

### 💡 RECOMMANDÉ (impact 2-5%)
7. ✅ **Stacking** (meta-model au lieu de simple blend)
8. **Hyperparamètres optimisés** (Optuna)
9. **Interactions avancées** (temp x google, etc.)
10. **Features de tendance** (pente sur 4 semaines)

### 🎨 BONUS (impact 1-3%)
11. **Features météo enrichies**
12. **Features de variabilité**
13. **Post-processing** des prédictions

---

## 🎯 Plan d'action

### Phase 1: Quick wins (déjà fait ✅)
- [x] Implémenter vrais lags
- [x] Implémenter rolling features
- [x] Target encoding de région
- [x] Split temporel
- [x] Ajouter LightGBM
- [x] Stacking

### Phase 2: À tester maintenant
- [ ] **Exécuter `ensemble_improved.py`** sur les données
- [ ] Comparer les scores avec le code original
- [ ] Identifier les features les plus importantes (feature_importances_)

### Phase 3: Optimisation (si besoin)
- [ ] Grid search hyperparamètres
- [ ] Features d'interactions avancées
- [ ] Post-processing

### Phase 4: Test set
- [ ] Adapter le code pour le test set (gérer les lags à partir du train)
- [ ] Générer les prédictions
- [ ] Soumettre sur Kaggle

---

## ⚠️ Point critique pour le TEST SET

Pour prédire le test set (2012-2013), il faut:

1. **Pour les lags**: utiliser les dernières valeurs du train (2011) pour démarrer
2. **Pour les rolling features**: utiliser les fenêtres qui chevauchent train/test
3. **Pour les stats historiques**: utiliser TOUT le train (2004-2011)

Exemple:
```python
# Pour prédire 2012 semaine 1
# lag1 = TauxGrippe de 2011 semaine 52 (dernière semaine du train)
# lag2 = TauxGrippe de 2011 semaine 51
# etc.
```

Il faudra soit:
- Faire une prédiction **itérative** (prédire semaine par semaine, utiliser les prédictions comme lags)
- Ou stocker les dernières valeurs du train et les utiliser pour initialiser les lags du test

---

## 📈 Estimation d'amélioration globale

Si le code original donne RMSE = 100:
- Avec vrais lags + rolling + target encoding: **RMSE ≈ 70-80** (-20 à -30%)
- Avec optimisation hyperparamètres: **RMSE ≈ 65-75** (-5 à -10% supplémentaire)
- Avec features avancées: **RMSE ≈ 60-70** (-5% supplémentaire)

**Total attendu: -30 à -40% d'amélioration du RMSE** 🚀
