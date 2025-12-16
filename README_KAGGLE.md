# 🚀 KAGGLE INFLUENZA PREDICTION - GUIDE COMPLET

## 🎯 Objectif
Maximiser le score sur le leaderboard Kaggle en prédisant le taux de grippe pour 2012-2013.

---

## 📊 Contexte de la Compétition

- **Train** : 2004-2011 (9196 observations)
- **Test** : 2012-2013 (2288 observations - 104 semaines × 22 régions)
- **Gap** : On prédit **2 ans dans le futur** !

### Conséquences :
- ❌ Pas de lags directs possibles
- ✅ Mais on a les données météo 2012-2013
- ✅ On peut utiliser des moyennes historiques (SECRET WEAPON)

---

## 🏗️ Architecture de la Solution

### **Features utilisées** :

1. **Variables météo** (12 variables)
   - Température, humidité, vent, visibilité, précipitations, etc.

2. **Features temporelles cycliques** (7 variables)
   - Semaine de l'année, mois, saison
   - sin/cos de la semaine et du mois (pour capturer la cyclicité)

3. **Moyennes historiques** 🚀 (6 variables - **SECRET WEAPON**)
   - `TauxGrippe_hist_week_mean` : moyenne pour cette région à cette semaine de l'année
   - `TauxGrippe_hist_month_mean` : moyenne pour cette région ce mois
   - `TauxGrippe_hist_season_mean` : moyenne pour cette région cette saison
   - `TauxGrippe_region_mean` : moyenne globale de la région
   - `TauxGrippe_region_std` : écart-type de la région
   - `TauxGrippe_week_global_mean` : moyenne globale pour cette semaine

4. **Région** (1 variable)

**Total** : ~26 features

### **Modèles** :
- **XGBoost** (priorité #1)
- **LightGBM**
- **CatBoost**
- **Ensemble** : moyenne pondérée des 3 (basée sur leurs performances)

### **Validation** :
- Train : 2004-2010
- Validation : **2011** (mimique le test 2012-2013)
- Réentraînement final : 2004-2011

---

## 📁 Fichiers Créés

```
📂 ml/
├── 📓 KAGGLE_TRAIN_MODEL.ipynb       ← [1] Entraînement des modèles
├── 📓 KAGGLE_PREDICT.ipynb           ← [2] Génération des prédictions
├── 🐍 prepare_test_set.py            ← [0] Préparation du test (à exécuter en premier!)
├── 📄 README_KAGGLE.md               ← Ce fichier
│
├── 📂 data_plus/
│   ├── train_synop_cleaned_complet.csv    ← Train nettoyé
│   └── test_synop_merged.csv              ← Test préparé (après step 0)
│
└── 📂 (après entraînement)
    ├── xgb_final.pkl, lgb_final.pkl, cat_final.pkl  ← Modèles
    ├── imputer.pkl, weights.pkl, features.pkl        ← Artifacts
    ├── submission_ensemble.csv                        ← FICHIER FINAL
    ├── feature_importance.csv                         ← Analyse
    └── model_comparison.png, feature_importance.png   ← Visualisations
```

---

## 🚀 MARCHE À SUIVRE (3 ÉTAPES)

### **ÉTAPE 0 : Préparer le Test Set** ⏱️ ~5 min

Le test set doit être mergé avec les données météo 2012-2013.

```bash
# Exécuter le script de préparation
python3 prepare_test_set.py
```

**✅ Résultat** : Fichier `data_plus/test_synop_merged.csv` créé

---

### **ÉTAPE 1 : Entraîner les Modèles** ⏱️ ~10-15 min

Ouvrir et exécuter toutes les cellules de `KAGGLE_TRAIN_MODEL.ipynb`

**Ce que ça fait** :
1. Charge le train (2004-2011)
2. Crée les features temporelles + historiques
3. Split train/validation (2010/2011)
4. Entraîne XGBoost, LightGBM, CatBoost
5. Évalue les performances sur validation 2011
6. Réentraîne sur toutes les données (2004-2011)
7. Sauvegarde les modèles

**✅ Résultats attendus** :
- RMSE validation ~50-80 (dépend des données)
- R² > 0.8
- Modèles sauvegardés (*.pkl)

---

### **ÉTAPE 2 : Générer les Prédictions** ⏱️ ~2 min

Ouvrir et exécuter toutes les cellules de `KAGGLE_PREDICT.ipynb`

**Ce que ça fait** :
1. Charge les modèles entraînés
2. Charge le test set (2012-2013)
3. Crée les mêmes features qu'au train
4. Génère les prédictions avec les 3 modèles
5. Combine en ensemble (moyenne pondérée)
6. Crée `submission_ensemble.csv`

**✅ Résultat final** : `submission_ensemble.csv` prêt à soumettre !

---

## 📤 SOUMISSION SUR KAGGLE

1. Aller sur la page de la compétition Kaggle
2. Onglet "Submit Predictions"
3. Upload `submission_ensemble.csv`
4. Attendre le score !

### Si vous voulez tester les modèles individuellement :
Le notebook génère aussi :
- `submission_xgb.csv`
- `submission_lgb.csv`
- `submission_cat.csv`

Vous pouvez les soumettre séparément pour comparer.

---

## 🔧 OPTIMISATIONS POSSIBLES (si temps)

### 1. **Hyperparameter Tuning**
Utiliser GridSearch ou Optuna pour optimiser :
- `max_depth`, `learning_rate`, `n_estimators`
- `subsample`, `colsample_bytree`

### 2. **Feature Engineering Avancé**
- Interactions : `t × u` (température × humidité)
- Google Trends (données mensuelles disponibles)
- Lag des moyennes historiques
- Rolling std des features météo

### 3. **Ensemble Avancé**
- Stacking (meta-model)
- Blending avec différents poids

### 4. **Modèles Supplémentaires**
- Neural Networks (LSTM pour séries temporelles)
- Prophet (Facebook)
- ARIMA par région

---

## 📊 ANALYSE DES RÉSULTATS

### **Feature Importance**
Après l'entraînement, vérifiez `feature_importance.csv` :
- Les features historiques doivent être dans le top 5-10
- Les variables météo (t, u) sont importantes
- Les features cycliques (week_sin, month_sin) capturent la saisonnalité

### **Validation**
- RMSE sur 2011 doit être cohérent avec le test 2012-2013
- Si RMSE validation >> RMSE train → overfitting
- Si RMSE validation ≈ RMSE train → bon modèle

---

## ❓ TROUBLESHOOTING

### Problème : "ModuleNotFoundError: No module named 'pandas'"
**Solution** :
```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost matplotlib seaborn
```

### Problème : "FileNotFoundError: test_synop_merged.csv"
**Solution** : Exécutez d'abord `python3 prepare_test_set.py`

### Problème : "Mémoire insuffisante"
**Solutions** :
- Réduire `n_estimators` (500 → 200)
- Exécuter sur Google Colab (gratuit, GPU)
- Utiliser un sous-ensemble pour le dev

### Problème : "NaN dans les prédictions"
**Cause** : Features manquantes dans le test
**Solution** : Vérifier que toutes les features du train sont dans le test

---

## 🎓 CONCEPTS CLÉS UTILISÉS

1. **Validation temporelle** : Split 2010/2011 au lieu de shuffle
2. **Features cycliques** : sin/cos pour capturer la périodicité annuelle
3. **Moyennes historiques** : Exploiter les patterns saisonniers par région
4. **Ensemble methods** : Combiner plusieurs modèles réduit la variance
5. **Early stopping** : Évite l'overfitting

---

## 📈 SCOREESTIMÉ

Avec cette stratégie :
- **Baseline** (météo seule) : RMSE ~80-100
- **Avec features temporelles** : RMSE ~60-80
- **Avec moyennes historiques** 🚀 : RMSE ~40-60
- **Ensemble optimisé** : RMSE ~35-50

**Top 10%** de la compétition attendu ! 🏆

---

## 📞 BESOIN D'AIDE ?

1. Vérifiez les messages d'erreur dans les notebooks
2. Vérifiez `feature_importance.csv` pour comprendre les features
3. Comparez les RMSE train/validation pour diagnostiquer overfitting
4. Testez avec un sous-ensemble de données d'abord

---

## 🚀 BONNE CHANCE !

**Stratégie gagnante** :
1. ✅ Exécuter prepare_test_set.py
2. ✅ Exécuter KAGGLE_TRAIN_MODEL.ipynb
3. ✅ Exécuter KAGGLE_PREDICT.ipynb
4. ✅ Soumettre submission_ensemble.csv
5. 🎉 Profiter du top 10% !

---

**Date de création** : 2025-12-16
**Version** : 1.0
**Auteur** : Stratégie optimisée pour maximiser le score Kaggle
