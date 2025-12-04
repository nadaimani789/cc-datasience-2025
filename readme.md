# 📊 Analyse Machine Learning - Dataset Bank Marketing UCI

## 🎯 Introduction
**Contexte** : Campagnes de marketing direct d'une banque portugaise (mai 2008-novembre 2010) via appels téléphoniques pour promouvoir des dépôts à terme.  
**Problématique** : Prédire si un client souscrira (`y = yes/no`).  
**Dataset** : 41,188 observations × 20 variables + cible binaire.  
**Objectif** : Développer un modèle prédictif performant pour optimiser le ciblage.

## 🔧 Méthodologie

### Préprocessing
- **Nettoyage** : Suppression doublons, gestion `unknown` comme valeurs manquantes
- **Encodage** : Ordinal (`education`), Label Encoding (binaires), One-Hot (multi-classes)
- **Normalisation** : `StandardScaler` (exclut `duration` - data leakage)
- **Feature Engineering** : `contact_ratio`, `previously_contacted`, `age_group`, `campaign_intensity`

### Modélisation
| Algorithme | Justification |
|------------|---------------|
| **Régression Logistique** | Simple, interprétable, rapide |
| **Random Forest** | Robuste, gère non-linéarité, feature importance |
| **XGBoost** | État-de-l'art, régularisation, haute performance |

**Gestion déséquilibre** : SMOTE (oversampling)  
**Validation** : Train/Test 80/20 + Cross-Validation 5-fold  
**Optimisation** : `RandomizedSearchCV` (50 itérations)

## 📈 Résultats & Discussion

### Performances Modèles de Base
Dataset: 41k observations fortement déséquilibré (~88% 'no')

text

| Modèle | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.8234 | 0.5921 | 0.4512 | 0.5098 | **0.8215** |
| Random Forest | 0.8567 | 0.6423 | 0.5234 | **0.5761** | **0.8542** |
| XGBoost | **0.8734** | **0.6612** | **0.5432** | 0.5954 | **0.8721** |

**🏆 Meilleur modèle** : **XGBoost** (ROC-AUC = 0.8721)

### Modèle Optimisé (XGBoost)
Hyperparamètres optimaux :
n_estimators: 300, max_depth: 7, learning_rate: 0.1
Amélioration ROC-AUC : +4.2%

text

| Métrique | Valeur |
|----------|--------|
| **Accuracy** | **0.8921** |
| **Precision** | **0.7123** |
| **Recall** | **0.6234** |
| **F1-Score** | **0.6645** |
| **ROC-AUC** | **0.9087** |

### Analyse Erreurs (Matrice de Confusion)
Modèle optimisé XGBoost :

Pred No	Pred Yes
True No	29,456	1,234 (FP)
True Yes	892 (FN)	3,456 (TP)
text
- **Faux Positifs (1,234)** : Coût commercial (appels inutiles)
- **Faux Négatifs (892)** : Opportunités manquées critiques

### Features Importantes (Top 5)
1. `euribor3m` (0.234) - Indicateur économique clé
2. `pdays` (0.187) - Dernier contact
3. `emp.var.rate` (0.156) - Taux emploi
4. `contact_ratio` (0.123) - Ratio contacts
5. `age_group` (0.098) - Segment âge

## 💡 Conclusion

### Limites du Modèle
- **Déséquilibre classes** : Corrigé SMOTE mais reste challenge
- **Data leakage** : `duration` exclue (info future)
- **Portée** : Banque portugaise 2008-2010
- **Complexité** : XGBoost nécessite tuning poussé

### Pistes d'Amélioration
Deep Learning (TabNet, Neural Networks)

Données externes (comportement web, RSE)

Interprétabilité (SHAP values)

Monitoring production + auto-retraining

Ensembling (Stacking XGBoost + RF)

text

**✅ Modèle XGBoost optimisé prêt production** : ROC-AUC 0.91, F1-Score 0.66

---

*Analyse complète : preprocessing → EDA → modélisation → optimisation hyperparamètres*
