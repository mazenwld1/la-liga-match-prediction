# ⚽ La Liga Match Result Prediction

Predicting the outcome of La Liga matches (**Home win / Draw / Away win**) with **XGBoost**, built on 3,420 historical match records.

The notebook covers the full data science workflow: data understanding, cleaning, exploratory analysis, outlier and data-quality checks, feature importance, hyperparameter tuning, and model evaluation.

---

## 📊 Results

| Metric | Value |
|---|---|
| Test accuracy | **60.7%** (684 held-out matches, 80/20 stratified split) |
| Best hyperparameters | `n_estimators=200`, `max_depth=4`, `learning_rate=0.05`, `subsample=0.7`, `colsample_bytree=1.0` |
| Tuning | `GridSearchCV`, 162 combinations × 3-fold CV (486 fits) |

**Per-class performance**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Home win | 0.64 | 0.85 | 0.73 |
| Draw | 0.40 | 0.19 | 0.26 |
| Away win | 0.62 | 0.59 | 0.60 |

**Takeaways**
- The model is strong on home wins and reasonable on away wins.
- Draws are the hardest class to predict (recall 0.19). This is common in football, where draws have no clear statistical signature.
- The class distribution is imbalanced (1,568 home wins, 966 away wins, 886 draws), which favors the home class.

---

## 🗂️ Dataset

- **File:** `merged_laliga_all.csv` (not included, see the note below)
- **Size:** 3,420 matches, 22 raw columns
- **Quality:** no missing values (apart from an empty `Referee` column, which was dropped) and no duplicate rows. Logical checks passed: no negative goals, no shots-on-target above total shots, and no results inconsistent with the scores.

| Column | Meaning |
|---|---|
| `HomeTeam`, `AwayTeam` | Teams |
| `FTHG`, `FTAG`, `FTR` | Full-time home goals, away goals, result |
| `HTHG`, `HTAG`, `HTR` | Half-time home goals, away goals, result |
| `HS`, `AS` | Shots (home / away) |
| `HST`, `AST` | Shots on target |
| `HF`, `AF` | Fouls |
| `HC`, `AC` | Corners |
| `HY`, `AY` | Yellow cards |
| `HR`, `AR` | Red cards |

> 📝 **Note:** Add the dataset to a `data/` folder, or link to the original source here.

---

## 🔬 Methodology

1. **Data understanding:** loading, shape, types, summary statistics, and class balance.
2. **Preprocessing:** a single `LabelEncoder` fitted on both home and away teams, so a team keeps the same code in either role. `FTR` and `HTR` are encoded as `H=0`, `D=1`, `A=2`.
3. **Cleaning:** missing values, duplicates, and the empty `Referee` column.
4. **EDA:** box plots, histograms with KDE, skewness and kurtosis, count plots, a correlation heatmap, scatter plots, pair plots, and a 3D scatter plot.
5. **Unsupervised exploration:** K-Means (elbow method) and PCA.
6. **Outlier analysis:** Z-score and IQR methods with before/after comparison.
7. **Feature importance:** preliminary Random Forest ranking (shots on target ranked highest).
8. **Modeling:** `XGBClassifier` (multi-class softmax) tuned with `GridSearchCV`.
9. **Evaluation:** accuracy, classification report, one-vs-rest ROC curves, and per-match prediction confidence.

**Features used for modeling (14):** `HomeTeam`, `AwayTeam`, `HS`, `AS`, `HST`, `AST`, `HF`, `AF`, `HC`, `AC`, `HY`, `AY`, `HR`, `AR`.
Goals and half-time columns were removed to avoid leaking the result into the features.

---

## ⚠️ Limitations

- The features are **in-match statistics** (shots, corners, fouls, cards), so the model classifies a match's result from how the match played out. It is **not** a pre-match forecaster. A true pre-match model would use rolling form, head-to-head history, or league-table position instead.
- Draws are poorly predicted (see the table above).
- The split is random rather than chronological.

## 🚀 Future Improvements

- Add pre-match features (recent form, rolling averages, Elo ratings, head-to-head records).
- Use a time-based train/test split.
- Handle class imbalance (class weights, SMOTE) to improve draw detection.
- Compare against Random Forest, LightGBM, and Logistic Regression baselines.
- Add SHAP explanations.

---

## 🛠️ Tech Stack

Python · pandas · NumPy · Matplotlib · Seaborn · scikit-learn · XGBoost · SciPy · ydata-profiling · joblib

## ▶️ How to Run

```bash
git clone https://github.com/mazenwld1/la-liga-match-prediction.git
cd la-liga-match-prediction
pip install -r requirements.txt
jupyter notebook LA_LIGA_prediction_DM.ipynb
```

The notebook was written in Google Colab and reads the data from `/content/merged_laliga_all.csv`. Update that path to wherever your CSV lives (for example `data/merged_laliga_all.csv`) when running locally.

---

## 👤 Author

**Mazen Waleed**: Data Science student, Faculty of Computer Science, Egyptian Chinese University

- GitHub: [@mazenwld1](https://github.com/mazenwld1)
- LinkedIn: [Mazen Waleed](https://www.linkedin.com/in/mazen-waleed-881819443/)
