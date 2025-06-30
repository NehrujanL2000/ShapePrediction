import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import VotingRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

# === Create output folder for saving models ===
output_folder = "/content/output"
os.makedirs(output_folder, exist_ok=True)

# === Load Dataset ===
file_path = "/content/drive/MyDrive/WithShapeLabels.xlsx"
df = pd.read_excel(file_path)

# === Split the data ===
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")

# ========================== MODEL 1: Linear Regression for Volumes ==========================
X1_train = train_df[['Max', 'Mid', 'Min']]
y1_train = train_df[['Actual volume', 'Covex hull volume', 'surface area']]

X1_val = val_df[['Max', 'Mid', 'Min']]
y1_val = val_df[['Actual volume', 'Covex hull volume', 'surface area']]

X1_test = test_df[['Max', 'Mid', 'Min']]
y1_test = test_df[['Actual volume', 'Covex hull volume', 'surface area']]

model1 = LinearRegression()
model1.fit(X1_train, y1_train)

val_preds1 = model1.predict(X1_val)
test_preds1 = model1.predict(X1_test)

val_r2 = r2_score(y1_val, val_preds1, multioutput='uniform_average')
test_r2 = r2_score(y1_test, test_preds1, multioutput='uniform_average')

print(f"\n✅ Model 1 (Linear Regression - Volumes): Val R2: {val_r2:.3f}, Test R2: {test_r2:.3f}")

joblib.dump(model1, os.path.join(output_folder, 'model1_linear_reg.pkl'))

# ========================== MODEL 2: Feature Engineering ==========================
def calculate_features(row):
    min_, mid, max_ = row['Min'], row['Mid'], row['Max']
    av, chv, sa = row['Actual volume'], row['Covex hull volume'], row['surface area']
    EI = mid / max_ if max_ != 0 else 0
    FI = min_ / mid if mid != 0 else 0
    AR = (EI + FI) / 2
    CI = av / chv if chv != 0 else 0
    S = ((36 * np.pi * (av ** 2)) ** (1/3)) / sa if sa != 0 else 0
    return pd.Series([EI, FI, AR, CI, S])

for subset in [train_df, val_df, test_df, df]:
    subset[['EI', 'FI', 'AR', 'CI', 'S']] = subset.apply(calculate_features, axis=1)

feature_cols = ['EI', 'FI', 'AR', 'CI', 'S']

scaler = StandardScaler()
X2_train = scaler.fit_transform(train_df[feature_cols])
X2_val = scaler.transform(val_df[feature_cols])
X2_test = scaler.transform(test_df[feature_cols])

joblib.dump(scaler, os.path.join(output_folder, 'scaler_model2.pkl'))

y2_train_reg = train_df[['Friction angle', 'Void Ratio']].values
y2_val_reg = val_df[['Friction angle', 'Void Ratio']].values
y2_test_reg = test_df[['Friction angle', 'Void Ratio']].values

# ========================== 10-FOLD CV: Model 2 Regression (Voting Regressor MultiOutput) ==========================
X_all = scaler.transform(df[feature_cols])
y_all_reg = df[['Friction angle', 'Void Ratio']].values

kf = KFold(n_splits=10, shuffle=True, random_state=42)
r2_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_all)):
    X_train_cv, X_val_cv = X_all[train_idx], X_all[val_idx]
    y_train_cv, y_val_cv = y_all_reg[train_idx], y_all_reg[val_idx]

    estimators = [
        ('hgb', HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=42)),
        ('knn', KNeighborsRegressor(n_neighbors=5)),
        ('bayes', BayesianRidge()),
        ('svr', SVR())
    ]
    voting_reg = VotingRegressor(estimators)
    model_cv = MultiOutputRegressor(voting_reg)
    model_cv.fit(X_train_cv, y_train_cv)

    val_preds_cv = model_cv.predict(X_val_cv)
    r2_cv = r2_score(y_val_cv, val_preds_cv, multioutput='uniform_average')
    r2_scores.append(r2_cv)
    print(f"Fold {fold + 1} Regression R2: {r2_cv:.3f}")

print(f"\nAverage 10-fold CV Regression R2: {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")

# ========================== 10-FOLD CV: Model 2 Classification (Naive Bayes) ==========================
X_cls_all = scaler.transform(df[feature_cols])
y_cls_all = LabelEncoder().fit_transform(df['Shape'].values)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
acc_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_cls_all, y_cls_all)):
    X_train_cv, X_val_cv = X_cls_all[train_idx], X_cls_all[val_idx]
    y_train_cv, y_val_cv = y_cls_all[train_idx], y_cls_all[val_idx]

    gnb = GaussianNB()
    gnb.fit(X_train_cv, y_train_cv)
    val_preds_cv = gnb.predict(X_val_cv)
    acc_cv = accuracy_score(y_val_cv, val_preds_cv)
    acc_scores.append(acc_cv)
    print(f"Fold {fold + 1} Classification Accuracy: {acc_cv:.3f}")

print(f"\nAverage 10-fold CV Classification Accuracy: {np.mean(acc_scores):.3f} ± {np.std(acc_scores):.3f}")

# ========================== Final Model Training and Saving ==========================
# Regression Final Model
voting_reg_final = VotingRegressor(estimators)
model2_reg_final = MultiOutputRegressor(voting_reg_final)
model2_reg_final.fit(X2_train, y2_train_reg)

val_preds2_reg_final = model2_reg_final.predict(X2_val)
test_preds2_reg_final = model2_reg_final.predict(X2_test)

val_r2_reg_final = r2_score(y2_val_reg, val_preds2_reg_final, multioutput='uniform_average')
test_r2_reg_final = r2_score(y2_test_reg, test_preds2_reg_final, multioutput='uniform_average')

print(f"\n✅ Model 2 Regression Final: Val R2: {val_r2_reg_final:.3f}, Test R2: {test_r2_reg_final:.3f}")

joblib.dump(model2_reg_final, os.path.join(output_folder, 'model2_voting_multioutput.pkl'))

# Classification Final Model
y2_train_cls = train_df['Shape'].values
y2_val_cls = val_df['Shape'].values
y2_test_cls = test_df['Shape'].values

label_encoder = LabelEncoder()
y2_train_cls_enc = label_encoder.fit_transform(y2_train_cls)
y2_val_cls_enc = label_encoder.transform(y2_val_cls)
y2_test_cls_enc = label_encoder.transform(y2_test_cls)

gnb_final = GaussianNB()
gnb_final.fit(X2_train, y2_train_cls_enc)

val_preds2_cls_final = gnb_final.predict(X2_val)
test_preds2_cls_final = gnb_final.predict(X2_test)

val_acc_cls_final = accuracy_score(y2_val_cls_enc, val_preds2_cls_final)
test_acc_cls_final = accuracy_score(y2_test_cls_enc, test_preds2_cls_final)

print(f"\n✅ Model 2 Classification Final: Val Acc: {val_acc_cls_final:.3f}, Test Acc: {test_acc_cls_final:.3f}")

joblib.dump(gnb_final, os.path.join(output_folder, 'model2_gnb_cls.pkl'))
joblib.dump(label_encoder, os.path.join(output_folder, 'shape_label_encoder.pkl'))

print("\n✅✅ All models trained, validated, 10-fold cross-validated, tested, and saved ✅✅")
print(f"\n✅ Saved models and scaler to: {output_folder}")