# train.py
# Usage: python train.py
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

np.random.seed(42)
plt.rcParams.update({'figure.figsize': (8,5), 'font.size': 12})

DATA_CSV = os.path.join('summative', 'linear_regression', 'processed_bmi.csv')
OUTDIR = os.path.join('summative', 'linear_regression')
os.makedirs(OUTDIR, exist_ok=True)

if not os.path.exists(DATA_CSV):
    raise FileNotFoundError(f"Processed dataset not found at {DATA_CSV}. Run process_data.py first.")

df = pd.read_csv(DATA_CSV)
print("Loaded processed dataset:", DATA_CSV, " shape:", df.shape)

# FEATURES and TARGET
features = ['Age','Height','Weight','Bmi','daily_distance_km','terrain_factor']
target = 'battery_Wh'

for col in features + [target]:
    if col not in df.columns:
        raise KeyError(f"Missing required column: {col}")

X = df[features].copy()
y = df[target].copy()

# EDA: save correlation heatmap and histograms
sns.heatmap(X.join(y).corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation heatmap')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'correlation_heatmap.png'))
plt.clf()

X.hist(bins=30)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'feature_histograms.png'))
plt.clf()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# Standardize
scaler = StandardScaler().fit(X_train)
Xtr = scaler.transform(X_train)
Xte = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(OUTDIR, 'scaler.pkl'))
print("Saved scaler.pkl")

# Models
lr = LinearRegression().fit(Xtr, y_train)

# SGD (gradient descent) with history
sgd = SGDRegressor(max_iter=1, tol=None, learning_rate='invscaling', eta0=0.01, random_state=42, warm_start=True)
n_epochs = 200
train_losses = []
test_losses = []
for epoch in range(n_epochs):
    sgd.partial_fit(Xtr, y_train)
    train_losses.append(mean_squared_error(y_train, sgd.predict(Xtr)))
    test_losses.append(mean_squared_error(y_test, sgd.predict(Xte)))

# Save loss curves
plt.plot(train_losses, label='Train MSE')
plt.plot(test_losses, label='Test MSE')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.title('SGDRegressor Loss Curves')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'sgd_loss_curves.png'))
plt.clf()

dt = DecisionTreeRegressor(random_state=42).fit(Xtr, y_train)
rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1).fit(Xtr, y_train)

# Evaluate
models = {'LinearRegression': lr, 'SGDRegressor': sgd, 'DecisionTree': dt, 'RandomForest': rf}
metrics = {}
for name, model in models.items():
    preds = model.predict(Xte)
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    metrics[name] = {'mse': mse, 'mae': mae, 'r2': r2}

metrics_df = pd.DataFrame(metrics).T.sort_values('mse')
metrics_df.to_csv(os.path.join(OUTDIR, 'model_metrics.csv'))
print("Saved model_metrics.csv")
print(metrics_df.round(4))

# Choose best model (lowest test MSE)
best_name = metrics_df.index[0]
best_model = models[best_name]
joblib.dump(best_model, os.path.join(OUTDIR, 'best_model.pkl'))
print(f"Best model: {best_name} saved to {os.path.join(OUTDIR, 'best_model.pkl')}")

# Save test predictions
preds = best_model.predict(Xte)
pd.DataFrame({'actual': y_test.values, 'predicted': preds}).to_csv(os.path.join(OUTDIR, 'test_predictions.csv'), index=False)
print("Saved test_predictions.csv")

# Save scatter before/after (Weight vs battery_Wh and Predicted vs Actual)
plt.subplot(1,2,1)
plt.scatter(X['Weight'], y, alpha=0.6, s=20)
# linear fit overlay using lr: create line with median features
w_vals = np.linspace(X['Weight'].min(), X['Weight'].max(), 100)
med = X.median()
X_line = pd.DataFrame({c: np.repeat(med[c], 100) for c in X.columns})
X_line['Weight'] = w_vals
X_line_scaled = scaler.transform(X_line)
y_line = lr.predict(X_line_scaled)
plt.plot(w_vals, y_line, color='red', linewidth=2)
plt.xlabel('Weight (kg)'); plt.ylabel('battery_Wh'); plt.title('Weight vs battery_Wh + Linear fit')

plt.subplot(1,2,2)
plt.scatter(y_test, preds, alpha=0.6, s=20)
mn = min(y_test.min(), preds.min()); mx = max(y_test.max(), preds.max())
plt.plot([mn,mx],[mn,mx], color='red', linestyle='--')
plt.xlabel('Actual battery_Wh'); plt.ylabel('Predicted battery_Wh'); plt.title('Predicted vs Actual')

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'scatter_before_after.png'))
plt.clf()

print("Training pipeline complete. Artifacts saved in", OUTDIR)

