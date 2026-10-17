# Forest Cover Type Prediction: Gradient Boosting vs XGBoost vs LightGBM
# **Business context.** A forest resource management company needs to automatically
# predict 7 land cover types from terrain data. The results support survey zoning,
# sustainable harvest planning, wildfire risk assessment, and prioritizing field
# inspections.
#
# This notebook compares three boosting libraries on the real UCI Covertype dataset,
# using class-balanced sample weights because the classes are heavily imbalanced.

# Imports and settings
import time
import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb

from sklearn.datasets import fetch_covtype
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, log_loss, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

try:
    from IPython.display import display  # available in Colab / Jupyter
except ImportError:
    display = print

warnings.filterwarnings("ignore", category=UserWarning)
sns.set_theme(style="whitegrid")

RANDOM_STATE = 42

# This is only a COMPUTE limit, not synthetic data. Every row still comes from the
# real Covertype dataset (all 581,012 observations are downloaded first).
BENCHMARK_SAMPLE_SIZE = 20_000  # None = train on the full real dataset
N_ESTIMATORS = 100
LEARNING_RATE = 0.08

print("XGBoost:", xgb.__version__)
print("LightGBM:", lgb.__version__)

# Load data
TARGET_NAMES = np.array([
    "Spruce/Fir", "Lodgepole Pine", "Ponderosa Pine",
    "Cottonwood/Willow", "Aspen", "Douglas-fir", "Krummholz",
])
N_CLASSES = len(TARGET_NAMES)

DATA_SOURCE = "UCI Covertype - https://archive.ics.uci.edu/dataset/31/covertype"
EXPECTED_SHAPE = (581_012, 54)

dataset = fetch_covtype(as_frame=True)
X_full = dataset.data
y_full = dataset.target

assert X_full.shape == EXPECTED_SHAPE, (
    f"Unexpected dataset shape {X_full.shape}, expected {EXPECTED_SHAPE}"
)

# Optionally subsample (stratified) to keep training time reasonable
if BENCHMARK_SAMPLE_SIZE is not None and BENCHMARK_SAMPLE_SIZE < len(X_full):
    X, _, y_raw, _ = train_test_split(
        X_full, y_full, train_size=BENCHMARK_SAMPLE_SIZE,
        stratify=y_full, random_state=RANDOM_STATE,
    )
else:
    X, y_raw = X_full.copy(), y_full.copy()

# XGBoost requires class labels 0..K-1 (Covertype labels are 1..7)
encoder = LabelEncoder()
y = encoder.fit_transform(y_raw)

print("Source:", DATA_SOURCE)
print(f"Verified real dataset: {X_full.shape[0]:,} rows x {X_full.shape[1]} features "
      f"x {len(np.unique(y_full))} classes")
print(f"Benchmark data: {X.shape[0]:,} real rows x {X.shape[1]} features")
print(f"Missing values: {int(X.isna().sum().sum())}")
display(X.head())

# Class distribution
class_counts = pd.Series(y).value_counts().sort_index()
class_distribution = pd.DataFrame({
    "class": TARGET_NAMES,
    "count": class_counts.values,
    "percent": 100 * class_counts.values / len(y),
})
display(class_distribution.round(2))

plt.figure(figsize=(11, 4))
sns.barplot(data=class_distribution, x="class", y="percent", color="#3B82F6")
plt.xticks(rotation=25, ha="right")
plt.ylabel("Share of samples (%)")
plt.xlabel("")
plt.title("Class distribution - clearly imbalanced")
plt.tight_layout()
plt.show()

# Train/test split and class-balanced sample weights
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)
# Rare classes get larger weights so the models don't ignore them
sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

print("Train:", X_train.shape, "| Test:", X_test.shape)
print("Sample weight range:", sample_weight.min().round(3),
      "->", sample_weight.max().round(3))

# Define models
models = {
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE,
        max_depth=10, min_samples_leaf=100, subsample=0.85,
        random_state=RANDOM_STATE,
    ),
    "XGBoost": xgb.XGBClassifier(
        objective="multi:softprob", num_class=N_CLASSES,
        n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE,
        max_depth=7, min_child_weight=3, subsample=0.85,
        random_state=RANDOM_STATE,
    ),
    "LightGBM": lgb.LGBMClassifier(
        objective="multiclass", num_class=N_CLASSES,
        n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE,
        max_depth=7, min_child_weight=3,
        subsample=0.85, subsample_freq=1,  # LightGBM ignores subsample unless freq > 0
        random_state=RANDOM_STATE, verbose=-1,
    ),
}

# Train and evaluate
results = []
predictions = {}
probabilities = {}

for name, model in models.items():
    print(f"Training {name}...")
    start = time.perf_counter()
    model.fit(X_train, y_train, sample_weight=sample_weight)
    train_seconds = time.perf_counter() - start

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    y_train_pred = model.predict(X_train)  # used to measure the generalization gap

    predictions[name] = y_pred
    probabilities[name] = y_prob

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_pred)
    results.append({
        "Model": name,
        "Train accuracy": train_acc,
        "Test accuracy": test_acc,
        "Generalization gap": train_acc - test_acc,
        "Balanced accuracy": balanced_accuracy_score(y_test, y_pred),
        "Macro F1": f1_score(y_test, y_pred, average="macro"),
        "Weighted F1": f1_score(y_test, y_pred, average="weighted"),
        "Log loss": log_loss(y_test, y_prob, labels=np.arange(N_CLASSES)),
        "Train time (s)": train_seconds,
    })
    print(f"  done in {train_seconds:.1f}s | Macro-F1={results[-1]['Macro F1']:.4f}")

results_df = pd.DataFrame(results).set_index("Model")
display(results_df.round(4).sort_values("Macro F1", ascending=False))

# Compare quality and training time
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

quality_cols = ["Test accuracy", "Balanced accuracy", "Macro F1"]
results_df[quality_cols].plot(
    kind="bar", ax=axes[0], ylim=(0, 1), color=["#2563EB", "#F59E0B", "#10B981"]
)
axes[0].set_title("Quality on the test set")
axes[0].set_ylabel("Score")
axes[0].set_xlabel("")
axes[0].tick_params(axis="x", rotation=0)

results_df["Train time (s)"].sort_values().plot(kind="bar", ax=axes[1], color="#8B5CF6")
axes[1].set_title("Training time - lower is better")
axes[1].set_ylabel("Seconds")
axes[1].set_xlabel("")
axes[1].tick_params(axis="x", rotation=0)

plt.tight_layout()
plt.show()

# Per-class recall
recall_df = pd.DataFrame(
    {
        name: recall_score(
            y_test, y_pred, labels=np.arange(N_CLASSES), average=None, zero_division=0
        )
        for name, y_pred in predictions.items()
    },
    index=TARGET_NAMES,
).T  # rows = models, columns = cover types

plt.figure(figsize=(13, 3.8))
sns.heatmap(recall_df, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=1)
plt.title("Recall per cover type")
plt.xlabel("True class")
plt.ylabel("Model")
plt.tight_layout()
plt.show()
display(recall_df.round(3))

# Confusion matrices
fig, axes = plt.subplots(1, len(predictions), figsize=(18, 5))
for ax, (name, y_pred) in zip(axes, predictions.items()):
    cm = confusion_matrix(y_test, y_pred, normalize="true")
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
        xticklabels=np.arange(1, N_CLASSES + 1),
        yticklabels=np.arange(1, N_CLASSES + 1), ax=ax,
    )
    ax.set_title(name)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
plt.suptitle("Confusion matrices normalized by true class", y=1.03, fontsize=14)
plt.tight_layout()
plt.show()

# Feature importance
# Each library measures importance differently (impurity, gain, split count), so the
# values are normalized to sum to 1 and should only be compared in relative terms.
importance_frames = []
for name, model in models.items():
    importance = np.asarray(model.feature_importances_, dtype=float)
    importance = importance / importance.sum()
    importance_frames.append(pd.DataFrame({
        "Feature": X.columns, "Importance": importance, "Model": name,
    }))

importance_df = pd.concat(importance_frames, ignore_index=True)
top_features = (
    importance_df.groupby("Feature")["Importance"].mean().nlargest(12).index
)
plot_importance = importance_df[importance_df["Feature"].isin(top_features)]

plt.figure(figsize=(11, 7))
sns.barplot(data=plot_importance, y="Feature", x="Importance", hue="Model",
            order=top_features)
plt.title("Top 12 features - relative comparison across models")
plt.tight_layout()
plt.show()

# Summary
best_quality = results_df["Macro F1"].idxmax()
fastest = results_df["Train time (s)"].idxmin()
smallest_gap = results_df["Generalization gap"].idxmin()
hardest_class = recall_df.mean(axis=0).idxmin()

print("CONCLUSIONS FROM THIS RUN")
print("-" * 55)
print(f"- Highest Macro-F1 : {best_quality} "
      f"({results_df.loc[best_quality, 'Macro F1']:.4f})")
print(f"- Fastest training : {fastest} "
      f"({results_df.loc[fastest, 'Train time (s)']:.2f}s)")
print(f"- Smallest gap     : {smallest_gap} "
      f"({results_df.loc[smallest_gap, 'Generalization gap']:.4f})")
print(f"- Hardest class    : {hardest_class} "
      f"(mean recall={recall_df[hardest_class].mean():.4f})")
print("\nThe choice should weigh Macro-F1, recall on the classes that matter most, "
      "training time, and the generalization gap - not accuracy alone.")