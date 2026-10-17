# Forest Cover Type Prediction: Gradient Boosting vs XGBoost vs LightGBM

## Project Overview

A forest resource management company needs to automatically predict **7 types of forest cover** from terrain data. The predictions support survey zoning, sustainable harvest planning, wildfire risk assessment, and prioritizing field inspections.

This project compares three gradient boosting libraries on the same task:

1. **Gradient Boosting** (`scikit-learn`)
2. **XGBoost**
3. **LightGBM**

They are compared on prediction quality, per-class performance, training time, and overfitting, not on accuracy alone.

## Dataset

The project uses the real **UCI Covertype** dataset, loaded with `sklearn.datasets.fetch_covtype`.

- **Source:** https://archive.ics.uci.edu/dataset/31/covertype
- **Size:** 581,012 observations, 54 features, 7 classes
- **Features:** terrain and environment measurements such as elevation, slope, distance to water, roads and fire points, hillshade, wilderness area, and soil type
- **Target:** forest cover type

| Label | Cover type |
|---|---|
| 1 | Spruce/Fir |
| 2 | Lodgepole Pine |
| 3 | Ponderosa Pine |
| 4 | Cottonwood/Willow |
| 5 | Aspen |
| 6 | Douglas-fir |
| 7 | Krummholz |

The classes are **heavily imbalanced**: Spruce/Fir and Lodgepole Pine make up most of the data, while Cottonwood/Willow is very rare.

By default, a stratified sample of **20,000 real rows** is used to keep training fast. This is only a compute limit; no synthetic data is used. Set `BENCHMARK_SAMPLE_SIZE = None` in the script to train on the full dataset.

## Methodology

1. **Load and verify** the dataset (the script asserts the expected shape of 581,012 x 54).
2. **Subsample** (optional, stratified) and encode labels as 0 to 6, because XGBoost requires labels starting at 0.
3. **Split** 75/25 into train and test sets, stratified by class, `random_state=42`.
4. **Handle imbalance** with class-balanced sample weights, so rare classes are not ignored during training.
5. **Train** the three models with comparable settings:

| Setting | Gradient Boosting | XGBoost | LightGBM |
|---|---|---|---|
| Trees (`n_estimators`) | 100 | 100 | 100 |
| Learning rate | 0.08 | 0.08 | 0.08 |
| Max depth | 10 | 7 | 7 |
| Row subsampling | 0.85 | 0.85 | 0.85 (`subsample_freq=1`) |
| Other | `min_samples_leaf=100` | `min_child_weight=3` | `min_child_weight=3` |

6. **Evaluate** each model and compare.

## Evaluation Metrics

| Metric | Why it matters |
|---|---|
| Test accuracy | Overall correctness, but misleading on imbalanced data |
| Balanced accuracy | Average recall across classes, so rare classes count equally |
| Macro F1 | Treats every class equally; the main ranking metric |
| Weighted F1 | F1 weighted by class size |
| Log loss | Quality of the predicted probabilities |
| Generalization gap | Train accuracy minus test accuracy; a measure of overfitting |
| Train time | Practical cost of each library |
| Per-class recall | Shows which cover types each model finds or misses |

## Project Structure

```
.
├── covertype_boosting_comparison.py   # full pipeline: data, training, evaluation, plots
└── README.md
```

The script is split into `# %%` cells, so it works as a normal Python file or can be pasted into Google Colab, Jupyter, or VS Code.

## Installation

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm ipython
```

In Google Colab, most libraries are preinstalled. You only need:

```python
!pip install xgboost lightgbm
```

## Usage

```bash
python covertype_boosting_comparison.py
```

The first run downloads the Covertype dataset (about 10 MB) and caches it.

The script produces:

- Class distribution table and chart
- Results table with all metrics for the three models
- Bar charts of test quality and training time
- Per-class recall heatmap
- Normalized confusion matrices for each model
- Top 12 feature importances, compared across models
- A short text summary: best Macro F1, fastest model, smallest generalization gap, and hardest class

## Results

Fill in after running the script:

| Model | Test accuracy | Balanced accuracy | Macro F1 | Gen. gap | Train time (s) |
|---|---|---|---|---|---|
| Gradient Boosting | | | | | |
| XGBoost | | | | | |
| LightGBM | | | | | |

Hardest class to predict: _fill in_

### Notes on interpreting the results

- **Do not rely on accuracy alone.** Because two classes dominate the data, a model can have high accuracy while missing rare cover types. Macro F1, balanced accuracy, and per-class recall show how the rare classes are handled.
- **Training time differs a lot.** scikit-learn's Gradient Boosting is much slower than XGBoost and LightGBM, especially with depth 10, 7 classes, and 100 trees. This is expected.
- **Feature importance is not directly comparable.** Each library measures it differently (impurity, gain, or split count). The values are normalized to sum to 1 and should be read as relative rankings only.
- **Results come from a 20,000-row sample by default.** Using the full dataset will change the numbers, usually improving all models.

## Possible Improvements

- Train on the full 581,012 rows with `BENCHMARK_SAMPLE_SIZE = None`.
- Tune hyperparameters (learning rate, depth, number of trees) with cross-validation or early stopping.
- Use stratified k-fold cross-validation instead of a single split for steadier estimates.
- Add SHAP values for a more consistent feature importance comparison.
- Prioritize recall on the classes most relevant to fire risk or harvest planning.

## Requirements

- Python 3.9+
- numpy, pandas, matplotlib, seaborn, scikit-learn, xgboost, lightgbm (ipython optional)

## Conclusion

This project compares three boosting libraries on a real, imbalanced, 7-class forest cover problem. The best choice depends on the trade-off between Macro F1, recall on the classes that matter most, training time, and overfitting, not just overall accuracy.