# DL Lab 1 — Single Layer Perceptron for Binary Classification

**Course:** CS3807 – Deep Learning Laboratory
**Institution:** Shiv Nadar University Chennai
**Program:** B.Tech Artificial Intelligence & Data Science

## Objective
Implement a Single Layer Perceptron from scratch to understand the perceptron learning algorithm, activation functions, and binary classification on a real-world dataset.

## Dataset
**Banknote Authentication Dataset** (UCI Machine Learning Repository)
- 1372 instances, 4 numerical features (Variance, Skewness, Curtosis, Entropy)
- Binary target: 0 = Authentic, 1 = Forged
- No missing values; classes are roughly balanced (~55.5% / 44.5%)

## What's in this notebook
1. **Dataset exploration** — shape, missing values, descriptive statistics
2. **EDA** — histograms, correlation heatmap, scatter plot, boxplots by class
3. **Preprocessing** — stratified 80/20 train-test split (before scaling, to avoid data leakage), `StandardScaler`
4. **Perceptron implementation** — built from scratch in NumPy (step activation, weighted sum, perceptron learning rule)
5. **Training** — 50 epochs, learning rate 0.01, with per-epoch misclassification/weight/bias logging
6. **Evaluation** — Accuracy, Precision, Recall, F1-score, Confusion Matrix
7. **Learning rate comparison** — η = 0.001, 0.01, 0.1, under both zero and random weight initialization
8. **Comparison with scikit-learn's `Perceptron`**
9. **Additional tasks:**
   - Perceptron learning for AND, OR, NOT gates, with decision boundary plotted after every weight update
   - Perceptron learning for XOR — demonstrates and analyzes why a single-layer perceptron cannot converge on a non-linearly-separable problem

## Key Results

| Metric | Value |
|---|---|
| Accuracy | 98.5% |
| Precision | 96.8% |
| Recall | 100% |
| F1-score | 98.4% |

- Training never converged to zero error (oscillated between 12–20 misclassified samples across 50 epochs) because the dataset is only approximately, not perfectly, linearly separable.
- All test-set errors were False Positives; zero False Negatives — a favorable error profile for a currency-authentication use case.
- Skewness and curtosis showed multicollinearity (correlation ≈ 0.8), which affected how weight magnitude was distributed relative to each feature's individual correlation with the class label.
- With zero-initialized weights, learning rate had no effect on the classification trajectory (a consequence of the step function's scale-invariance) — this stopped being true once random initialization was used instead.
- AND, OR, and NOT gates converged normally (linearly separable); XOR never converged within 20 epochs, cycling through a repeating set of decision boundaries instead of settling — a direct, visual consequence of XOR not being linearly separable.

## How to Run
1. Open `Experiment_1_Perceptron.ipynb` in Google Colab or Jupyter.
2. Run all cells top to bottom — the dataset is loaded directly from the UCI repository URL, no manual download needed.
3. Requires: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`.

## References
- F. Rosenblatt, "The Perceptron," *Psychological Review*, 1958.
- UCI Machine Learning Repository – Banknote Authentication Dataset
- M. Minsky and S. Papert, *Perceptrons: An Introduction to Computational Geometry*, MIT Press, 1969.
