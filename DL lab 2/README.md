# DL Lab 2 — MLP for Multi-Class Image Classification

Implementation of a Multi-Layer Perceptron (MLP) using TensorFlow/Keras on the **Fashion-MNIST** dataset, including preprocessing, training, evaluation, and automated hyperparameter optimization.

**Course:** CS3807 — Deep Learning Laboratory
**Environment:** Google Colab (GPU recommended)

---

## Objective

Build an MLP that classifies Fashion-MNIST images into 10 clothing categories, then use automated hyperparameter search to try to improve on a baseline model — and analyse the result honestly.

## Dataset

| Property | Value |
|---|---|
| Dataset | Fashion-MNIST |
| Train / Test | 60,000 / 10,000 (predefined split) |
| Classes | 10 (balanced, 6,000 per class) |
| Image size | 28 × 28 grayscale |
| Flattened dimension | 784 |

## Model Architecture

```
784 → Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, Softmax)
```

Total parameters: **109,386** (≈92% in the first layer, due to full connectivity).

---

## Pipeline

1. **Load & explore** — inspect shapes, view samples, plot class distribution
2. **Preprocess** — flatten to 784, normalize to [0,1], one-hot encode labels
3. **Build** — baseline MLP (Adam, categorical cross-entropy)
4. **Train** — 20 epochs, batch size 32, 20% validation split
5. **Evaluate** — accuracy, macro precision/recall/F1, confusion matrix
6. **Optimize** — RandomizedSearchCV (5-fold CV) over an 8-hyperparameter space
7. **Retrain & compare** — best config on full data vs baseline

---

## Results

| Metric | Baseline | Optimized |
|---|---|---|
| Accuracy | **0.8784** | 0.8615 |
| Precision (macro) | 0.8817 | 0.8640 |
| Recall (macro) | 0.8784 | 0.8615 |
| F1-score (macro) | 0.8788 | 0.8623 |
| Training time (s) | 132.3 | 185.2 |

**Best hyperparameters found:** 3 layers, 32 neurons, Tanh, dropout 0.2, RMSProp, lr 0.001, batch 32, 30 epochs.

### Key finding

The "optimized" model performed **worse** than the baseline. This is a genuine result, not a bug:

- The search ran on an **8,000-image subsample** (6,400 per CV fold). On that little data, a small, heavily-regularized model is optimal because a larger one would overfit.
- Retrained on the full 48,000 images, that same small model **underfits** — its train/validation accuracy gap was only ~0.001, meaning it lacked the capacity to fit the problem.
- **Lesson:** hyperparameter search on a subsample does not guarantee improvement at full scale.

### Per-class insight

Overall accuracy hides an uneven error distribution. **Shirt** was the weakest class (F1 0.69) with high recall but low precision — its decision region absorbs ambiguous examples from T-shirt/top, Pullover, and Coat. **Trouser** was the strongest (F1 0.98), thanks to its distinctive silhouette. Even with perfectly balanced classes, accuracy alone is insufficient for evaluation.

---

## How to Run

1. Open `MLP.ipynb` in Google Colab.
2. Set the runtime to **GPU** (`Runtime → Change runtime type → T4 GPU`) — the hyperparameter search is much faster on GPU.
3. `Runtime → Run all`.

The first cell pins compatible dependency versions:

```python
!pip install -q "scikit-learn==1.5.2" "scikeras==0.13.0"
```

> **Note on dependencies:** SciKeras is incompatible with scikit-learn ≥ 1.6 (throws an `AttributeError: '__sklearn_tags__'`). The version pin above resolves it. Restart the runtime after the install if prompted.

## Requirements

TensorFlow / Keras · scikit-learn 1.5.2 · SciKeras 0.13.0 · NumPy · Pandas · Matplotlib · Seaborn

---

## Plots Generated

Sample images · class distribution · training/validation accuracy · training/validation loss · confusion matrix (baseline & optimized) · hyperparameter search results · baseline vs optimized comparison.

## References

1. Goodfellow, Bengio, Courville — *Deep Learning*, MIT Press, 2016.
2. Xiao, Rasul, Vollgraf — *Fashion-MNIST*, arXiv:1708.07747, 2017.
3. Bergstra & Bengio — *Random Search for Hyper-Parameter Optimization*, JMLR 13, 2012.
4. TensorFlow/Keras & SciKeras documentation.
