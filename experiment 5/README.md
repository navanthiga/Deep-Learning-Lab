# Experiment 5: CNN Training and Transfer Learning

This repository contains a Google Colab notebook for **Experiment 5: CNN
Training and Transfer Learning**. The experiment studies how different
training choices affect image-classification performance using
**MobileNetV2** with transfer learning.

The notebook focuses on initialization, regularization, optimizers,
hyperparameter tuning, feature extraction, fine-tuning, and a small
5-fold cross-validation study.

## Objectives

-   Study the effect of different weight initializers on classifier
    training.
-   Compare regularization techniques such as:
    -   L2 regularization
    -   Dropout
    -   Batch Normalization
-   Understand Batch Normalization using a small numerical example.
-   Compare different optimizers.
-   Study the effect of learning rate, batch size, and dropout rate.
-   Compare feature extraction with fine-tuning.
-   Perform a small 5-fold cross-validation study.
-   Evaluate the final selected model on an untouched test set.

## Dataset

The notebook uses the **Oxford-IIIT Pet Dataset** through TensorFlow
Datasets.

-   Number of classes: **37**
-   Images are resized to **224 × 224**
-   The official training split is divided into:
    -   Training set: 80%
    -   Validation set: 20%
-   The official test split is kept separate for final evaluation.
-   Batch size used by default: **32**

The dataset is loaded directly using:

``` python
tfds.load("oxford_iiit_pet")
```

## Model Architecture

The main model uses **MobileNetV2** as a pretrained feature extractor
with ImageNet weights.

### Architecture

``` text
Input Image (224 × 224 × 3)
        ↓
MobileNetV2 Backbone
        ↓
Global Average Pooling
        ↓
Optional Batch Normalization
        ↓
Optional Dropout
        ↓
Dense Layer
        ↓
Softmax Output (37 classes)
```

Initially, the MobileNetV2 base is frozen and only the newly added
classifier is trained.

For fine-tuning, the upper layers of MobileNetV2 are unfrozen and
trained using a smaller learning rate.

## Experiments Performed

### 1. Weight Initialization

The newly added classifier layers are tested with:

-   Random Normal
-   Glorot Uniform
-   He Normal

Training loss and validation accuracy are compared to study early
learning behaviour.

### 2. Regularization

Four configurations are compared:

  Configuration         Dropout     L2 Batch Normalization
  ------------------- --------- ------ ---------------------
  No regularization           0      0 No
  L2                          0   1e-4 No
  Dropout                   0.5      0 No
  BatchNorm                   0      0 Yes

Training and validation curves are plotted to examine generalization and
overfitting.

### 3. Batch Normalization Numerical Example

A small numerical example using:

``` text
[2, 4, 6, 8]
```

is used to calculate the mean, variance, and normalized values so that
the Batch Normalization formula can be connected to actual numerical
values.

### 4. Optimizer Comparison

The notebook compares different optimizers, including:

-   SGD
-   SGD with Momentum
-   RMSprop
-   Adam

For each optimizer, training loss, validation accuracy, training time,
and the best validation accuracy are recorded.

### 5. Hyperparameter Tuning

The notebook performs one-factor-at-a-time experiments for:

#### Learning Rate

``` text
1e-3
1e-4
```

#### Batch Size

``` text
16
32
64
```

#### Dropout Rate

``` text
0.0
0.25
0.5
```

The best validation accuracy is recorded for each setting.

### 6. Feature Extraction vs Fine-Tuning

Two stages are performed:

**Feature Extraction** - MobileNetV2 base is frozen. - Only the
classifier is trained. - Learning rate: `1e-3`

**Fine-Tuning** - Upper layers of MobileNetV2 are unfrozen. - Most
earlier layers remain frozen. - The last 30 layers are allowed to be
trainable. - Learning rate is reduced to `1e-5`.

The validation accuracy of both stages is compared.

### 7. 5-Fold Cross-Validation

A manageable subset of **500 training samples** is used to keep the
experiment practical in Colab.

Three configurations are compared:

``` text
C1: Dropout = 0.25, Adam, Learning Rate = 1e-3
C2: Dropout = 0.50, Adam, Learning Rate = 1e-3
C3: Dropout = 0.25, SGD,  Learning Rate = 1e-3
```

For each configuration:

-   5 folds are created using `StratifiedKFold`.
-   Validation accuracy is recorded for every fold.
-   Mean accuracy is calculated.
-   Standard deviation is calculated.

The configuration is selected based primarily on mean validation
accuracy while also considering variability.

## Final Evaluation

The selected configuration is trained again and evaluated on the
untouched test set.

The notebook reports:

-   Test Accuracy
-   Weighted Precision
-   Weighted Recall
-   Weighted F1-score
-   Training Time
-   Number of Model Parameters

A confusion matrix is also generated to visualize class-level prediction
performance.

## Installation and Requirements

The notebook is designed to run in **Google Colab**.

Main libraries used:

``` text
Python
TensorFlow
TensorFlow Datasets
NumPy
Pandas
Matplotlib
Scikit-learn
```

The notebook specifically installs:

``` bash
pip install "protobuf>=5.29.1,<6.0.0"
pip install tensorflow-datasets==4.9.9
```

## How to Run

### Option 1: Google Colab

1.  Open the `.ipynb` notebook in Google Colab.
2.  Run the cells from top to bottom.
3.  Allow TensorFlow Datasets to download the Oxford-IIIT Pet dataset.
4.  Observe the generated plots, metrics, tables, and confusion matrix.
5.  Use the values produced during your own execution for the final lab
    record.

### Option 2: Jupyter Notebook

Install the required dependencies first:

``` bash
pip install tensorflow tensorflow-datasets numpy pandas matplotlib scikit-learn
```

Then open:

``` text
Experiment_5_CNN_Study_Colab(1).ipynb
```

## Reproducibility

A random seed of:

``` python
SEED = 42
```

is used.

However, exact results can still vary depending on the runtime
environment, TensorFlow version, hardware, training duration, and other
execution details. Therefore, the metrics displayed after executing the
notebook should be treated as the final experimental results.

## Notebook Structure

``` text
Experiment 5
│
├── Setup
├── Load and Prepare Dataset
├── Data Visualization
├── Model and Training Helpers
├── Weight Initialization
├── Regularization and Overfitting
├── Batch Normalization Numerical Example
├── Optimizer Comparison
├── Hyperparameter Tuning
├── Feature Extraction and Fine-Tuning
├── 5-Fold Cross-Validation
├── Final Model Evaluation
├── Overall Results
└── Final Observation
```

## Key Takeaway

The experiment demonstrates that CNN performance depends on more than
the network architecture itself. Initialization affects early learning,
regularization affects generalization, optimizers influence training
behaviour, hyperparameters affect convergence, and careful fine-tuning
can improve a pretrained model.

The final model should therefore be assessed using **validation
performance, cross-validation variability, computational cost, and final
test performance**, rather than relying on a single accuracy value.

## Files

``` text
.
├── Experiment_5_CNN_Study_Colab(1).ipynb
└── README.md
```

## Author

**Navan**\
B.Tech Artificial Intelligence & Data Science
