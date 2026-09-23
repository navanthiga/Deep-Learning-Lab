# CS3807 – Deep Learning Laboratory
## Experiment 6: End-to-End Study of RNN, LSTM and GRU for Sequence Learning and Video Understanding

**Shiv Nadar University Chennai**  
**B.Tech Artificial Intelligence & Data Science**  
**Academic Year: 2026–27**

This repository contains the Jupyter/Google Colab notebook for **Experiment 6** of the Deep Learning Laboratory.

## Objective

The experiment provides an end-to-end study of recurrent neural networks for sequential data. It covers:

1. Human Activity Recognition (HAR) using **SimpleRNN, LSTM and GRU**
2. A numerical **Backpropagation Through Time (BPTT)** exercise
3. The effect of different sequence lengths on recurrent-model performance
4. Video understanding using **CNN feature extraction + LSTM/GRU**
5. Sequence-to-sequence learning using an **encoder–decoder LSTM**

The notebook follows the experimental workflow used in the laboratory manual and includes preprocessing, model training, evaluation, visualizations and consolidated results.

## Notebook Contents

### 1. Setup
The notebook imports the required Python libraries, fixes random seeds for reproducibility and prepares the TensorFlow/Keras environment.

### 2. UCI HAR Dataset

The **UCI Human Activity Recognition dataset** is used for sensor-sequence classification.

- Input format: `N × 128 × 9`
- 128 time steps per sequence
- 9 sensor features at each time step
- 6 activity classes
- Train/validation/test split: **70% / 15% / 15%**
- Stratified splitting is used
- Normalization statistics are calculated using the training data only

### 3. Temporal Data Visualization

Sensor signals are visualized across different activity classes and channels, including:

- `body_acc_x`
- `body_gyro_x`
- `total_acc_x`

### 4. BPTT Numerical Exercise

A simple RNN recurrence is evaluated step-by-step using:

```text
h_t = tanh(W_x x_t + W_h h_(t-1) + b)
```

The notebook computes the hidden states programmatically so that they can be compared with the corresponding hand calculation.

### 5. RNN / LSTM / GRU Model Builder

A common model structure is used for a controlled comparison. Only the recurrent layer is changed:

- SimpleRNN
- LSTM
- GRU

The models use recurrent units followed by dropout, a dense layer and a softmax classification output.

### 6. RNN, LSTM and GRU Training

The three recurrent models are trained using the full sequence length:

```text
T = 128
```

Training and validation loss/accuracy curves are plotted for comparison.

### 7. Test-set Evaluation

The trained models are evaluated on the test set using:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Confusion matrices

### 8. Model Comparison

A consolidated comparison table and bar plot are generated for the RNN, LSTM and GRU models.

The comparison includes:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Number of parameters
- Training time

### 9. Effect of Sequence Length

The recurrent models are tested with:

```text
T ∈ {32, 64, 128}
```

The experiment examines how reducing or retaining the sequence length affects model performance.

### 10. Video Understanding

A small **UCF101 subset** is used for video classification.

The notebook performs:

1. Video loading
2. Frame sampling
3. Image resizing
4. Feature extraction using **MobileNetV2**
5. Temporal sequence modelling using LSTM/GRU
6. Classification and evaluation

Sample video frames and training/evaluation plots are generated.

### 11. Sequence-to-Sequence Learning

An encoder–decoder LSTM is trained on a synthetic sequence reversal task.

Example:

```text
Input:  [1, 4, 7, 2]
Output: [2, 7, 4, 1]
```

The implementation uses **teacher forcing** during training.

Evaluation includes:

- Token accuracy
- Complete sequence accuracy
- Sample predictions
- Training loss

### 12. Consolidated Results

The notebook collects the major experimental results into a final consolidated table for easier comparison.

---

## Datasets

### UCI HAR Dataset

The notebook downloads the UCI HAR Dataset automatically when it is not already available in the Colab environment.

The notebook uses the raw inertial signals provided by the dataset.

### UCF101 Subset

For the video experiment, the notebook uses a small UCF101 subset suitable for the laboratory experiment rather than requiring the complete multi-gigabyte UCF101 dataset.

The notebook contains download logic for obtaining the required subset.

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- OpenCV
- Google Colab
- MobileNetV2
- SimpleRNN
- LSTM
- GRU

---

## Running the Notebook

The notebook is designed primarily for **Google Colab**.

### Step 1: Open the notebook

Upload:

```text
Experiment_6 (1).ipynb
```

to Google Colab.

### Step 2: Run the cells in order

The notebook downloads the required datasets and performs preprocessing automatically.

It is recommended to execute the notebook from the first cell to the last because later sections depend on variables and trained models created earlier.

### Step 3: GPU

A GPU runtime is recommended, especially for:

- RNN/LSTM/GRU training
- MobileNetV2 feature extraction
- Video classification
- Sequence-to-sequence training

In Google Colab:

```text
Runtime → Change runtime type → T4 GPU
```

if a GPU is available.

---

## Reproducibility

The notebook sets:

```python
SEED = 42
```

and uses the seed for the Python, NumPy and data-splitting operations where applicable.

Training time and some neural-network results can still vary slightly depending on the hardware and TensorFlow execution environment.

---

## Generated Visualizations

The notebook generates visualizations including:

- Temporal sensor-signal plots
- RNN/LSTM/GRU training and validation loss curves
- RNN/LSTM/GRU training and validation accuracy curves
- Confusion matrices
- Model-comparison bar plots
- Sequence-length comparison plots
- Sample video frames
- CNN–LSTM/CNN–GRU training curves and evaluation plots

---

## Code Repository

The complete implementation is available on GitHub:

**https://github.com/navanthiga/Deep-Learning-Lab**

---

## Files

```text
Deep-Learning-Lab/
│
├── Experiment_6 (1).ipynb
└── README_Experiment_6.md
```

---

## Notes

- Run the notebook cells sequentially.
- Internet access is required for the first-time dataset downloads.
- A GPU runtime is recommended for faster execution.
- The notebook contains both numerical calculations and deep-learning experiments.
- Some training times and model metrics may vary slightly between runs.

