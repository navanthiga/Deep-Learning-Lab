# CS3807 – Deep Learning Laboratory
## Experiment 7: Autoencoders, Convolutional Autoencoders, Denoising Autoencoders and Variational Autoencoders

This repository contains the implementation for **Experiment 7 of CS3807 – Deep Learning Laboratory**.

The experiment studies different types of autoencoders using the **MNIST handwritten digit dataset**, starting from a fully connected autoencoder and progressing to convolutional, denoising, and variational autoencoders.

---

## 📌 Objective

The main objective of this experiment is to understand how autoencoders can be used for:

- Image representation
- Image reconstruction
- Image denoising
- Latent-space visualization
- Generative modelling

The complete pipeline followed in the experiment is:

```text
MNIST Image
     ↓
Encoder
     ↓
Latent Representation
     ↓
Decoder
     ↓
Reconstructed Image
     ↓
Evaluation
```

For denoising:

```text
Clean Image
     ↓
Add Noise
     ↓
Denoising Autoencoder
     ↓
Denoised Reconstruction
```

For the VAE:

```text
Image
  ↓
Encoder
  ↓
μ, log σ²
  ↓
Reparameterization
  ↓
z
  ↓
Decoder
  ↓
Generated / Reconstructed Image
```

---

## 📂 Repository Contents

```text
.
├── CS3807_Experiment_7_Autoencoders_VAE.ipynb
└── README.md
```

The Jupyter/Colab notebook contains the complete implementation, visualizations, evaluation metrics, and additional experiments.

---

## 🧰 Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- scikit-image
- Google Colab / Jupyter Notebook

---

## 📊 Dataset

The experiment uses the **MNIST Handwritten Digit Dataset**.

Each image is:

```text
28 × 28 × 1
```

Pixel values are normalized from:

```text
[0, 255] → [0, 1]
```

For the main laboratory execution:

- Training images: **10,000**
- Test images: **2,000**

The digit labels are not used as reconstruction targets. The original image itself is used as the target.

---

# 🔬 Experiments Implemented

## 1. Fully Connected Autoencoder

The fully connected autoencoder uses the following architecture:

```text
784
 ↓
Dense(128, ReLU)
 ↓
Dense(32, ReLU)
 ↓
Latent(16)
 ↓
Dense(32, ReLU)
 ↓
Dense(128, ReLU)
 ↓
Dense(784, Sigmoid)
```

Configuration:

| Parameter | Value |
|---|---|
| Latent dimension | 16 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 128 |
| Epochs | 20 |
| Loss | Binary Cross-Entropy |

The model is evaluated using:

- MSE
- MAE
- SSIM

The notebook also plots training and validation reconstruction loss and compares original images with reconstructed images.

---

## 2. Convolutional Autoencoder

The convolutional autoencoder preserves the spatial structure of the input image.

Architecture:

```text
Input (28×28×1)
      ↓
Conv2D(32)
      ↓
MaxPooling2D
      ↓
Conv2D(64)
      ↓
MaxPooling2D
      ↓
Conv2D(64)
      ↓
UpSampling2D
      ↓
Conv2D(32)
      ↓
UpSampling2D
      ↓
Conv2D(1, Sigmoid)
```

The FC-AE and CAE are compared using both visual reconstruction quality and quantitative metrics.

---

## 3. Denoising Convolutional Autoencoder

The denoising autoencoder receives a corrupted image as input but uses the original clean image as the target.

Two types of noise are implemented.

### Gaussian Noise

The notebook evaluates:

```text
σ = 0.1
σ = 0.2
σ = 0.3
```

The noisy image is generated using:

```text
x̃ = x + n
```

where:

```text
n ~ N(0, σ²)
```

The resulting image is clipped to `[0, 1]`.

### Salt-and-Pepper Noise

The notebook evaluates:

```text
p = 0.05
p = 0.10
p = 0.20
```

Pixels are randomly replaced with values close to 0 or 1.

The denoising models are evaluated using:

- MSE
- MAE
- SSIM

---

# 4. Variational Autoencoder

The VAE uses a **2-dimensional latent space**.

The encoder produces:

```text
μ ∈ R²
log σ² ∈ R²
```

A latent vector is sampled using the reparameterization trick:

```text
z = μ + σ ⊙ ε
```

where:

```text
ε ~ N(0, I)
```

The VAE loss consists of two components:

```text
VAE Loss = Reconstruction Loss + KL Loss
```

The KL divergence encourages the learned latent distribution to remain close to the standard normal prior:

```text
N(0, I)
```

The notebook includes:

- VAE reconstruction
- Reconstruction loss
- KL divergence loss
- Total VAE loss
- 2D latent-space visualization
- Random image generation
- Latent-space interpolation

---

# 📈 Evaluation Metrics

Three main image-reconstruction metrics are used.

### Mean Squared Error

```text
MSE = mean((x - x̂)²)
```

### Mean Absolute Error

```text
MAE = mean(|x - x̂|)
```

### Structural Similarity Index

SSIM is used to measure structural similarity between the original and reconstructed images.

---

# 📊 Visualizations

The notebook generates the required visualizations, including:

1. Original vs reconstructed images
2. Training and validation loss for the basic autoencoder
3. Fully connected AE vs convolutional AE
4. Clean vs noisy vs denoised images
5. Noise level vs MSE, MAE and SSIM
6. VAE 2D latent-space visualization
7. Randomly generated VAE images
8. Latent-space interpolation
9. VAE training and validation losses
10. Reconstruction-error distribution
11. Five highest reconstruction-error samples
12. Latent dimension vs reconstruction MSE

---

# 🧪 Additional Experiments

The notebook also includes the additional exercises from the experiment.

### FC-AE Latent Dimension Study

Latent dimensions:

```text
2, 8, 16, 32
```

The reconstruction MSE and SSIM are compared.

### CAE Bottleneck Study

Bottleneck channel dimensions:

```text
4, 8, 16, 32
```

### Transposed-Convolution Decoder

A CAE using `Conv2DTranspose` is implemented and compared with the standard upsampling-based decoder.

### VAE Latent Dimension 8

A second VAE is implemented with:

```text
latent dimension = 8
```

### KL-Weight Study

Different KL-loss contributions are investigated using:

```text
KL weights = 0.1, 0.5, 1.0, 2.0
```

### Larger VAE Generation

A larger grid containing **100 randomly generated VAE samples** is generated.

### Noise-Type Comparison

Gaussian noise and salt-and-pepper noise are compared using the same denoising architecture.

### Noise-Level Comparison

Two separately trained denoising models are used to compare reconstruction quality at different Gaussian noise levels.

---

# 📁 Result Files

The notebook includes an export section that saves important numerical results as CSV files.

The generated files include:

```text
consolidated_model_results.csv
gaussian_noise_results.csv
salt_pepper_results.csv
fc_latent_dimension_results.csv
cae_latent_dimension_results.csv
kl_weight_results.csv
```

These files contain the numerical results obtained from the student's execution of the notebook.

---

# ▶️ How to Run

## Option 1: Google Colab

1. Open Google Colab.
2. Upload `CS3807_Experiment_7_Autoencoders_VAE.ipynb`.
3. Select:

```text
Runtime → Change runtime type → T4 GPU
```

4. Run the notebook from top to bottom.

A GPU is recommended because several neural-network models are trained during the experiment.

## Option 2: Jupyter Notebook

Install the required libraries:

```bash
pip install tensorflow numpy pandas matplotlib scikit-image
```

Then launch Jupyter:

```bash
jupyter notebook
```

Open the notebook and run the cells sequentially.

---

# 📝 Discussion and Analysis

The notebook also includes the discussion questions associated with the experiment, covering:

- Autoencoder architecture
- Encoder and decoder roles
- Latent representations
- Effect of latent dimension
- Convolutional vs fully connected reconstruction
- Denoising autoencoders
- Noise levels
- MSE, MAE and SSIM
- Variational autoencoders
- Reparameterization trick
- KL divergence
- VAE prior distribution
- Latent-space visualization
- Latent-space interpolation
- VAE image generation
- Reconstruction vs generation
- Train/test separation

---

# 🎯 Expected Learning Outcomes

After completing the experiment, the implementation demonstrates the ability to:

- Build a fully connected autoencoder.
- Build a convolutional autoencoder.
- Add controlled image corruption.
- Build a denoising autoencoder.
- Evaluate image reconstruction using MSE, MAE and SSIM.
- Visualize deterministic and probabilistic latent representations.
- Implement the VAE reparameterization trick.
- Calculate reconstruction and KL losses.
- Generate new images using a VAE.
- Perform latent-space interpolation.
- Analyse reconstruction errors.
- Study the effect of latent dimensionality and KL weighting.

---

## 👤 Author

**Navan**

B.Tech Artificial Intelligence & Data Science  
Shiv Nadar University Chennai

**Course:** CS3807 – Deep Learning Laboratory  
**Experiment:** 7

---

## 📚 References

1. Ian Goodfellow, Yoshua Bengio and Aaron Courville, *Deep Learning*, MIT Press, 2016.
2. Diederik P. Kingma and Max Welling, *Auto-Encoding Variational Bayes*, ICLR, 2014.
3. Pascal Vincent et al., *Stacked Denoising Autoencoders: Learning Useful Representations with a Local Denoising Criterion*, JMLR, 2010.
4. Yann LeCun, Corinna Cortes and Christopher J. C. Burges, *MNIST Handwritten Digit Database*.
5. TensorFlow Documentation
6. Keras Documentation
7. scikit-image Documentation
