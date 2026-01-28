# Learning-Probability-Density-Functions-using-data
# Air Quality Analysis – NO₂ Probability Density Estimation using GAN

---

## Overview

This project analyzes air quality measurements with a specific focus on **Nitrogen Dioxide (NO₂)** concentrations.  
The objective is to apply a **roll-number-dependent nonlinear transformation** to the NO₂ data and estimate the **probability density function (PDF)** of the transformed variable using a **Generative Adversarial Network (GAN)**.

The estimated distribution obtained from GAN-generated samples is compared with the empirical distribution using **histograms** and **Kernel Density Estimation (KDE)**.

---

## Dataset

- **Source File:** `data.csv`  
- **Feature Analyzed:** `no2`  
- **Data Type:** Continuous numerical values  

**Preprocessing Details:**
- The dataset is loaded using the Pandas library.
- The `no2` column is selected for analysis.
- Any missing or non-numeric values are removed to ensure data consistency.

---

## Methodology

### 1. Data Preprocessing

- The dataset is read from the CSV file using Pandas.
- The `no2` feature is extracted and converted to numeric format.
- Invalid or missing entries are excluded from further analysis.

---

### 2. Data Transformation

The original NO₂ observations \( x \) are transformed into a new variable \( z \) using the following expression:

\[
z = x + a_r \cdot \sin(b_r \cdot x)
\]

**Roll Number:** `102303927`

Constants derived from the roll number:
- \( a_r = 0.5 \times (r \bmod 7) \)
- \( b_r = 0.3 \times ((r \bmod 5) + 1) \)

After transformation, the data is **normalized** to improve training stability.

---

### 3. GAN-based PDF Estimation

A **1-Dimensional Generative Adversarial Network (GAN)** is used to learn the distribution of the transformed variable \( z \).

#### Generator
- Input: 1D Gaussian noise
- Fully connected neural network
- Output: Synthetic samples resembling transformed NO₂ values

#### Discriminator
- Input: Real or generated samples
- Fully connected neural network
- Output: Probability indicating whether a sample is real or generated

**Training Configuration:**
- Loss Function: Binary Cross Entropy (BCE)
- Optimizer: Adam
- Learning Rate: 0.0002
- Batch Size: 128
- Epochs: 2000
- Device: CPU / GPU (CUDA if available)

---

## Results

The GAN successfully learns the underlying probability distribution of the transformed NO₂ data.

### Observations:
- The histogram of GAN-generated samples closely matches the histogram of real transformed data.
- KDE plots show strong overlap between real and generated PDFs.
- This confirms the effectiveness of GANs for **1D density estimation** tasks.

---

## Visualization

The following visualizations are generated for analysis:

- **Histogram Comparison**
  - Real transformed data vs GAN-generated samples
- **Kernel Density Estimation (KDE) Plot**
  - Smooth PDF comparison between real and generated data

All plots are displayed during execution and can be saved for reference.

---

## Requirements

The following libraries are required to run the project:

```bash
pip install numpy pandas matplotlib scikit-learn
pip install torch torchvision torchaudio
