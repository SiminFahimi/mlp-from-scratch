# Neural Network from Scratch (NumPy)

A fully connected feedforward neural network implemented entirely in Python using NumPy — without relying on deep learning frameworks such as PyTorch or TensorFlow.

This project is designed to provide a deep, hands-on understanding of neural networks by implementing every component from first principles: forward propagation, backpropagation, optimization, and evaluation.

---

## Key Features

- **Pure NumPy implementation** (no ML frameworks)
- Forward and backward propagation from scratch
- Mini-batch gradient descent
- Binary cross-entropy loss
- L2 regularization (weight decay)
- **He initialization** (hidden layers) and **Xavier initialization** (output layer)
- Numerical gradient checking (finite differences)
- K-fold cross-validation
- Automated hyperparameter search (learning rate and regularization strength)
- Feature engineering (polynomial expansion + Pearson correlation selection)
- Visualization tools for training dynamics and experiments

---

## Project Structure
.
```
├── main.py # Entry point — runs full pipeline
├── model.py # Neural network implementation (forward, backward, training)
├── build_network.py # Model construction and configuration
├── data.py # Data loading, preprocessing, feature engineering
├── eval.py # Cross-validation and hyperparameter tuning
├── plots.py # Visualization utilities
├── utils.py # Activation functions and derivatives
├── results/
│ ├── loss.png
│ ├── lr_effect.png
│ ├── lambda_effect.png
│ ├── size_effect_on_accuracy.png
│ └── size_effect_on_cost.png
└── README.md
```
## Dataset

- **Breast Cancer Wisconsin (Original) dataset** (UCI ID: 15)
- Binary classification:
  - Malignant → `1`
  - Benign → `0`
- Missing values are imputed using column medians
- ID column (`Sample_code_number`) is removed
- Data split: **80% training / 20% testing**

---

## Model Architecture
Input Layer (n features)
↓ Hidden Layer 1: 16 units, ReLU
↓ Hidden Layer 2: 8 units, ReLU
↓ Output Layer: 1 unit, Sigmoid

text

All architectural choices (layer sizes, activations, regularization) are configurable via `build_network.py`.

---

## Feature Engineering

Optional feature augmentation based on **Pearson correlation**:

- Select the top **2 features** most correlated with the target
- Append their squared values (\(x^2\)) to the input

### Performance Comparison

| Input Type           | Test Accuracy |
|----------------------|---------------|
| Raw features         | 0.9587        |
| Engineered features  | 0.9663        |

The improvement is consistent, though modest.

---

## Gradient Checking

Backpropagation correctness is verified using **finite differences**:

\[
\frac{\partial J}{\partial \theta} \approx \frac{J(\theta + \varepsilon) - J(\theta - \varepsilon)}{2\varepsilon}
\]

- Checked on **10 random parameters per layer**
- Executed during the **first epoch**
- Enable via: `debug=True` in `model.fit(...)`

---

## Hyperparameter Tuning

### Search Space

- **Learning rate:** `(0.005, 0.01, 0.02, 0.05, 0.1, 0.5)`
- **L2 regularization (\(\lambda\)):** `(0, 0.002, 0.005, ..., 10.24)`

### Method

- **3-fold cross-validation**
- Selection based on **minimum validation loss**

---

## Experiments & Results

The project includes multiple experiment analyses:

- Training loss over epochs
- Effect of dataset size on performance
- Impact of regularization strength (\(\lambda\))
- Learning rate sensitivity

*(See images in the `results/` directory)*

---

## Installation & Usage

```bash
pip install -r requirements.txt
python main.py
```

---

## Possible Extensions

- Adam / RMSProp optimizers
- Multi-class classification (softmax output)
- Dropout regularization
- Early stopping
- PyTorch reimplementation for benchmarking
- More advanced feature selection methods

---

## Notes

- Built entirely with NumPy for clarity and educational value
- Bias units are added during forward propagation (not stored separately)
- L2 regularization excludes bias weights (standard practice)
