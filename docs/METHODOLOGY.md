# Methodology

## Objective

This repository independently reimplements the paper's central machine-learning pipeline: hyperspectral spectra are reduced by ANOVA F-score band selection and then classified with MLP, 1D-CNN, LSTM, and CNN-LSTM networks.

## Implemented workflow

1. Load labelled spectra.
2. Create a stratified 80/20 train-test split.
3. Reserve 10% of the training portion for validation.
4. Fit ANOVA `SelectKBest(f_classif)` on the effective training data only.
5. Retain 162 of the original 204 bands.
6. Fit `StandardScaler` on selected training features only.
7. Train a PyTorch model with Adam, cross-entropy, dropout, learning-rate reduction, and early stopping.
8. Evaluate on the untouched test set using accuracy, macro precision, macro recall, macro F1, per-class metrics, and a confusion matrix.

## Leakage controls

Band selection and standardization are fitted exclusively on the effective training subset. Validation and test samples are transformed using those fitted objects. This prevents test information from influencing feature selection or normalization.

## Models

- **MLP:** 512, 256, and 128-unit ReLU layers with dropout 0.3.
- **1D-CNN:** two 1D convolution blocks followed by a dense classifier.
- **LSTM:** one 128-unit recurrent layer operating over the selected bands.
- **CNN-LSTM:** a 1D convolutional front end followed by a 64-unit LSTM.

The paper does not specify every lower-level architectural detail for all models. Any necessary engineering choices are therefore documented as independent implementation decisions rather than represented as official author code.

## Validation datasets

### Salinas

Salinas is a real, public 204-band AVIRIS hyperspectral benchmark. It verifies that the implementation works on genuine hyperspectral measurements. Its 16 agricultural classes are different from the paper's nine beverage classes.

### Synthetic beverage spectra

The repository can generate structured synthetic spectra for the paper's nine beverage labels. This validates the software interface and exact 9-class shape, but cannot establish forensic accuracy.
