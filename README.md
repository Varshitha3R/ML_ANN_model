
# Patient Risk Prediction Model

This project aims to predict patient risk levels based on their medical data using a deep learning model (Artificial Neural Network). The model categorizes patients into three risk levels (Low, Mild, Severe) based on various features such as age, BMI, physical fitness, and other clinical measurements. The system consists of two main components: the backend that handles model training and prediction, and the frontend that provides a user interface for input and results.

## Project Structure

```
.
├── ANN_model.py                # Contains code for defining, training, and evaluating the ANN model
├── prediction.py               # Contains code for handling predictions with the trained model
├── prediction.html             # HTML frontend for making predictions based on user input
├── train.csv                   # Training data
├── test.csv                    # Test data
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Installation

1. Clone the repository to your local machine:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Files Description

### `ANN_model.py`
This Python script defines, trains, and evaluates the Artificial Neural Network (ANN) model. It:
- Loads the training data (`train.csv`) and preprocesses it.
- Defines an ANN with two hidden layers (64 and 32 neurons respectively) and a softmax output layer for classification.
- Trains the model using categorical cross-entropy loss and the Adam optimizer.
- Saves the trained model for later use in making predictions.

### `prediction.py`
This script handles the prediction process for new patient data. It:
- Loads the trained model from `ANN_model.h5` (model file saved after training).
- Takes the user input via the `prediction.html` frontend.
- Preprocesses the input data.
- Outputs the predicted risk level (`Low`, `Mild`, or `Severe`) for the patient based on the input.

### `prediction.html`
The frontend HTML file allows users to input their medical information, such as age, BMI, and other features. It uses basic form elements for input and then interacts with the `prediction.py` script (via Flask or a similar framework) to display the results.

### `train.csv`
This file contains the training data used for training the model. It includes various patient features such as age, BMI, and physical performance metrics. The target variable `sii` (Severity Indicator Index) is used to categorize the risk level into `Low`, `Mild`, or `Severe`.

### `test.csv`
This file contains the test data, which is used to evaluate the trained model.

## Model Overview

The model is a simple feed-forward neural network built with Keras (TensorFlow). The input features include patient data such as:
- Age
- Sex
- BMI
- Fitness Endurance (Max Stage)
- Other clinical measures

The target variable `sii` (Severity Indicator Index) is transformed into categorical labels (`Low`, `Mild`, `Severe`) using a custom categorization function. The model uses a softmax output layer for classification.

### Risk Level Categorization
- **Low risk**: `sii` < 1
- **Mild risk**: 1 <= `sii` < 2
- **Severe risk**: `sii` >= 2

## Running the Project

### Training the Model

To train the model:
1. Run `ANN_model.py` to load the training data, preprocess it, define the model, and start training.
   ```bash
   python ANN_model.py
   ```

2. The trained model will be saved as `ANN_model.h5` after training.

### Making Predictions

To use the trained model for making predictions, the user can enter their data via the frontend (`prediction.html`).

1. Run the prediction backend script (e.g., using Flask or another framework):
   ```bash
   python prediction.py
   ```

2. Open the `prediction.html` in a browser to input new patient data and receive predictions for the risk level.

### Model Evaluation

After training, the model will be evaluated on the test set, and the test accuracy will be displayed in the terminal.

### Visualizing Results

The training and validation accuracy and loss curves will be plotted after training. These plots help visualize how well the model is learning over the epochs.

## Requirements

- Python 3.x
- TensorFlow
- Keras
- scikit-learn
- pandas
- numpy
- matplotlib

To install the required dependencies:
```bash
pip install -r requirements.txt
```

### `requirements.txt`

```text
tensorflow
keras
scikit-learn
pandas
numpy
matplotlib
flask
```

## Example Use Case

To test the model, you can load the test data from `test.csv` and use the model to predict the risk level of patients based on their features. The results can then be visualized and analyzed for further improvement of the model.

---

