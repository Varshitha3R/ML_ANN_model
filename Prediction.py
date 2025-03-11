import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.utils.class_weight import compute_class_weight

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load and preprocess the dataset
data = pd.read_csv('C:\\Users\\Admin\\Downloads\\child-mind-institute-problematic-internet-use\\train.csv')  # Update path as necessary

# Check for missing values and handle them
data.isnull().sum()

# Fill missing values with the mean for numerical columns
data['Basic_Demos-Age'].fillna(data['Basic_Demos-Age'].mean(), inplace=True)
data['Physical-HeartRate'].fillna(data['Physical-HeartRate'].mean(), inplace=True)
data['Physical-BMI'].fillna(data['Physical-BMI'].mean(), inplace=True)

# Filter data to include only children in the age range of 5 to 13
data = data[(data['Basic_Demos-Age'] >= 5) & (data['Basic_Demos-Age'] <= 13)]

# Create the target variable 'risk_level' based on 'sii'
def categorize_risk(sii):
    if sii < 1:
        return 0  # Low risk
    elif 1 <= sii < 2:
        return 1  # Mild risk
    else:
        return 2  # Severe risk

# Apply this function to the 'sii' column to create the 'risk_level' target variable
data['risk_level'] = data['sii'].apply(categorize_risk)

# Preprocess the data
X = data[['Basic_Demos-Age', 'Physical-HeartRate', 'Physical-BMI']].values
y = data['risk_level'].values  # Target variable is now the categorized 'risk_level'

# Convert the labels into one-hot encoding
y = to_categorical(y, num_classes=3)  # Three categories: 0, 1, 2 (Low, Mild, Severe)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Compute class weights to address class imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(y_train.argmax(axis=1)), y=y_train.argmax(axis=1))
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Build the ANN model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))  # First hidden layer
model.add(Dense(32, activation='relu'))  # Second hidden layer
model.add(Dense(3, activation='softmax'))  # Output layer with 3 neurons (for 3 classes)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model with class weights
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), class_weight=class_weight_dict)

# Save the trained model
model.save('our_trained_model.h5')

# Function to predict risk level for new input values (age, heart rate, BMI)
@app.route('/api/predict', methods=['POST'])
def predict_risk():
    # Get JSON data from the request
    data = request.get_json()

    # Extract features from the incoming request
    age = data['age']
    heart_rate = data['heart_rate']
    bmi = data['bmi']

    # Prepare the input data for prediction (same preprocessing as in training)
    input_data = np.array([[age, heart_rate, bmi]])
    input_data = scaler.transform(input_data)  # Scale the input data

    # Make predictions
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction, axis=1)[0]  # Get the class index (0: Low, 1: Mild, 2: Severe)

    # Map the class index to a label
    class_labels = {0: 'Low', 1: 'Mild', 2: 'Severe'}
    risk_level = class_labels[predicted_class]

    # Return the prediction result as a JSON response
    return jsonify({'risk_level': risk_level})

if __name__ == '__main__':
    app.run(debug=True)
