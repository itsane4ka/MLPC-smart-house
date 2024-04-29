import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from data_loader import AudioDataset
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from model import SpeechRecognitionModel
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load the features
features = np.load(r'development_numpy\development.npy')

# Load the metadata for labels
metadata = pd.read_csv(r'metadata\development.csv')
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(metadata['word'].values)  # Ensure the column name matches

# Check unique labels
labels_unique = np.unique(np.array(metadata['word'].values))

# Split the dataset into training and test sets
# Very important is to shuffle it!
X_train, X_test, y_train, y_test = train_test_split(features, labels, shuffle=True, test_size=0.25, random_state=42)

print(f'Training set shape: {X_train.shape}')
print(f'Test set shape: {X_test.shape}')
print(f'Training labels shape: {y_train.shape}')
print(f'Test labels shape: {y_test.shape}\n')

# Reshape the feature data to fit the Random Forest model
# Flattening the time steps and features into one dimension per sample
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Print the new shapes to verify
print("New training set shape:", X_train_flat.shape)
print("New test set shape:", X_test_flat.shape)

# Initialize and fit Quadratic Discriminant Analysis
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train_flat, y_train)
y_pred_qda = qda.predict(X_test_flat)
accuracy_qda = accuracy_score(y_test, y_pred_qda)
print("Accuracy of Quadratic Discriminant Analysis:", accuracy_qda)


# Result: accuracy of the Quadratic Discriminant Analysis classifier: 0.77