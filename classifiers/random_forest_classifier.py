import numpy as np
import pandas as pd 
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from data_loader import AudioDataset
from model import SpeechRecognitionModel
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

# Load the features
features = np.load(r'development_numpy\development.npy')

# Load the metadata for labels
metadata = pd.read_csv(r'metadata\development.csv')
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(metadata['word'].values)  # Ensure the column name matches

# Check unique labels
labels_unique = np.unique(np.array(metadata['word'].values))

print(labels_unique)

# Split the dataset into training and test sets
# Very important is to shuffle it!
X_train, X_test, y_train, y_test = train_test_split(features, labels, shuffle=True, test_size=0.3, random_state=42)

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

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=20, random_state=42)

# Train the classifier on the flattened data
rf_classifier.fit(X_train_flat, y_train)

# Predict on the test set
y_pred = rf_classifier.predict(X_test_flat)

# Calculate accuracy of the predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the Random Forest classifier:", accuracy)


# Result: accuracy of the Random Forest classifier: 0.87

# Generate classification report
class_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print("Classification Report:\n", class_report)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=label_encoder.classes_)
disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
plt.title("Confusion Matrix")
plt.show()

# Binarize the labels for multiclass classification
y_test_bin = label_binarize(y_test, classes=np.arange(len(label_encoder.classes_)))
n_classes = y_test_bin.shape[1]

# Get the predicted probabilities for each class
y_score = rf_classifier.predict_proba(X_test_flat)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curve for each class
plt.figure(figsize=(12, 8))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'blue', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow', 'black', 'teal', 'lime', 'navy', 'silver', 'maroon', 'gold'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(label_encoder.classes_[i], roc_auc[i]))
    plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4,
         label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid()
plt.show()

# Calculate and print ROC AUC score for each class and the average
roc_auc_avg = roc_auc_score(y_test_bin, y_score, multi_class="ovr")
print("Average ROC AUC Score (One-vs-Rest):", roc_auc_avg)

importances = rf_classifier.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
for f in range(X_train_flat.shape[1]):
    print(f"{f + 1}. Feature {indices[f]} ({importances[indices[f]]})")

# Plot the feature importances
plt.figure(figsize=(12, 8))
plt.title("Feature importances")
plt.bar(range(X_train_flat.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train_flat.shape[1]), indices)
plt.xlim([-1, X_train_flat.shape[1]])
plt.show()