import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from docx import Document
from docx.shared import Inches

# Load the dataset
file_path = 'path_to_your_file/cancer patient data sets.csv'
data = pd.read_csv(file_path)

# Encode the target variable
label_encoder = LabelEncoder()
data['Level'] = label_encoder.fit_transform(data['Level'])

# Define features and target
X = data.drop(columns=['index', 'Patient Id', 'Level'])
y = data['Level']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize classifiers
svm_classifier = SVC(probability=True, random_state=42)
knn_classifier = KNeighborsClassifier()
nb_classifier = GaussianNB()

# Train the classifiers
svm_classifier.fit(X_train, y_train)
knn_classifier.fit(X_train, y_train)
nb_classifier.fit(X_train, y_train)

# Predict the results
y_pred_svm = svm_classifier.predict(X_test)
y_pred_knn = knn_classifier.predict(X_test)
y_pred_nb = nb_classifier.predict(X_test)

# Calculate accuracies
accuracy_svm = accuracy_score(y_test, y_pred_svm)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
accuracy_nb = accuracy_score(y_test, y_pred_nb)

# Calculate confusion matrices
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
conf_matrix_nb = confusion_matrix(y_test, y_pred_nb)

# Generate ROC curves and AUC
y_prob_svm = svm_classifier.predict_proba(X_test)
y_prob_knn = knn_classifier.predict_proba(X_test)
y_prob_nb = nb_classifier.predict_proba(X_test)

fpr_svm, tpr_svm, _ = roc_curve(y_test, y_prob_svm[:, 1], pos_label=1)
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_prob_knn[:, 1], pos_label=1)
fpr_nb, tpr_nb, _ = roc_curve(y_test, y_prob_nb[:, 1], pos_label=1)

roc_auc_svm = auc(fpr_svm, tpr_svm)
roc_auc_knn = auc(fpr_knn, tpr_knn)
roc_auc_nb = auc(fpr_nb, tpr_nb)

# Plot confusion matrices
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.heatmap(conf_matrix_svm, annot=True, fmt='d', cmap='Blues')
plt.title('SVM Confusion Matrix')

plt.subplot(1, 3, 2)
sns.heatmap(conf_matrix_knn, annot=True, fmt='d', cmap='Blues')
plt.title('KNN Confusion Matrix')

plt.subplot(1, 3, 3)
sns.heatmap(conf_matrix_nb, annot=True, fmt='d', cmap='Blues')
plt.title('Naive Bayes Confusion Matrix')

plt.tight_layout()
plt.savefig('conf_matrix.png')
plt.show()

# Plot ROC curves
plt.figure(figsize=(10, 7))

plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC = {roc_auc_svm:.2f})')
plt.plot(fpr_knn, tpr_knn, label=f'KNN (AUC = {roc_auc_knn:.2f})')
plt.plot(fpr_nb, tpr_nb, label=f'Naive Bayes (AUC = {roc_auc_nb:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.savefig('roc_curve.png')
plt.show()

# Create a new Document
doc = Document()
doc.add_heading('Disease Prediction Model Comparison', 0)

# Add introduction
doc.add_paragraph('This document compares the performance of three classifiers (SVM, KNN, Naive Bayes) '
                  'in predicting disease severity levels. The dataset includes various patient characteristics, '
                  'and the target variable is the severity level of the disease (Low, Medium, High).')

# Add accuracies
doc.add_heading('Classifier Accuracies', level=1)
doc.add_paragraph(f'SVM Accuracy: {accuracy_svm * 100:.2f}%')
doc.add_paragraph(f'KNN Accuracy: {accuracy_knn * 100:.2f}%')
doc.add_paragraph(f'Naive Bayes Accuracy: {accuracy_nb * 100:.2f}%')

# Add confusion matrices
doc.add_heading('Confusion Matrices', level=1)
doc.add_picture('conf_matrix.png', width=Inches(6))

# Add ROC curves
doc.add_heading('ROC Curves', level=1)
doc.add_picture('roc_curve.png', width=Inches(6))

# Save the document
doc.save('Disease_Prediction_Model_Comparison.docx')
