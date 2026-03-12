# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = load_iris()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Create an SVM classifier with a linear kernel
svm = SVC(kernel='linear')

# Train the SVM classifier on the training set
svm.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = svm.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Define the parameter grid for GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'degree': [2, 3, 4],
    'coef0': [0.0, 0.1, 0.5],
    'gamma': ['scale', 'auto']
}

# Create an SVM classifier
svm = SVC()

# Perform grid search to find the best set of parameters
grid_search = GridSearchCV(svm, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Print the best set of parameters and their accuracy
print("Best parameters:", grid_search.best_params_)
print("Best accuracy:", grid_search.best_score_)

# Use the best model from grid search to make predictions
best_svm = grid_search.best_estimator_
y_pred_best = best_svm.predict(X_test)

# Calculate the accuracy of the best model
accuracy_best = accuracy_score(y_test, y_pred_best)
print("Best Model Accuracy:", accuracy_best)

# Sample prediction for a new data point
new_data_point = [[5.1, 3.5, 1.4, 0.2]]  # Example new data point
predicted_class = best_svm.predict(new_data_point)
print("Predicted class for new data point:", predicted_class)
print("Predicted class name:", iris.target_names[predicted_class[0]])
