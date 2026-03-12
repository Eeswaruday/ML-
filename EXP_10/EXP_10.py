import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
#Load Dataset:
iris = load_iris()
X = iris.data
y = iris.target
#Split Dataset into Training and Testing Sets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#Instantiate the Decision Tree Classifier:
dt_classifier = DecisionTreeClassifier(random_state=42)\
#Define Hyperparameters for Tuning:
param_grid = {
'criterion': ['gini', 'entropy'],
'max_depth': [3, 5, 10, None],
'min_samples_split': [2, 5, 10],
'min_samples_leaf': [1, 2, 4]
}
#Perform Grid Search with Cross-Validation:to find the best combination of hyperparameters.
grid_search = GridSearchCV(estimator=dt_classifier, param_grid=param_grid,
cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)
#Evaluate the Best Model:
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Best Hyperparameters: {grid_search.best_params_}')
print(f'Test Set Accuracy: {accuracy:.4f}')