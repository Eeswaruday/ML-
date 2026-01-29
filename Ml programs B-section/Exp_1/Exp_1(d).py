import pandas as pd
from sklearn.datasets import load_iris
# Load Iris dataset
iris_data = load_iris()
# Create DataFrame
iris_df = pd.DataFrame(
iris_data.data,
columns=iris_data.feature_names
)
# Add class column
iris_df['class'] = iris_data.target
# Display DataFrame
print("Iris DataFrame:")
print(iris_df.head())
#
# In a new cell
print(iris_df.describe())
# In a new cell
print(iris_df.info())