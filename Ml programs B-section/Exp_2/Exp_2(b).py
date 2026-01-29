#ii.Bar plots-iris example
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
# Load Iris dataset
iris = load_iris()
# Create DataFrame
df = pd.DataFrame(
iris.data,
columns=iris.feature_names
)
# Add class labels
df['class'] = iris.target
# Compute mean of each feature grouped by class
mean_values = df.groupby('class').mean()
# Plot bar graph using pandas inbuilt visualization
mean_values.plot(kind='bar')
# Display plot
plt.title("Mean Feature Values for Each Iris Class")
plt.xlabel("Class")
plt.ylabel("Mean Value")
plt.show()