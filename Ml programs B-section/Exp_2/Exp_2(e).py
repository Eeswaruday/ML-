#v.Scatter plot-relationship between two numerical variables
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
iris = load_iris()
# Create DataFrame
df = pd.DataFrame(
iris.data,
columns=iris.feature_names
)
# Add class labels
df['class'] = iris.target
# Scatter plot: Petal Length vs Petal Width
df.plot(
kind='scatter',
x='petal length (cm)',
y='petal width (cm)',
c='class',
colormap='viridis'
)
# Labels and title
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("Scatter Plot of Petal Length vs Petal Width (Iris Dataset)")
# Show plot
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
data = {
'Study_Hours': [1, 2, 3, 4, 5, 6, 7, 8],
'Marks': [35, 40, 50, 55, 65, 70, 78, 85]
}
# Create DataFrame
df = pd.DataFrame(data)
# Scatter plot using pandas
df.plot(
kind='scatter', x='Study_Hours', y='Marks'
)
# Labels and title
plt.xlabel("Study Hours per Day")
plt.ylabel("Marks Obtained")
plt.title(" Study Hours vs Marks")
plt.show()
