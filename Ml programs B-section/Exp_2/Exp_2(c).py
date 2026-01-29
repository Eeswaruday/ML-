#iii.Histograms -to understand how each numerical feature is distributed.
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
# Plot histogram for petal length
df['petal length (cm)'].plot(
 kind='hist',
 bins=15,
 legend=True
)
# Labels and title
plt.xlabel("Petal Length (cm)")
plt.ylabel("Frequency")
plt.title("Histogram of Petal Length in Iris Dataset")
# Show plot
plt.show()