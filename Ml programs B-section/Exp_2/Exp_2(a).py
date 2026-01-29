#i.Bar plots- Simple Example
import pandas as pd
import matplotlib.pyplot as plt
# Create simple numerical data
data = {
 'Maths': 85,
 'Physics': 90,
 'Chemistry': 78,
 'Biology': 88
}
# Create Pandas Series
marks = pd.Series(data, name="Marks")
# Plot bar graph using pandas
marks.plot(kind='bar', legend=True)
# Add labels and title
plt.title("Marks in Different Subjects")
plt.xlabel("Subjects")
plt.ylabel("Marks")
# Show plot
plt.show()