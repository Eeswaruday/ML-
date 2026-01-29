#a) Write a python program to implement Pandas Series with labels.
import pandas as pd
# Creating a Pandas Series with labels
data = [85, 90, 88, 92]
labels = ['Maths', 'Physics', 'Chemistry', 'Biology']
series = pd.Series(data, index=labels)
# Display the Series
print("Pandas Series with Labels:")
print(series)