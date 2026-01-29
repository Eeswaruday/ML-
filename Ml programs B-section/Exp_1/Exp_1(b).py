#b) Create a Pandas Series from a dictionary.
import pandas as pd
# Dictionary data
data = {'Maths': 85, 'Physics': 90, 'Chemistry': 88}
# Creating Series from dictionary
s = pd.Series(data)
print("Pandas Series from dictionary:")
print(s)
import pandas as pd
# Creating dictionary using dict keyword
data = dict(Maths=85, Physics=90, Chemistry=88)
# Creating Pandas Series from dictionary
s = pd.Series(data)
print("Pandas Series from dictionary:")
print(s)