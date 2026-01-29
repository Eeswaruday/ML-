import pandas as pd
import matplotlib.pyplot as plt

# Creating a sample DataFrame
data = {
    'Year': [2018, 2019, 2020, 2021, 2022],
    'Sales': [250, 300, 280, 350, 400],
    'Profit': [50, 70, 65, 90, 120]
}

df = pd.DataFrame(data)

# Correct print statement
print(df)
