#iv.Line plot
import pandas as pd
import matplotlib.pyplot as plt
#data
data = {
'Day': [1, 2, 3, 4, 5, 6],
'Temperature': [22, 24, 23, 26, 28, 30]
}
# Create DataFrame
df = pd.DataFrame(data)
# Plot line graph using pandas
df.plot(x='Day', y='Temperature', kind='line', legend=True)
# Add labels and title
plt.xlabel("Day")
plt.ylabel("Temperature (°C)")
plt.title("Line Plot of Temperature Over Days")
# Show plot
plt.show()