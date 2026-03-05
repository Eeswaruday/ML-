import numpy as np
import pandas as pd

# Load dataset (make sure play_tennis.csv is in the same folder)
df = pd.read_csv(r"C:\Users\Student\Desktop\Ml programs B-section\EXP_12\play_tennis.csv")

# Display dataset
print("Dataset:")
print(df)

# Count values of 'play'
print("\nPlay counts:")
print(df['play'].value_counts())

# Prior probabilities
Pyes = 9/14
Pno = 5/14
print("\nPrior Probabilities:")
print("P(Yes) =", Pyes)
print("P(No)  =", Pno)

# Outlook vs Play
crosstab_outlook = pd.crosstab(df['outlook'], df['play'])
print("\nOutlook vs Play:")
print(crosstab_outlook)

Pon = 0
Prn = 2/5
Psn = 3/5

Poy = 4/9
Pry = 3/9
Psy = 2/9

# Temp vs Play
crosstab_temp = pd.crosstab(df['temp'], df['play'])
print("\nTemp vs Play:")
print(crosstab_temp)

Pcn = 1/5
Phn = 2/5
Pmn = 2/5

Pcy = 3/9
Phy = 2/9
Pmy = 4/9

# Humidity vs Play
crosstab_humidity = pd.crosstab(df['humidity'], df['play'])
print("\nHumidity vs Play:")
print(crosstab_humidity)

Pnh = 4/5
Pnn = 1/5

Pyh = 3/9
Pyn = 6/9

# Wind vs Play
crosstab_wind = pd.crosstab(df['wind'], df['play'])
print("\nWind vs Play:")
print(crosstab_wind)

Pns = 3/5
Pnw = 2/5
Pys = 3/9
Pyw = 6/9

# Problem: outlook=Sunny, Temp=Hot, Humidity=High, Wind=Weak
print("\nPrediction for outlook=Sunny, temp=Hot, humidity=High, wind=Weak:")

PY = Psy * Phy * Pyh * Pyw * Pyes
PN = Psn * Phn * Pnh * Pnw * Pno

print("P(Yes) =", PY)
print("P(No)  =", PN)

if PY > PN:
    print("Answer: Play Tennis (Yes)")
else:
    print("Answer: Do Not Play Tennis (No)")
