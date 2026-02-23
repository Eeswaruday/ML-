import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# 1. Load the dataset
# -------------------------------
data = pd.read_csv("Bengaluru_House_Data.csv")

# -------------------------------
# 2. Basic cleaning
# -------------------------------
# Drop unnecessary columns (ignore if they don't exist)
data.drop(columns=['area_type', 'availability', 'society', 'balcony'], inplace=True, errors='ignore')

# Fill missing values
data['location'] = data['location'].fillna('Sarjapur Road')
data['size'] = data['size'].fillna('2 BHK')
data['bath'] = data['bath'].fillna(data['bath'].median())

# Convert BHK to integer
data['bhk'] = data['size'].str.split().str.get(0).astype(int)

# -------------------------------
# 3. Handle total_sqft column
# -------------------------------
def convertRange(x):
    import re
    try:
        if pd.isnull(x):
            return None
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if '-' in s:
            parts = s.split('-')
            return (float(parts[0].strip()) + float(parts[1].strip())) / 2
        s = s.replace(',', '')
        m = re.match(r"^\d+\.?\d*", s)
        if m:
            return float(m.group())
        return None
    except Exception:
        return None

data['total_sqft'] = data['total_sqft'].apply(convertRange)
# Remove rows with invalid or zero total_sqft
data = data[~data['total_sqft'].isnull()]
data = data[data['total_sqft'] > 0]

# -------------------------------
# 4. Price per sqft
# -------------------------------
data['price_per_sqft'] = data['price'] * 1000000 / data['total_sqft']

# Clean location names
data['location'] = data['location'].apply(lambda x: x.strip())

# Group rare locations as "other"
location_count = data['location'].value_counts()
location_count_less_10 = location_count[location_count <= 10]
location_small = set(location_count_less_10.index)
data['location'] = data['location'].apply(lambda x: 'other' if x in location_small else x)

# -------------------------------
# 5. Remove outliers
# -------------------------------
# Outlier: total_sqft per BHK
data = data[(data['total_sqft'] / data['bhk']) >= 300]

# Outlier: price per sqft within each location
def remove_outliers_sqft(df):
    df_output = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        vals = subdf.price_per_sqft.dropna()
        if vals.empty:
            continue
        m = np.mean(vals)
        st = np.std(vals)
        gen_props = subdf[(subdf.price_per_sqft > (m - st)) & (subdf.price_per_sqft <= (m + st))]
        df_output = pd.concat([df_output, gen_props], ignore_index=True)
    return df_output

data = remove_outliers_sqft(data)

# Outlier: BHK price comparison
def bhk_outlier_remover(df):
    exclude_indices = []
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk - 1)
            if stats and stats['count'] > 5:
                bad_idx = bhk_df[bhk_df.price_per_sqft < stats['mean']].index.tolist()
                exclude_indices.extend(bad_idx)
    if exclude_indices:
        exclude_indices = list(set(exclude_indices))
        return df.drop(labels=exclude_indices, axis='index')
    return df

data = bhk_outlier_remover(data)

# Drop unnecessary columns
data.drop(columns=['size', 'price_per_sqft'], inplace=True)

# -------------------------------
# 6. Feature selection
# -------------------------------
features = ['total_sqft', 'bath', 'bhk']
X = data[features]
y = data['price']

# -------------------------------
# 7. Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# -------------------------------
# 8. Feature scaling
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# 9. Train KNN Regressor
# -------------------------------
knn_regressor = KNeighborsRegressor(n_neighbors=5)
knn_regressor.fit(X_train_scaled, y_train)

# -------------------------------
# 10. Predictions & Evaluation
# -------------------------------
y_pred = knn_regressor.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")
