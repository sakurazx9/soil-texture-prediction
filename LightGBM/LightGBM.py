import rasterio
import geopandas as gpd
import pandas as pd
import numpy as np
from rasterio.transform import from_origin
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import lightgbm as lgb 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from scipy.stats import kruskal


points = pd.read_csv(r'C:\CSV file path\soil_texture.csv')


X = points[[ 'DEM','Slope','PX','PMQL', 'kTVDI', 'NDVI','Band7','TWI','SPI','BSI','SAVI','NDSI','SCI','SI','Clay','Fe']]
y = points['SLHL']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

kruskal_stat, kruskal_p = kruskal(y_test, y_pred)

print(f"Kruskal-Wallis Statistic: {kruskal_stat}")
print(f"P-value: {kruskal_p}")

if kruskal_p < 0.05:
    print("The difference between predicted and true values is statistically significant (p < 0.05).")
else:
    print("No statistically significant difference found between predicted and true values (p >= 0.05).")

mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

def lin_concordance_correlation_coefficient(y_true, y_pred):
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))

    numerator = 2 * covariance
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

    return numerator / denominator

ccc = lin_concordance_correlation_coefficient(y_test, y_pred)

def calculate_rpiq(y_true, rmse):
    q75, q25 = np.percentile(y_true, [75, 25]) 
    iqr = q75 - q25 
    return iqr / rmse

rpiq = calculate_rpiq(y_test, rmse)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R² Score: {r2}")
print(f"Lin's Concordance Correlation Coefficient (CCC): {ccc}")
print(f"RPIQ: {rpiq}")

