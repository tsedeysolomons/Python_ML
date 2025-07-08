#from sklearn.matrics import mean_absolute_error, mean_square_error, accurate_score, precision_score,f1_score
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Actual and predicted values
y_true = [100, 200, 300]
y_pred = [110, 180, 290]

# Calculate metrics
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)

print("MAE = ", mae)
print("MSE = ", mse)
print("RMSE = ", rmse)
print("R² Score = ", r2)