#from sklearn.matrics import mean_absolute_error, mean_square_error, accurate_score, precision_score,f1_score
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score

# Actual and predicted values
y_true = [100, 200, 300,400,578,150]
y_pred = [110, 180, 290,213,789, 458]
x_true = [500,780,250,489]
x_pred = [200,781,350,689]

# Calculate metrics
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)
actual = accuracy_score(y_true, y_pred)
f1 = f1_score(x_true, x_pred, average='macro')


print("This is the test of Regeneration and Classification type test \n ")
print("MAE = ", mae)
print("MSE = ", mse)
print("RMSE = ", rmse)
print("R² Score = ", r2)

# classification
print("Accuracy = " , actual)
print("f1_Score = ", f1)

