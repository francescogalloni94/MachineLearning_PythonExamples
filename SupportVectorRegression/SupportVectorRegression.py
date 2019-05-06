import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2:3].values

from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X,y)

#Have to perform feature scaling because svr it's not automatically doing it
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_scaled = sc_X.fit_transform(X)
y_scaled = sc_y.fit_transform(y)


regressor_scaled = SVR(kernel='rbf')
regressor_scaled.fit(X_scaled,y_scaled)

plt.subplot(2,1,1)
plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title('Support Vector Regression')
plt.xlabel('Position Level')
plt.xlabel('Salary')


plt.subplot(2,1,2)
plt.scatter(X_scaled,y_scaled,color='red')
plt.plot(X_scaled,regressor_scaled.predict(X_scaled),color='blue')
plt.title('Support Vector Regression (Features Scaling)')
plt.xlabel('Position Level')
plt.xlabel('Salary')
plt.tight_layout()
plt.show()
