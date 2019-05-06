import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10,random_state=0)
regressor.fit(X,y)
regressor_100 = RandomForestRegressor(n_estimators=100,random_state=0)
regressor_100.fit(X,y)
regressor_300 = RandomForestRegressor(n_estimators=300,random_state=0)
regressor_300.fit(X,y)

X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue',label='10 predictor trees')
plt.plot(X_grid,regressor_100.predict(X_grid),color='green',label='100 predictor trees')
plt.plot(X_grid,regressor_300.predict(X_grid),color='black',label='300 predictor trees')
plt.title('Random Forest Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.legend()
plt.show()