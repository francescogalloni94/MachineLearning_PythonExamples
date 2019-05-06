import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X,y)

from sklearn.preprocessing import PolynomialFeatures
polinomial_regressor_2 = PolynomialFeatures(degree=2)
X_polinomial_2 = polinomial_regressor_2.fit_transform(X)
linear_regressor_2 = LinearRegression()
linear_regressor_2.fit(X_polinomial_2,y)

polinomial_regressor_4 = PolynomialFeatures(degree=4)
X_polinomial_4 = polinomial_regressor_4.fit_transform(X)
linear_regressor_4 = LinearRegression()
linear_regressor_4.fit(X_polinomial_4,y)

plt.subplot(3, 1, 1)
plt.scatter(X,y,color='red')
plt.plot(X,linear_regressor.predict(X),color='blue')
plt.title('Linear Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')

plt.subplot(3, 1, 2)
plt.scatter(X,y,color='red')
plt.plot(X,linear_regressor_2.predict(polinomial_regressor_2.fit_transform(X)),color='blue')
plt.title('Polinomial Regression (Grade 2)')
plt.xlabel('Position Level')
plt.ylabel('Salary')

plt.subplot(3, 1, 3)
plt.scatter(X,y,color='red')
plt.plot(X,linear_regressor_4.predict(polinomial_regressor_4.fit_transform(X)),color='blue')
plt.title('Polinomial Regression (Grade 4)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.tight_layout()
plt.show()



