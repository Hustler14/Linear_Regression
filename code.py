#reading data from file
import pandas as pd
data = pd.read_excel("slr06.xls")
data.head()

#plot a scatter of data points
import matplotlib
from matplotlib import pyplot as plt
plt.scatter(data['X'], data['Y'], c = 'r')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

#mean normalization and feature scaling
import numpy as np
x  = data['X'].to_numpy()
x_m = x.mean()
x_s = x.std()
y = data['Y'].to_numpy()
y_m = y.mean()
y_s = y.std()
x = (x - x_m)/x_s
y = (y - y_m)/y_s

#plot mean normalized and feature scaled data
plt.scatter(x, y, c = 'r')
plt.xlabel("x")
plt.ylabel("y")
plt.show()

#fitting to linear regression
from sklearn.linear_model import LinearRegression
x = x.reshape(-1,1)
y = y.reshape(-1,1)
reg = LinearRegression()
reg.fit(x,y)
pred = reg.predict(x)

#plotting linear regression
plt.scatter(x, y, c = 'r')
plt.plot(x, pred, c = "blue", linewidth = 2)
plt.xlabel("x")
plt.ylabel("y")
plt.show()

#testing data 
x = 108
x = (x - x_m) / x_s
t = np.asarray([x])
m = reg.predict(t.reshape(1, -1))
m = m*y_s + y_m
print(m)



