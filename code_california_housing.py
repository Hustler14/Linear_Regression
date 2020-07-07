#reading data from file
import pandas as pd
data = pd.read_csv("california_housing_train.csv")
data.head()

data.isnull().any()
#data = data.fillna(method = 'ffil')

#split data
x = data[['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']].values
y = data['median_house_value'].values

#fitting to linear regression
from sklearn.linear_model import LinearRegression
#x = x.reshape(-1,1)
#y = y.reshape(-1,1)
reg = LinearRegression()
reg.fit(x,y)
pred = reg.predict(x)

#testing data 
test = pd.read_csv("california_housing_test.csv")
x_test = test[['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']].values
y_test = test['median_house_value'].values
pred_value = reg.predict(x_test)
#print(pred_value[0], y_test[0])
l = np.square((pred_value - y_test)).mean()
print(l)
