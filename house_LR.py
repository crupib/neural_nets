import sklearn
from sklearn import linear_model
from sklearn.datasets import fetch_california_housing

cal_house = fetch_california_housing()
cal_house_X_train = cal_house.data[:-20]
cal_house_X_test = cal_house.data[-20:]
cal_house_y_train = cal_house.target[:-20]
cal_house_y_test = cal_house.target[-20:]
regr = linear_model.LinearRegression()
regr.fit(cal_house_X_train, cal_house_y_train)
predictions = regr.predict(cal_house_X_test)
print('MSE: {:.2f}'.format(sklearn.metrics.mean_squared_error(cal_house_y_test,predictions)))
