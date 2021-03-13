# -*- coding: utf-8 -*-

from pandas import read_csv
from pandas import to_datetime
#from pandas import DataFrame
from fbprophet import Prophet
from matplotlib import pyplot

# load data
path = 'Dataset/deaths_pivot.csv'
df = read_csv(path, header=0)
# prepare expected column names and necessary columns
df.columns = ['ülke','ds', 'y']
df['ds']= to_datetime(df['ds'])
del df['ülke']
#split the dataset into train and test sets
X_train = df[0:300]
Y_test = df[300:-1]
# define the model
model = Prophet()
# fit the model
model.fit(X_train)
# use the model to make a forecast
forecast = model.predict(Y_test)
# expected vs actual
y_true = Y_test['y'].values
y_pred = forecast['yhat'].values
# showing plot
pyplot.plot(y_true, label='Actual')
pyplot.plot(y_pred, label='Predicted')
pyplot.legend()
pyplot.show()
