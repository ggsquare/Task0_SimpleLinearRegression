from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

trainfilename = '/Users/Anna/polybox/IntroductionML/Tasks/Dummy/task0_sl19d9/train.csv'
testfilename = '/Users/Anna/polybox/IntroductionML/Tasks/Dummy/task0_sl19d9/test.csv'

trainfile = pd.read_csv(trainfilename, delimiter = ',')
testfile = pd.read_csv(testfilename, delimiter = ',')

Y_train = trainfile['y']
X_test = testfile._drop_axis(['Id'], axis=1)
X_train = trainfile._drop_axis(['Id','y'], axis=1) #axis=0: column-wise, axis=1: row-wise

regr = linear_model.LinearRegression()
regr.fit(X_train, Y_train)
Y_pred=regr.predict(X_test)

y=np.mean(X_test, axis=1)
RMSE = mean_squared_error(y, Y_pred)**0.5

# The coefficients
print('Coefficients: \n', regr.coef_)

# The mean squared error
print("Mean squared error: %.10f"% RMSE) #.7 nachkommastellen

# output results
d={'Id': testfile['Id'], 'y': Y_pred}
output=pd.DataFrame(d)
output.to_csv('output.csv', index=False)
