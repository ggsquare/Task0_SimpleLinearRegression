import csv
import numpy as np
from sklearn import linear_model
import pandas as pd

Id_train = pd.read_csv('train.csv',usecols=[0])
y_train = pd.read_csv('train.csv',usecols=[1])
X_train = pd.read_csv('train.csv',usecols=[2, 3, 4, 5, 6, 7, 8, 9, 10])

Id_test = pd.read_csv('test.csv',usecols=[0])
X_test = pd.read_csv('test.csv',usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9])

print(Id_test)

# Create linear regression object
linReg = linear_model.LinearRegression()

# Train the model using the training sets
linReg.fit(X_train, y_train)

# Make predictions using the testing set
yPred = linReg.predict(X_test)
#
print(yPred)

Output=zip(Id_test,yPred)
Output=np.concatenate( [ np.array( Id_test ), np.array( yPred ) ] , axis = 1)
with open('predict.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for row in Output:
        writer.writerow(row)