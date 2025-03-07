import pandas as pd
import pandas as ps
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv("student-mat.csv", sep=";") # sep = separator-instead of "," a ";" is used to separate attributes

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]] # G3= final grade

predict = "G3"

X = np.array(data.drop([predict], axis=1))
Y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

linear = linear_model.LinearRegression()

linear.fit (x_train, y_train)
accuracy = linear.score(x_test, y_test)
print(accuracy)

print("Co: ", linear.coef_)
print("intercept", linear.intercept_)

predictions = linear.predict(x_test)

for x in range (len (predictions)):
    print(predictions[x], x_test[x], y_test[x])