#https://towardsdatascience.com/simple-and-multiple-linear-regression-in-python-c928425168f9
#https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6
# https://machinelearningmastery.com/train-final-machine-learning-model/   -- Great article on this stuff :}
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import seaborn as sns
import mpl_toolkits
from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model  ## imports datasets from scikit-learn
# from sklearn.cross_validation import train_test_split   #Deprecated
from sklearn.model_selection import train_test_split
# import Statsmodels
# Necessary imports:
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics

from sklearn import ensemble     #Model booster !! Boost accuracy of model
from sklearn.ensemble import GradientBoostingRegressor


# data = pd.read_csv('dataFiles/kc_house_data.csv')
data = datasets.load_boston()    ## loads Boston dataset from datasets library

#print (data.head())
#print (data.describe())     ///Statistical data


print ("starting...")


# Load the Diabetes Housing dataset
columns = 'age sex bmi map tc ldl hdl tch ltg glu'.split()  # Declare the columns names
diabetes = datasets.load_diabetes()                         # Call the diabetes dataset from sklearn
df = pd.DataFrame(diabetes.data, columns=columns)           # load the dataset as a pandas data frame
y = diabetes.target                                         # define the target variable (dependent variable) as y

# create training and testing vars
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
#print (X_train.shape, y_train.shape)
#print (X_test.shape, y_test.shape)

# fit a model
lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)

print (predictions[0:5])
#print (np.array([205.68012533, 64.58785513, 175.12880278, 169.95993301,128.92035866]))

# Accuracy score
print ('Score:', model.score(X_test, y_test))


## The line / model
plt.scatter(y_test, predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()

#########################################################################################################
# define the data/predictors as the pre-set feature names
# df = pd.DataFrame(data.data, columns=data.feature_names)
#
# # Put the target (housing value -- MEDV) in another DataFrame
# target = pd.DataFrame(data.target, columns=['MEDV'])
#
# X = df
# y = target['MEDV']
#
#
# lm = linear_model.LinearRegression()
#
# model = lm.fit(X,y)
#
# predictions = lm.predict(X)
# print(predictions[0:5])
########################################################################################################

# data['bedrooms'].value_counts().plot(kind='bar')
# plt.title('number of Bedroom')
# plt.xlabel('Bedrooms')
# plt.ylabel('Count')


# plt.figure(figsize=(10,10))
# sns.jointplot(x=data.lat.values, y=data.long.values, size=10)
# plt.ylabel('Longitude', fontsize=12)
# plt.xlabel('Latitude', fontsize=12)


###################### Works ok ###################
# plt.scatter(data.price,data.sqft_living)
# plt.xlabel("Price")
# plt.ylabel('Square feet')
# plt.title("Price vs Square Feet")
###################### Works ok ###################




# plt.scatter(data.price,data.sqft_living)
# plt.title("Price vs Square Feet")

# plt.show()
# plt1 = plt()
# sns.despine

###################### Works ok ###################
# plt.scatter(data.price,data.lat)
# plt.xlabel("Price")
# plt.ylabel('Latitude')
# plt.title("Latitude vs Price")
###################### Works ok ###################

###################### Works ok ###################
# plt.scatter(data.bedrooms,data.price)
# plt.title("Bedroom and Price ")
# plt.xlabel("Bedrooms")
# plt.ylabel("Price")
# plt.show()
# sns.despine
###################### Works ok ###################

# data.floors.value_counts().plot(kind='bar')
#plt.scatter(data.floors,data.price)


# reg = LinearRegression()
# labels = data['price']
# conv_dates = [1 if values == 2014 else 0 for values in data.date ]
# data['date'] = conv_dates
# train1 = data.drop(['id', 'price'],axis=1)
#
# x_train , x_test , y_train , y_test = train_test_split(train1 , labels , test_size = 0.10,random_state =2)
#
# reg.fit(x_train,y_train)
#
# print ("Raw Accuracy", reg.score(x_test,y_test))      #Print model accuracy
#
# #model accuracy booster
# clf = ensemble.GradientBoostingRegressor(n_estimators = 400, max_depth = 5, min_samples_split = 2,
#           learning_rate = 0.08, loss = 'ls')
# clf.fit(x_train, y_train)

#
# plt.scatter(x_test, x_test)
# plt.title("Id and Price ")
# plt.xlabel("Id")
# plt.ylabel("Price")
# plt.show()


# print ("Boosted Accuracy", clf.score(x_test,y_test))

# est = GradientBoostingRegressor(n_estimators=2000, max_depth=1) .fit(x_train,y_train)
# plt.plot(x_train[:,0], pred=0, color='r',alpha=0.1)
#plt.show()


print ("finished...")
