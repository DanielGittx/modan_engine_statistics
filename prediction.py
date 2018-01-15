import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mpl_toolkits
from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model  ## imports datasets from scikit-learn
from sklearn.model_selection import train_test_split
from sklearn import ensemble


data = pd.read_csv('dataFiles/kc_house_data.csv')

# print (data.head())           #Print statistical information - count, mean, median, MAX, MIN etc
# print (data.describe())

###################################################
data['bedrooms'].value_counts().plot(kind='bar')
plt.title('Number of bedrooms')
plt.xlabel('Bedrooms')
plt.ylabel('Count')
plt.show()
sns.despine()


###################################################
plt.scatter(data.price,data.sqft_living)
plt.title('Price Vs Square Feet')
plt.show()
sns.despine()

###################################################

reg = LinearRegression()
###################################################

labels = data['price']
conv_dates = [1 if values == 2014 else 0 for values in data.date]
data['date'] = conv_dates
train1 = data.drop(['id', 'price'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(train1, labels, test_size=0.10, random_state=2)
reg.fit(x_train,y_train)
LinearRegression (copy_X=True, fit_intercept=True,n_jobs=1,normalize=False)
accuracy_score = reg.score(x_test, y_test)

# print ('Ratio:train:test',x_train.shape, x_test.shape)
print ("Raw Score",accuracy_score)

###################################################

gradboosted = ensemble.GradientBoostingRegressor(n_estimators=400, max_depth=5, min_samples_split=2,learning_rate=0.08, loss='ls')
gradboosted.fit(x_train,y_train)
print (gradboosted.predict(x_test))

# boosted_score = gradboosted.score(x_test,y_test)
# print("Boosted Score",boosted_score)





###################################################

