import pandas as pd #we used a dataset i.e housing dataset from pandas
import matplotlib.pyplot as plt #to plot histogram for visualization
from pandas.tools.plotting import scatter_matrix #used for scatter matriz between latitudes and longitudes
from sklearn.linear_model import LinearRegression #linear regression model
import numpy as np
from sklearn.metrics import mean_squared_error #used to calculate mean square error
from sklearn.metrics import mean_absolute_error # used to calculate mean absolute error
from sklearn.ensemble import RandomForestRegressor# another prediction technique similar to linear regression
from sklearn import ensemble
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.cross_validation import train_test_split
sf = pd.read_csv('C:/Users/user/Desktop/PythonC/MachineLear/data.csv') #here we read thecsv file
print sf.describe()#describes the variousa attributes of the datatset

sf.drop(sf.columns[[0, 2, 3, 15, 17, 18]], axis=1, inplace=True) #we don't need some columns hence we drop them from consideration
print sf.info()
print sf.lastsolddate.min(), sf.lastsolddate.max() 
sf['zindexvalue'] = sf['zindexvalue'].str.replace(',', '')
sf['zindexvalue'] = sf['zindexvalue'].convert_objects(convert_numeric=True)#here we convert zindexvalue into numeric
print sf.lastsolddate.min(), sf.lastsolddate.max()

corr_matrix = sf.corr() #here we find he correlation between variables
corr_matrix["lastsoldprice"].sort_values(ascending=False) #relating to lastsoldprice we correlate all the attributes to it


sf.describe()

sf.hist(bins=50, figsize=(20,15)) #we plot a histogram between various attributes
plt.savefig("attribute_histogram_plots")
#plt.show()

sf.plot(kind="scatter", x="longitude", y="latitude", alpha=0.2) #to understand relationship between altitudes and longiudes we use scatter matrix
plt.savefig('map1.png')
#plt.show()

sf.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, figsize=(10,7),
    c="lastsoldprice", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)
#plt.show()

corr_matrix = sf.corr()
dep=corr_matrix["lastsoldprice"].sort_values(ascending=False)#again we find correlation


attributes = ["lastsoldprice", "finishedsqft", "bathrooms", "zindexvalue"]#plotting a scatter matrix between the mentioned attributes
scatter_matrix(sf[attributes], figsize=(12, 8))
plt.savefig('matrix.png')

sf.plot(kind="scatter", x="finishedsqft", y="lastsoldprice", alpha=0.5)
plt.savefig('scatter.png')

#plt.show()

sf['price_per_sqft'] = sf['lastsoldprice']/sf['finishedsqft']
corr_matrix = sf.corr()
corr_matrix["lastsoldprice"].sort_values(ascending=False)

l=len(sf['neighborhood'].value_counts())#here we find the number of neighbourhoods
freq = sf.groupby('neighborhood').count()['address']
#print freq
mean = sf.groupby('neighborhood').mean()['price_per_sqft']
#print mean
cluster = pd.concat([freq, mean], axis=1) #concat means we put the values of freq and mean on the view
#print cluster

cluster['neighborhood'] = cluster.index 
#print cluster['neighborhood']
cluster.columns = ['freq', 'price_per_sqft','neighborhood']
#print cluster.columns
info=cluster.describe()
#print info
cluster1 = cluster[cluster.price_per_sqft < 756]
#print cluster1.index

cluster_temp = cluster[cluster.price_per_sqft >= 756]
cluster2 = cluster_temp[cluster_temp.freq <123]
#print cluster2.index

cluster3 = cluster_temp[cluster_temp.freq >=123]
#print cluster3.index

def get_group(x):
    if x in cluster1.index:
        return 'low_price'
    elif x in cluster2.index:
        return 'high_price_low_freq'
    else:
        return 'high_price_high_freq'
sf['group'] = sf.neighborhood.apply(get_group)

sf.drop(sf.columns[[0, 4, 6, 7, 8, 13]], axis=1, inplace=True)
sf = sf[['bathrooms', 'bedrooms', 'finishedsqft', 'totalrooms', 'usecode', 'yearbuilt','zindexvalue', 'group', 'lastsoldprice']]
#print sf.head()

X = sf[['bathrooms', 'bedrooms', 'finishedsqft', 'totalrooms', 'usecode', 'yearbuilt', 
         'zindexvalue', 'group']]
Y = sf['lastsoldprice']

#print X.head()
print sf.group
print "//////////////////////////////"
n = pd.get_dummies(sf.group)
#print n
X = pd.concat([X, n], axis=1)
#print X
print sf.usecode
m = pd.get_dummies(sf.usecode)
print m
X = pd.concat([X, m], axis=1)
#print X.head()
drops = ['group', 'usecode']
X.drop(drops, inplace=True, axis=1)
#print X.head()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
#print('Linear Regression R squared": %.4f' % regressor.score(X_test, y_test))

lin_mse = mean_squared_error(y_pred, y_test)
lin_rmse = np.sqrt(lin_mse)
#print('Linear Regression RMSE: %.4f' % lin_rmse)

lin_mae = mean_absolute_error(y_pred, y_test)
#print('Linear Regression MAE: %.4f' % lin_mae)

forest_reg = RandomForestRegressor(random_state=42)
forest_reg.fit(X_train, y_train)
#print('Random Forest R squared": %.4f' % forest_reg.score(X_test, y_test))

y_pred = forest_reg.predict(X_test)
forest_mse = mean_squared_error(y_pred, y_test)
forest_rmse = np.sqrt(forest_mse)
#print('Random Forest RMSE: %.4f' % forest_rmse)
model = ensemble.GradientBoostingRegressor()
model.fit(X_train, y_train)

#print('Gradient Boosting R squared": %.4f' % model.score(X_test, y_test))


