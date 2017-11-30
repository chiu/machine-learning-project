import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import heapq,operator
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from scipy.stats.stats import pearsonr 

#Standardizes a dataframe
def standardize(df):
    means = df.apply(lambda x:x.mean())
    sd = (df.apply(lambda x:x.var()))**0.5
    df = (df-means)/sd # Standardizing non-boolean attributes
    return df

#Splits a dataframe into two, one containing the numerical columns
#and the other the categorical ones
def splitNumCat(df):
    numericals = df.select_dtypes(include=[np.number])
    categoricals = df.select_dtypes(exclude=[np.number])
    return numericals,categoricals

df1 = pd.read_csv('train.csv')
df2 = pd.read_csv('test.csv')

target = df1['SalePrice'] #dropping the target variable
df1 = df1.drop(['Id','SalePrice'],axis=1);
testIDs = df2['Id']
df2 = df2.drop(['Id'],axis=1)

######### Preprocessing #############
numericals_train,categoricals_train = splitNumCat(df1)
numericals_test,categoricals_test = splitNumCat(df2)

numericals_train = numericals_train.fillna(numericals_train.mean())
numericals_test = numericals_test.fillna(numericals_test.mean())
numericals_train = standardize(numericals_train)
numericals_test = standardize(numericals_test)

categoricals = (pd.concat([categoricals_train,categoricals_test]))
categoricals = pd.get_dummies(categoricals)
lenCatTrain = categoricals_train.shape[0]
categoricals_train = categoricals.iloc[0:lenCatTrain]
categoricals_test = categoricals.iloc[lenCatTrain:]

preprocessedDataTrain = pd.concat([numericals_train,categoricals_train],axis=1)
preprocessedDataTest = pd.concat([numericals_test,categoricals_test],axis=1)

######### End of Preprocessing #############


X_train = preprocessedDataTrain.values
X_test = preprocessedDataTest.values
y_train = target.values

#Creating an instance of random forest regressor
regr = RandomForestRegressor(max_features=0.5,max_depth=20,n_estimators=400, random_state=0 )
regr.fit(X_train, y_train)
y_test = regr.predict(X_test)
predictedPrices = pd.DataFrame(y_test,columns=['SalePrice'])
df3 = pd.concat([testIDs,predictedPrices],axis=1)
df3.to_csv('submission.csv')

#plotting the most informative features
a = (regr.feature_importances_)
ind = heapq.nlargest(10, range(len(a)), a.take)
x_ticks = np.array([])
y=np.array([])
for i in range(len(ind)):
    feature = preprocessedDataTrain.columns.values[ind[i]]
    value = a[ind[i]]
    x_ticks = np.append(x_ticks,[feature])
    y = np.append(y,[value])

x=np.linspace(1,10,10)
plt.xticks(x, x_ticks, rotation='vertical')
plt.bar(x, y)
plt.ylabel('Mutual Information')
plt.xlabel('Feature')
plt.show()

#Plotting the most correlated features
dict1={}
for i in range(X_train.shape[1]):
    corr = pearsonr(X_train[:,i],y_train)[0]
    label = preprocessedDataTrain.columns.values[i]
    dict1[label]=corr
sorted_x = sorted(dict1.items(), key=operator.itemgetter(1),reverse=True)
num_feat = 15
vec = sorted_x[0:num_feat]
corr = np.array([])
feat = np.array([])
for i in range(len(vec)):
    feat = np.append(feat,[vec[i][0]])
    corr = np.append(corr,[vec[i][1]])
x=np.linspace(1,num_feat,num_feat)
plt.xticks(x, feat, rotation='vertical')
plt.bar(x, corr,color='indigo')
plt.ylabel('Correlation')
plt.xlabel('Feature')
plt.show()
