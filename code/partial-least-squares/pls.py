import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset available at: https://www.kaggle.com/c/house-prices-advanced-regression-techniques

def get_cat_num_columns(data):
    categorical_columns = list()
    numerical_columns = list()

    for column in data.columns:
        if data[column].dtype == 'int64' or data[column].dtype == 'float64':
            numerical_columns.append(column)
        else:
            categorical_columns.append(column)

    return categorical_columns, numerical_columns

def pre_processing(data):
    data.drop('Id', inplace=True, axis=1)
    categorical_columns, numerical_columns = get_cat_num_columns(data)
    
    # Finding the top 10 columns by the count of missing data (NaN), if more than 85% 
    # of the values are non-nulls, we accept the column. Else delete it.

    # top 10 columns by count of missing data
    total = data.isnull().sum().sort_values(ascending=False)
    percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['num_nulls', 'percentage'])

    # If more than 85% of the values of a particular column should not be null, else drop it
    threshold = len(data) * .85
    data.dropna(thresh = threshold, axis = 1, inplace = True)

    # TotRmsAbvGrd and GrLivArea seem to be highly correlated. This is because the number of rooms 
    # depends directly on the amount of living area which is represented by GrLivArea. So, we can 
    # remove one of these two columns. We will remove TotalRmsAbvGrd.

    data.drop('TotRmsAbvGrd', axis=1, inplace=True)
    # There are two sets of features with very high correlation, namely: ('TotalBsmtSF', '1stFlrSF') and 
    # ('GarageCars','GarageArea'). Usually the basement has the same floor area as the first floor, hence
    # the correlation makes sense. Also, the number of cars that can fit into a garage is proportional to
    # the area of the garage, which makes sense too. We can have one value from each set and drop the other. 
    # We shall drop GarageCars and TotalBsmtSF. 

    # dropping columns
    data.drop(['TotalBsmtSF', 'GarageCars'], axis=1, inplace=True)

    # top 10 columns by count of missing data
    total = data.isnull().sum().sort_values(ascending=False)
    percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['num_nulls', 'percentage'])

    null_cols = missing_data[missing_data['percentage'] > 0].index.values

    categorical_columns.append("MSSubClass")
    numerical_columns.remove("MSSubClass")

    for col in null_cols:
        if col in numerical_columns:
            data[col].fillna(data[col].mean(), inplace = True)
    
    return data

training = pd.read_csv("/Users/manojkarthick/Documents/Fall-17/Machine-Learning/Project/train.csv")
test = pd.read_csv("/Users/manojkarthick/Documents/Fall-17/Machine-Learning/Project/test.csv")

whole_data = training.append(test, ignore_index=True)
whole_data.drop('SalePrice', axis=1, inplace=True)

whole_data = pre_processing(whole_data)
categorical_columns, numerical_columns = get_cat_num_columns(whole_data)

categorical_columns.append("MSSubClass")
numerical_columns.remove("MSSubClass")

whole_data = pd.get_dummies(whole_data, columns=categorical_columns)
# print(categorical_columns)

training_data = whole_data.iloc[:1460,:]
test_data = whole_data.iloc[1460:,:]

# Standardizing the numerical features present in the data for better efficiency of SVM/SVR.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
training_data[numerical_columns] = scaler.fit_transform(training_data[numerical_columns])
test_data[numerical_columns] = scaler.fit_transform(test_data[numerical_columns])

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error

np.random.seed(0)

# Separating the training data into data matrix and target matrix
target = training['SalePrice']
independents = training_data

pls = PLSRegression(n_components=10)
pls.fit(independents, target)

predictions = pls.predict(test_data)

# Writing to file for Kaggle submission

count = 1461
ss = open('pls_submission.csv', 'w')
for prediction in predictions:
    to_write = "{},{}".format(count, prediction[0])
    ss.write(to_write)
    ss.write('\n')
    count += 1


# ****************************************
# The Code below performs cross-validation
# ****************************************

# n = len(independents)

# # 10-fold CV, with shuffle
# kf_10 = KFold(n_splits=10, shuffle=True, random_state=0)

# mse = []

# for i in np.arange(1, 200):
#     print("Running with {} components".format(i))
#     pls = PLSRegression(n_components=i)
#     score = cross_val_score(pls, independents, target, cv=kf_10, scoring='neg_mean_squared_error').mean()
#     mse.append(-score)

# ****************************************
# Plotting
# ****************************************

# # Plot results
# plt.plot(np.arange(1, 200), np.array(mse), '-v')
# plt.xlabel('Number of principal components in regression')
# plt.ylabel('MSE')
# plt.title('Salary')
# plt.xlim(xmin=-1)
# plt.show()