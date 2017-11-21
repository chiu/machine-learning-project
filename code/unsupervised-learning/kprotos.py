import sys
import operator
import re, string
import csv
import math
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes 
from sklearn.decomposition import PCA
import pandas as pd
import sqlite3
import matplotlib.pylab as plt

# Paper on k-modes and k-prototypes
# http://www.cs.ust.hk/~qyang/Teaching/537/Papers/huang98extensions.pdf

numRows = 1459 
numCols = 79

allFeatures = ['MSSubClass','MSZoning','LotFrontage','LotArea','Street','Alley','LotShape','LandContour','Utilities',
'LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','OverallQual','OverallCond',
'YearBuilt','YearRemodAdd','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','MasVnrArea','ExterQual',
'ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2',
'BsmtUnfSF','TotalBsmtSF','Heating','HeatingQC','CentralAir','Electrical','1stFlrSF','2ndFlrSF','LowQualFinSF',
'GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','KitchenQual','TotRmsAbvGrd',
'Functional','Fireplaces','FireplaceQu','GarageType','GarageYrBlt','GarageFinish','GarageCars','GarageArea','GarageQual',
'GarageCond','PavedDrive','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','PoolQC','Fence',
'MiscFeature','MiscVal','MoSold','YrSold','SaleType','SaleCondition']

numericalFeatures = ['Id','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MasVnrArea',
'BsmtFinSF1','MiscVal','GarageCars', 'GarageArea','Fireplaces','TotRmsAbvGrd','1stFlrSF','2ndFlrSF','LowQualFinSF',
'GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','LotFrontage','LotArea']

categoricalFeatures = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 
'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 
'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 
'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 
'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 
'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition']


# Check which columns of df have a percentage of null values (nan) higher than p
def checkNulls(df,p) :
	check_null = df.isnull().sum(axis=0).sort_values(ascending=False)/float(len(df))
	print(check_null[check_null>p])

# Fill nans with 0s for numerical columns and with 'NA' for categorical ones 
# also, drop the Id column
def processData(df) :
	# dfWithDummies = pd.get_dummies(df)
	classLabels = df['SalePrice']
	df.drop(['SalePrice'],axis=1,inplace=True)
	df.drop(['Id'],axis=1,inplace=True)
	# All numerical features with 0
	df['LotFrontage'].fillna(0,inplace=True)
	df['MasVnrArea'].fillna(0,inplace=True)
	df['GarageYrBlt'].fillna(0,inplace=True)
	# All categorical ones with NA
	df = df.fillna('NA')

	# Code in case we wanted them as integers,but doesn't make a difference
	# for feature in categoricalFeatures :
	# 	df[feature] = df[feature].astype('category')
	
	# cat_columns = df.select_dtypes(['category']).columns
	# df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

	return df,classLabels

# Discretize the prices in i categories
def processPrices(y,binNum) :
	histo = np.histogram(y,binNum)
	bins = histo[1]

	print("------------------ Real histogram division ------------------")
	print(histo[0])

	newPrices = []
	for i in range(len(y)) :
		for j in range(len(bins)-1) :
			if ((y[i] >= bins[j]) and (y[i] <= bins[j+1])) :
				newPrices.append(j)

	return newPrices

def dropCols(df,cols) :
	for col in cols :
		df.drop([col],axis=1,inplace=True)

# Check the score from chi2 and f-regression tests for each column
def checkScores(trainData,yTrain,tX2,tF) :
	# Tests with chi2 (Again, this is for classification so it's probably pointless)
	selector = SelectKBest(chi2, k='all').fit(trainData,yTrain)
	x_new = selector.transform(trainData) # not needed to get the score
	scores = selector.scores_

	print("Tests with X2 : ")
	for t in tX2 :
		count = 0
		for i in range(len(scores)) :
			if (scores[i] > t) :
				# Uncomment to print each feature and score
				# print(colNames[i] + " "+ str(scores[i]))
				count += 1
		print("Number of features above threshold "+str(t)+" : "+str(count))

	# Tests with f_regression :
	selector = SelectKBest(f_regression, k='all').fit(trainData,yTrain)
	x_new = selector.transform(trainData) # not needed to get the score
	scores = selector.scores_

	print()

	print("Tests with f_regression : (lower scores and thresholds)")
	for t in tF :
		count = 0
		for i in range(len(scores)) :
			if (scores[i] > t) :
				# Uncomment to print each feature and score
				# print(colNames[i] + " "+ str(scores[i]))
				count += 1
		print("Number of features above threshold "+str(t)+" : "+str(count))

# Get columns that are categorical
def getCatVars(data) :
	catVars = []
	for i in range(len(allFeatures)) :
		if allFeatures[i] in categoricalFeatures :
			catVars.append(i)

	return catVars

# Print a table showing the number of each class of elements existing in each cluster
def printTable(model,y) :
	print("------------------ Price category Vs cluster placement table ------------------")
	classtable = np.zeros((5, 5), dtype=int)
	for ii, _ in enumerate(y):
		classtable[int(y[ii]) , model.labels_[ii]] += 1 

	print("\n")
	print("    | Cl. 1 | Cl. 2 | Cl. 3 | Cl. 4 |Cl. 5  |")
	print("----|-------|-------|-------|-------|-------|")
	for ii in range(5):
		prargs = tuple([ii + 1] + list(classtable[ii, :]))
		print(" P{0} |    {1:>2} |    {2:>2} |    {3:>2} |    {4:>2} |    {5:>2} |".format(*prargs))

def printCentroidInfo(centroids,featuresNum,featuresCat) :
	print("------------------ Centroid Information ------------------")
	# Produces centroid information for both numerical and
	# categorical variables 
	centroidsNum = centroids[0]
	centroidsCat = centroids[1]

	# Obtain the features with different values in at least one cluster
	diffFeatures = []
	for i in range(len(featuresCat)) :
		equal = True
		j = 0
		while (j < len(centroidsCat)-1 and equal) :
			if (centroidsCat[j,i] != centroidsCat[j+1,i]) :
				diffFeatures.append(i)
				break
			j += 1

	# Print all features that affect the clusters and the values associated with them
	for f in diffFeatures :
		print("Feature : "+(featuresCat[f])) 
		for j in range(len(centroidsCat)) :
			print("Centroid "+str(j)+" : "+centroidsCat[j,f])

	print("Features not shown here have the same value for every cluster")
			


# Main program	    		
if __name__ == '__main__':
	# Read data
	fName = sys.argv[1]
	numClusters = int(sys.argv[2])
	df = pd.read_csv(fName)

	# Check columns with 50% or more nulls (nans)
	# checkNulls(df,0.5)

	# Drop columns wit high % of nulls
	# dropCols(df,['PoolQC','MiscFeature','Alley','Fence'])

	# Process the data (Fill nans)
	df,ydf = processData(df)

	# Get column names
	colNames = list(df)

	# Transform to numpy representation
	trainData = df.as_matrix()
	y = ydf.as_matrix()

	# Generate prices separated in numClusters bins
	priceCategories = processPrices(y,numClusters)

	# Obtain the number of each categorical column
	numCatVars = getCatVars(trainData)

	# Run k-prototypes and save the cluster of each example
	kproto = KPrototypes(n_clusters=numClusters, init='Cao', verbose=0)
	clusters = kproto.fit_predict(trainData, categorical=numCatVars)

	# Print cluster centroids of the trained model.
	printCentroidInfo(kproto.cluster_centroids_,numericalFeatures,categoricalFeatures)
	# Print training statistics
	# print(kproto.cost_)
	# Iteration
	# print(kproto.n_iter_)
	# Info of each cluster centroid (features)
	# print(kproto.cluster_centroids_)

	# Print table
	printTable(kproto,priceCategories)

	# Add dummies to the df
	dfWithDummies = pd.get_dummies(df)
	# Add a "cluster" column
	dfWithDummies['clusters'] = clusters

	# Use PCA to reduce the dimensions of the df to 2 to plot
	pca = PCA(2)
	plot_columns = pca.fit_transform(dfWithDummies)

	# Plot the clusters
	plt.scatter(x=plot_columns[:,1], y=plot_columns[:,0], c=dfWithDummies["clusters"])
	plt.show()
