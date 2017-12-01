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
# https://pdfs.semanticscholar.org/1955/c6801bca5e95a44e70ce14180f00fd3e55b8.pdf Cao method

# Histogram bins for 3 clusters with outlayers [  34900. , 274933.33333333 , 514966.66666667, 755000.]
# Histogram bins for 2 clusters with outlayers [  34900.  154900.  274900.]

# 50% or more nulls :
# PoolQC         0.995205
# MiscFeature    0.963014
# Alley          0.937671
# Fence          0.807534

numRows = 1459 
numCols = 79
histoBins = []

numericalFeatures = ['Id', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 
'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 
'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt',
 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 
 'MiscVal', 'MoSold', 'YrSold']


categoricalFeatures = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 
'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 
'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'Functional', 
'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 
'SaleType', 'SaleCondition']


# Check which columns of df have a percentage of null values (nan) higher than p
def checkNulls(df,p) :
	check_null = df.isnull().sum(axis=0).sort_values(ascending=False)/float(len(df))
	print(check_null[check_null>p])

# Fill nans with 0s for numerical columns and with 'NA' for categorical ones 
# also, drop the Id column
def processData(df) :
	# Keep only elements in the first bin :
	df.drop(df[df.SalePrice > 274934].index, inplace=True)

	classLabels = df['SalePrice']
	df.drop(['SalePrice'],axis=1,inplace=True)

	# All categorical features with NA
	df = df.fillna('NA')

	return df,classLabels

# Discretize the prices in i categories
def processPrices(y,binNum) :
	histo = np.histogram(y,binNum)
	bins = histo[1]

	print("------------------ Real histogram division ------------------")
	print(histo[0])
	print(bins)

	newPrices = []
	for i in range(len(y)) :
		for j in range(len(bins)-1) :
			if ((y[i] >= bins[j]) and (y[i] <= bins[j+1])) :
				newPrices.append(j)

	histoBins = bins
	return newPrices

def dropCols(df,cols) :
	for col in cols :
		df.drop([col],axis=1,inplace=True)

# Get columns that are categorical
def getCatVars(data) :
	catVars = []
	for row in data : 
		for i in range(len(row)) :
			if isinstance(row[i], str) :
				if i not in catVars : 
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

def printCentroidInfo(centroids,features) :
	print("------------------ Centroid Information ------------------")

	# Obtain the features with different values in at least one cluster
	diffFeatures = []
	for i in range(len(features)) :
		equal = True
		j = 0
		while (j < len(centroids)-1 and equal) :
			if (centroids[j,i] != centroids[j+1,i]) :
				diffFeatures.append(i)
				break
			j += 1

	# Print all features that affect the clusters and the values associated with them
	for f in diffFeatures :
		print("Feature : "+(features[f])) 
		for j in range(len(centroids)) :
			print("Centroid "+str(j+1)+" : "+centroids[j,f])

	print("Features not shown here have the same value for every cluster")

# Main program	    		
if __name__ == '__main__':
	# Read data
	fName = sys.argv[1]
	numClusters = int(sys.argv[2])
	if (len(sys.argv) > 3) :
		fName2 = sys.argv[2]
		fpreds = sys.argv[3]
		dffeatures = pd.read_csv(fName2)
		dfpreds = pd.read_csv(fpreds)
		df2 = pd.concat([dffeatures,dfpreds],axis=1)

	df = pd.read_csv(fName)

	if (len(sys.argv) > 3) :
		frames = [df,df2]
		df = pd.concat(frames)

	# Check columns with 50% or more nulls (nans)
	# checkNulls(df,0.5)

	# Drop numerical columns
	dropCols(df,numericalFeatures)

	# Process the data (Fill nans)
	df,ydf = processData(df)

	# Transform to numpy representation
	trainData = df.as_matrix()
	y = ydf.as_matrix()

	# Generate prices separated in numClusters bins
	priceCategories = processPrices(y,numClusters)

	# Create and fit the model
	model = KModes(n_clusters=numClusters, init='Cao', verbose=0)
	clusters = model.fit_predict(trainData)

	printCentroidInfo(model.cluster_centroids_,categoricalFeatures)

	printTable(model,priceCategories)

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