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

# Check which columns of df have a percentage of null values (nan) higher than p
def checkNulls(df,p) :
	check_null = df.isnull().sum(axis=0).sort_values(ascending=False)/float(len(df))
	print(check_null[check_null>p])

# Fill nans with 0s for numerical columns and with 'NA' for categorical ones 
# also, drop the Id column
def processData(df) :
	# dfWithDummies = pd.get_dummies(df)
	# df.drop(['SalePrice'],axis=1)
	# All categorical features with NA
	df = df.fillna('NA')

	return df

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
	for row in data : 
		for i in range(len(row)) :
			if isinstance(row[i], str) :
				if i not in catVars : 
					catVars.append(i)

	return catVars

# Main program	    		
if __name__ == '__main__':
	# Read data
	fName = sys.argv[1]
	df = pd.read_csv(fName)

	# Check columns with 50% or more nulls (nans)
	# checkNulls(df,0.5)

	# Drop numerical columns
	dropCols(df,['Id','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea',
		'YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1','MiscVal','MoSold','YrSold','GarageCars',
		'GarageArea','GarageYrBlt','Fireplaces','TotRmsAbvGrd','1stFlrSF','2ndFlrSF','LowQualFinSF',
		'GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BsmtFinSF2','MSSubClass',
		'BsmtUnfSF','TotalBsmtSF','SalePrice','MSZoning','LotFrontage','LotArea'])

	# Process the data (Fill nans)
	df = processData(df)

	# Transform to numpy representation
	trainData = df.as_matrix()

	kmodes_huang = KModes(n_clusters=3, init='Huang', verbose=1)
	clusters = kmodes_huang.fit_predict(trainData)

	# Print cluster centroids of the trained model.
	print('k-modes (Huang) centroids:')
	print(kmodes_huang.cluster_centroids_)
	# Print training statistics
	print('Final training cost: {}'.format(kmodes_huang.cost_))
	print('Training iterations: {}'.format(kmodes_huang.n_iter_))
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