import sys
import operator
import re, string
import csv
import math
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression
import pandas as pd
import sqlite3
import matplotlib.pylab as plt
# %matplotlib inline

numRows = 1459 
numCols = 79

def checkNulls(df) :
	check_null = df.isnull().sum(axis=0).sort_values(ascending=False)/float(len(df))
	print(check_null[check_null>0.5])

def processData(df) :
	dfWithDummies = pd.get_dummies(df)
	classLabels = df['SalePrice']
	dfWithDummies.drop(['SalePrice'],axis=1)
	dfWithDummies.drop(['Id'],axis=1)
	# LotFrontage still has NAs,change em to 0
	dfWithDummies = dfWithDummies.fillna(0)

	return dfWithDummies,classLabels

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

# Main program	    		
if __name__ == '__main__':
	fName = sys.argv[1]
	df = pd.read_csv(fName)

	# Check % of nulls in each column
	print("Features with more than 50% null values : ")
	checkNulls(df)
	print()

	# Generate dummy variables (PoolQC = NA doesn't have a dummy, because 0 0 0 means it doesn't have a pool already)
	# This applies to other similar features
	dfWithDummies,classLabels = processData(df)
	colNames = list(dfWithDummies.columns.values)

	# Transform to numpy representation
	trainData = dfWithDummies.as_matrix()
	yTrain = classLabels.as_matrix()

	# Checking scores for each feature for analysis purpose
	# Low scores mean that the feature is not very related to the class label
	# and could be eliminated
	
	# Lists of thresholds to test
	scoreThresholdX2 = [300,500,700,800,1000,1200]
	scoreThresholdF = [10,25,50,100,150,300]
	checkScores(trainData,yTrain,scoreThresholdX2,scoreThresholdF)


