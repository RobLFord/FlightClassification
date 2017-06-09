#Python 3.6.1
from urllib.request import Request, urlopen
import json
import ijson
import numpy as np
import pandas as pd

def create_df(useFile):
	good_attributes = [
		'Id',
		'Man',
		'Spd',
		'Op',
		'OpIcao',
		'CNum',
		'Year',
		'Gnd',
		'Alt',
		'TSecs',
		'Type',
		'To',
		'From',
		'Cou',
		'Species',
		'Mil',
	]

	if useFile:
		filename = '2016-06-20-0000Z.json'
		f = open(filename, 'r', encoding="utf8")
	else:
		#This part works. Able to read JSON 
		req = Request('https://public-api.adsbexchange.com/VirtualRadar/AircraftList.json', headers={'User-Agent': 'Mozilla/5.0'})
		f = json.loads(urlopen(req).read().decode())
		# after this point unable to get objects when coming from web

	objects = ijson.items(f, 'acList.item')
	
	#Parsing json for Flights only
	flights = (o for o in objects if o['Species'] == 1)
	flights = (o for o in flights if o['Mil'] == False)
	
	#Parsing data based on attributes we want to use
	data = []
	for flight in flights:
		row = []
		for attribute in good_attributes:
			row.append(flight.get(attribute))
			
		data.append(row)
	
	# Creating DataFrame
	flight_DF = pd.DataFrame(data, columns=good_attributes)
	
	return flight_DF
	
def preprocessing(df):
	#check to make sure there are not duplicate Ids
	Ids_dup = df.duplicated('Id')
	for IDs in Ids_dup:
		if IDs:
			print('There are some duplicates in Id')
			error = True
		else:
			error = False
		
	if not error:
		#See if there are any missing values
		# print(pd.isnull(df[:30]))
		
		# Remove flights that have landed
		GROUNDED = df[df['Gnd'] == True].index.tolist()
		df = df.drop(df.index[GROUNDED])
		
		# Changing target classifier to have International and Domestic Fields
		US = df.Cou == 'United States'
		NONUS = df.Cou != 'United States'
		
		df.loc[US,'Cou'] = 'Domestic'
		df.loc[NONUS,'Cou'] = 'International'
		
		# Fixing values with negative values
		LESSTHAN_0 = df.Alt < 0
		df.loc[LESSTHAN_0,'Alt'] = None
		
		#Replace the some of missing values with average of column
		df.Spd = df.Spd.fillna(df.Spd.mean().round())
		df.Alt = df.Alt.fillna(df.Alt.mean())
		df.Alt = df.Alt.round()
		df.TSecs = df.TSecs.fillna(df.TSecs.mean())
		df.TSecs = df.TSecs.round()
				
		bad_attributes = [
			'Id',
			'From',
			'To',
			'Gnd',
			'Species',
			'Mil',
			]
			
		for attribute in bad_attributes:
			del df[attribute]
			
	return(error,df)

if __name__ == '__main__':
	df = create_df(True)
	error, df = preprocessing(df)
	
	if not error:
		print(df.head()) #If receive encoding error enter for windows $ chcp 65001
		
		# ------------------- This Section had not been tested yet ------------------------#
		# from sklearn.cross_validation import train_test_split
		# from sklearn.tree import DecisionTreeClassifier
		# from sklearn.metrics import accuracy_score
		# from sklearn import tree
		# X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)
		
		# clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
										# max_depth=3, min_samples_leaf=5)
										
		# clf_gini.fit(X_train, y_train)
		
		# clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
										# max_depth=3, min_samples_leaf=5)
										
		# clf_entropy.fit(X_train, y_train)
		
		# y_pred = clf_gini.predict(X_test)
		
		# y_pred_en = clf_entropy.predict(X_test)
		
		# print("Accuracy of Gini ", accuracy_score(y_test,y_pred)*100)
		
		# print("Accuracy of Entropy ", accuracy_score(y_test,y_pred_en)*100)
		
		#----------------------------------------------------------------------------------#


