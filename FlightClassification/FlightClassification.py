#Python 3.6.1
import numpy as np
import pandas as pd
import ijson

def create_df():
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
		'Lat',
		'Long',
		'PosTime'
	]

	useFile = True
	if useFile:
		filename = '2016-06-20-0000Z.json'
		f= open(filename, 'r', encoding="utf8")
		objects = ijson.items(f, 'acList.item')
	#else:
		# Has not been tested yet
		# import urllib.request, json
		# with urllib.request.urlopen("https://public-api.adsbexchange.com/VirtualRadar/AircraftList.json") as url:
			# objects = json.loads(url.read().decode())

	#Parsing json for Flights only
	flights = (o for o in objects if o['Species'] == 1) #filter on ground aircraft
	flights = (o for o in flights if o['Mil'] == False) #only look at civilian aviation
	
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
		df = df.drop(df.index[GROUNDED],)

		#Remove flights without GPS information
		no_gps = df[pd.isnull(df['Lat'])].index.tolist()
		df = df.drop(no_gps)

		#Remove flights without Destination Information
		#WARNING: only for training set pre-process
		#Not sure if this is the final strategy or need to utilize 3rd party destination source
		no_dest = df[pd.isnull(df['To'])].index.tolist()
		df = df.drop(no_dest)
		
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

		#Limit results to near JFK Int'l Airport
		#JFK Latitude: 40.644623
		#JFK Longitude: -73.784180
		#Let's try a 50 mi radius around the airport, using www.gpsvisualizer.com
		#dlat = .726358
		#dlon = .972934
		#Max Latitude: 41.370981
		#Min Latitude: 39.918265
		#Max Longitude: -72.811246
		#Min Longitude: -74.757114

		#Latitude Constraint
		outside_latitude_bounds = df[df['Lat'] < 39.9182].index.tolist()
		outside_latitude_bounds += df[df['Lat'] > 41.3710].index.tolist()
		df = df.drop(outside_latitude_bounds)

		#Longitude Constraint
		outside_longitude_bounds = df[df['Long'] < -74.7571].index.tolist()
		outside_longitude_bounds += df[df['Long'] > -72.8112].index.tolist()
		df = df.drop(outside_longitude_bounds)
				
		bad_attributes = [
			'Id',
			'Gnd',
			'Species',
			'Mil',
		]
			
		for attribute in bad_attributes:
			del df[attribute]
			
	return(error,df)

if __name__ == '__main__':
	df = create_df()
	error, df = preprocessing(df)
	
	if not error:
		print(df.to_string()) #If receive encoding error enter for windows $ chcp 65001
		json_out = df.to_json()
		csv_out = df.to_csv()
		json_out_file = open("test_out.json", "w")
		json_out_file.write(json_out)
		json_out_file.close()
		csv_out_file = open("test_out.csv", "w")
		csv_out_file.write(csv_out)
		csv_out_file.close()
		print("Done.")
		
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


