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

	# Boolean for whether the flight is a major US carrier
	us_icaos = ["JBU", "AAL", "DAL", "UAL", "ASA", "AAY", "FFT", "HAL", "SWA", "NKS", "VRD", "ENY", "ASQ", "SKW"]
	icao_pattern = '|'.join(us_icaos)
	df['MajorUsCarrier'] = df['OpIcao'].str.contains(icao_pattern)

	# Fixing values with negative values
	LESSTHAN_0 = df.Alt < 0
	df.loc[LESSTHAN_0,'Alt'] = None

	# Extract Fields from Epoch Time
	df['DateTime'] = pd.to_datetime(df['PosTime'], unit='ms')
	df['Weekday'] = df['DateTime'].dt.dayofweek

	# Create boolean field for whether Int'l Flight - training/verification use
	# Note: All US Airport Designators begin with the letter K per the FAA
	df['Intl'] = df['To'].astype(str).str[0] != 'K'
	df['Intl'] = np.logical_or(df['Intl'].astype(bool), df['From'].astype(str).str[0] != 'K')

	###############################################
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
	##############################################

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
		'PosTime',
		'DateTime',
		'Lat',
		'Long',
		'Year'
	]

	for attribute in bad_attributes:
		del df[attribute]
	return df

def main():
	df = create_df()
	df = preprocessing(df)

	print(df.to_string()) #If receive encoding error enter for windows $ chcp 65001
	"""
	json_out = df.to_json()
	csv_out = df.to_csv()
	json_out_file = open("test_out.json", "w")
	json_out_file.write(json_out)
	json_out_file.close()
	csv_out_file = open("test_out.csv", "w")
	csv_out_file.write(csv_out)
	csv_out_file.close()
	"""
	print("Resultant Data Set Contains " + str(df.shape[0]) + " Records...")
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

if __name__ == '__main__':
	main()
