#Python 3.6.1
import numpy as np
import pandas as pd
import ijson
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
import time

def create_df(path):
	good_attributes = [
		'Id',
		'Man',
		'Spd',
		'Op',
		'OpIcao',
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

	file_number = 0
	data = []
	for filename in os.listdir(path):
		file_number += 1
		filepath = os.path.join(path, filename)
		print('Grabbing data from : '  + filepath)
		
		f= open(filepath, 'r', encoding="utf8")
		objects = ijson.items(f, 'acList.item')
		
		#Parsing json for Flights only
		flights = (o for o in objects if o['Species'] == 1) #filter on ground aircraft
		flights = (o for o in flights if o['Mil'] == False) #only look at civilian aviation
		
		#Parsing data based on attributes we want to use
		for flight in flights:
			row = []
			for attribute in good_attributes:
				row.append(flight.get(attribute))
			data.append(row)
			
	print('Total number of files read : ' + str(file_number))
	
	# Creating DataFrame
	flight_DF = pd.DataFrame(data, columns=good_attributes)
	print('Total number of records ' + str(flight_DF.shape[0]))
	return flight_DF

def output_raw_csv(df):
	filename = "raw_df_" + time.strftime("%b%d%Y_%H_%M_%S", time.localtime()) + ".csv"
	csv_out = df.to_csv(encoding='utf-8')
	csv_out_file = open(filename, "w",encoding='utf-8')
	csv_out_file.write(csv_out)
	csv_out_file.close()


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

	df.loc[US,'Cou'] = 0 # 0 = Domestic
	df.loc[NONUS,'Cou'] = 1 #1 = International

	# Boolean for whether the flight is a major US carrier
	us_icaos = ["JBU", "AAL", "DAL", "UAL", "ASA", "AAY", "FFT", "HAL", "SWA", "NKS", "VRD", "ENY", "ASQ", "SKW"]
	icao_pattern = '|'.join(us_icaos)
	df['MajorUsCarrier'] = df['OpIcao'].str.contains(icao_pattern)

	# Fixing values with negative values
	LESSTHAN_0 = df.Alt < 0
	df.loc[LESSTHAN_0,'Alt'] = 0

	# Extract Fields from Epoch Time
	df['DateTime'] = pd.to_datetime(df['PosTime'], unit='ms')
	df['Weekday'] = df['DateTime'].dt.dayofweek

	# Cruising Speed Categorization
	# Threshold: 400 knots
	df['CruisingSpd'] = df['Spd'] > 400.

	#Altitude Categorization
	#High = 30,000ft+
	#Medium = 10,000ft - 30,000ft
	#Low = Below 10,000ft
	df['AltCat'] = pd.cut(df['Alt'], bins=[0., 10000., 30000.,100000.], include_lowest=True, labels=[int(0),int(1),int(2)])

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
		'Year',
		'Alt',
		'Spd',
		'Op',
		'OpIcao',
		'TSecs',
		'To',
		'From',
		'Type',
		'Man'
	]

	for attribute in bad_attributes:
		del df[attribute]
	return df

def write_dot(tree, feature_names, filename):
    with open(filename, 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

def main():
	path = '.\\Data'

	df = create_df(path)
	output_raw_csv(df)
	df = preprocessing(df)

	output_json_csv = False
	if(output_json_csv):
		json_out = df.to_json()
		csv_out = df.to_csv()
		json_out_file = open("out.json", "w")
		json_out_file.write(json_out)
		json_out_file.close()
		csv_out_file = open("out.csv", "w")
		csv_out_file.write(csv_out)
		csv_out_file.close()

	print(df)
	print("Resultant Data Set Contains " + str(df.shape[0]) + " Records...")
	print("Done.")


	features = list(df.columns[:5])
	print(" Features: ", features, sep='\n')
	target = df.columns[5]
	print(" Target: ", target,sep='\n')

	y = df['Intl']
	x = df[features]

	#Automatically split into training and test, then run classification and output accuracy numbers
	X_train, X_test, y_train, y_test = train_test_split( x, y, test_size = 0.3, random_state = 100)
	clf_gini = DecisionTreeClassifier(criterion = "gini", min_samples_leaf=2)

	clf_gini.fit(X_train, y_train)
	clf_entropy = DecisionTreeClassifier(criterion = "entropy",  min_samples_leaf=2)
	clf_entropy.fit(X_train, y_train)

	y_pred = clf_gini.predict(X_test)
	y_pred_en = clf_entropy.predict(X_test)

	print("Accuracy of Gini ", accuracy_score(y_test,y_pred)*100)
	print("Accuracy of Entropy ", accuracy_score(y_test,y_pred_en)*100)

	write_dot(clf_gini, features, "gini.dot")
	write_dot(clf_entropy, features, "entropy.dot")

if __name__ == '__main__':
	main()
