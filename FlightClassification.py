#Python 3.6.1
import numpy as np
import pandas as pd
import ijson
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import time
import sys
import matplotlib.pyplot as plt

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
		flights = (o for o in objects if o['Species'] == 1) #filter on aircraft type
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
	filename = ".\\DataFrames\\raw_df.csv"
	csv_out = df.to_csv(encoding='utf-8')
	csv_out_file = open(filename, "w",encoding='utf-8')
	csv_out_file.write(csv_out)
	csv_out_file.close()


def preprocessing(df, lowAlt, medAlt, highAlt, cruising_threshold):

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

	df.loc[US,'Cou'] = False # 0 = Domestic
	df.loc[NONUS,'Cou'] = True #1 = International

	# Boolean for whether the flight is a major US carrier
	us_icaos = MAJORUS
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
	df['CruisingSpd'] = df['Spd'] > cruising_threshold

	#Altitude Categorization
	#High = 30,000ft+
	#Medium = 10,000ft - 30,000ft
	#Low = Below 10,000ft
	df['AltCat'] = pd.cut(df['Alt'], bins=[0., lowAlt, medAlt,highAlt], include_lowest=True, labels=[int(0),int(1),int(2)])

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
						
						
def write_json_csv(file_type, file_name, df):
	if(file_type == 'json'):
		json_out = df.to_json()
		json_out_file = open(file_name, "w")
		json_out_file.write(json_out)
		json_out_file.close()
	elif(file_type == 'csv'):
		csv_out = df.to_csv()
		csv_out_file = open(file_name, "w")
		csv_out_file.write(csv_out)
		csv_out_file.close()
	else:
		print('Please specify correct file type')
		
		
def decisionTree(X_train,
				X_test,
				y_train,
				y_test,
				features,
				max_depth,
				min_samples_split,
				min_samples_leaf,
				min_weight_fraction_leaf,
				max_leaf_nodes,
				file, model):
				
	clf_gini = DecisionTreeClassifier(criterion = "gini",
									max_depth=max_depth,
									min_samples_split=min_samples_split,
									min_samples_leaf=min_samples_leaf,
									min_weight_fraction_leaf=min_weight_fraction_leaf,
									max_leaf_nodes=max_leaf_nodes)
									
	clf_gini.fit(X_train, y_train)
	y_pred = clf_gini.predict(X_test)
	print("Accuracy of Gini ", accuracy_score(y_test,y_pred)*100)
	
	file.write("\nAccuracy of Gini " + str(accuracy_score(y_test,y_pred)*100))
	file.flush()
	
	dot_file_name = ".\\Dots\\gini_model_" + str(model) + '.dot'
	write_dot(clf_gini, features, dot_file_name)
	
def naiveBayes(X_train, X_test, y_train, y_test, file):
	clf_naive = GaussianNB()
	clf_naive.fit(X_train, y_train)
	y_pred_na = clf_naive.predict(X_test)
	print("Accuracy of Naive Bayes ", accuracy_score(y_test,y_pred_na)*100)
	
	file.write("\nAccuracy of Naive Bayes " + str(accuracy_score(y_test,y_pred_na)*100))
	file.flush()
	
def data_summary(file_name, df):
	summary_file = open(file_name, "w")
	summary_file.write(str(df.describe()))
	summary_file.close()
	
def before_preprocess_visual(df):
	pd.options.mode.chained_assignment = None
	raw_df = df[['Alt', 'Spd','OpIcao']]
	
	# Fixing values with negative values
	LESSTHAN_0 = raw_df.Alt < 0
	raw_df.loc[LESSTHAN_0,'Alt'] = 0
	
	print("Calculating summary of visual data")
	data_summary('.\\Summary\\raw_visual_data_before_preprocess_summary.txt', raw_df)
	
	raw_df['FlightData'] = raw_df['Alt'].index.tolist()
	
	fontsize=12
	dpi=500
	figsize=(15, 8)
	
	print("Creating plots of raw data")
	xlim = (0,len(raw_df['FlightData']))
	ylim = (0,125e3)
	alt_scat_plot = raw_df.plot(kind='scatter', x='FlightData', y='Alt', xlim=xlim, ylim=ylim, grid=True, title="Altitude ADS-B Records", figsize=figsize, fontsize=fontsize)
	alt_scat_plot.set_xlabel("Record Number")
	alt_scat_plot.set_ylabel("Altitude (Feet)")
	plt.savefig('.\\Plots\\alt_scat_plot_raw.jpg', dpi=dpi)
	plt.clf()
	
	alt_box_plot = raw_df.plot(kind='box', x='FlightData', y='Alt', ylim=ylim, grid=True, title="Altitude ADS-B Records", figsize=figsize, fontsize=fontsize)
	alt_box_plot.set_ylabel("Altitude (Feet)")
	plt.savefig('.\\Plots\\alt_box_plot_raw.jpg', dpi=dpi)
	plt.clf()
	
	ylim = (0,605)
	spd_scat_plot = raw_df.plot(kind='scatter', x='FlightData', y='Spd', xlim=xlim, ylim=ylim, grid=True, title="Speed ADS-B Records", figsize=figsize, fontsize=fontsize)
	spd_scat_plot.set_xlabel("Record Number")
	spd_scat_plot.set_ylabel("Speed (Knots)")
	plt.savefig('.\\Plots\\spd_scat_plot_raw.jpg', dpi=dpi)
	plt.clf()
	
	spd_box_plot = raw_df.plot(kind='box', x='FlightData', y='Spd', ylim=ylim, grid=True, title="Speed ADS-B Records", figsize=figsize, fontsize=fontsize)
	spd_box_plot.set_ylabel("Speed (Knots)")
	plt.savefig('.\\Plots\\spd_box_plot_raw.jpg', dpi=dpi)
	plt.clf()
	
	index_list = []
	for i in MAJORUS:
		index_list += raw_df[raw_df['OpIcao'] == i].index.tolist()
		
	raw_opicao_df = raw_df.loc[index_list]
	opicao_group = raw_opicao_df.groupby('OpIcao').size().sort_values(ascending=False)
	opicao_bar_plot = opicao_group.plot(kind='bar', grid=True, title="Major U.S OpIcao ADS-B Records", figsize=figsize, fontsize=fontsize)
	opicao_bar_plot.set_xlabel("OpIcao")
	opicao_bar_plot.set_ylabel("Total")
	plt.savefig('.\\Plots\\OpIcao_bar_plot_raw.jpg', dpi=dpi)
	plt.clf()
	
def after_preprocess_visual(df, model, remove_attribute):
	fontsize=12
	dpi=500
	figsize=(15, 8)
	
	if(remove_attribute != 5):
		cou_group = df.groupby('Cou').size().sort_values(ascending=False)
		cou_bar_plot = cou_group.plot(kind='bar', title="U.S Country", figsize=figsize, fontsize=fontsize, grid=True)
		cou_bar_plot.set_xlabel("U.S Country")
		cou_bar_plot.set_ylabel("Total")
		fig_name = '.\\Plots\\country_bar_plot_model_' + str(model) + '.jpg'
		plt.savefig(fig_name, dpi=dpi)
		plt.clf()
	
	if(remove_attribute != 4):
		majorCarrier_group = df.groupby('MajorUsCarrier').size().sort_values(ascending=False)
		majorCarrier_bar_plot = majorCarrier_group.plot(kind='bar', title="Major U.S Carrier", figsize=figsize, fontsize=fontsize, grid=True)
		majorCarrier_bar_plot.set_xlabel("Major U.S Carrier")
		majorCarrier_bar_plot.set_ylabel("Total")
		fig_name = '.\\Plots\\MajorUsCarrier_bar_plot_model_' + str(model) + '.jpg'
		plt.savefig(fig_name, dpi=dpi)
		plt.clf()
	
	if(remove_attribute != 1):
		cruising_group = df.groupby('CruisingSpd').size().sort_values(ascending=False)
		cruising_bar_plot = cruising_group.plot(kind='bar', title="Cruising Speed", figsize=figsize, fontsize=fontsize, grid=True)
		cruising_bar_plot.set_xlabel("Cruising Speed")
		cruising_bar_plot.set_ylabel("Total")
		fig_name = '.\\Plots\\cruising_bar_plot_model_' + str(model) + '.jpg'
		plt.savefig(fig_name, dpi=dpi)
		plt.clf()
	
	if(remove_attribute != 2):
		altCat_group = df.groupby('AltCat').size().sort_values(ascending=False)
		altCat_bar_plot = altCat_group.plot(kind='bar', title="Altitude Catagory", figsize=figsize, fontsize=fontsize, grid=True)
		altCat_bar_plot.set_xlabel("Altitude Catagory 0=low 1=med 2=high")
		altCat_bar_plot.set_ylabel("Total")
		fig_name = '.\\Plots\\alt_cat_bar_plot_model_' + str(model) + '.jpg'
		plt.savefig(fig_name, dpi=dpi)
		plt.clf()
	
	international_group = df.groupby('Intl').size().sort_values(ascending=False)
	international_bar_plot = international_group.plot(kind='bar', title="International vs Domestic", figsize=figsize, fontsize=fontsize, grid=True)
	international_bar_plot.set_xlabel("International")
	international_bar_plot.set_ylabel("Total")
	fig_name = '.\\Plots\\international_bar_plot_model_' + str(model) + '.jpg'
	plt.savefig(fig_name, dpi=dpi)
	plt.clf()
	
def make_models(df):
	model_log_file = '.\\Logs\\Models_log.txt'
	model_log = open(model_log_file, "w")
	for i in range(1,7):
		
		print('\nModel : ', i)
		model_log.write("\n----------------------------------------------------------------------------------\n")
		model_log.flush()
		model_log.write('Model : ' + str(i))
		model_log.flush()
		
		remove_attribute = 0
		if(i == 1):
			# Preprocessing parameters
			lowAlt = 10000.
			medAlt = 29000.
			highAlt = 100000.
			cruising_threshold = 375.
			# Decision tree classifier parameters
			max_depth = None # int
			min_samples_split = 5
			min_samples_leaf = 5
			min_weight_fraction_leaf = 0 # float 
			max_leaf_nodes = None # int
		elif(i == 2):
			# Preprocessing parameters
			# No change
			
			# Decision tree classifier parameters
			# max_depth = None (No change)
			# min_samples_split = (No change)
			min_samples_leaf = 1 # --- changed to improve accuracy---
			# min_weight_fraction_leaf = (No change)
			# max_leaf_nodes = (No change)
		elif(i == 3):
			# Preprocessing parameters
			lowAlt = 17050.
			medAlt = 32000.
			highAlt = 66000.
			# cruising_threshold = (No change)
			# Decision tree classifier parameters
			# No change
		elif(i == 4):
			# Preprocessing parameters
			# lowAlt = (No change)
			# medAlt = (No change)
			# highAlt = (No change)
			cruising_threshold = 248.252883 # Equal to 1 std from mean from before preprocess summary
			# Decision tree classifier parameters
			# No change
		elif(i == 5):
			# Preprocessing parameters
			# lowAlt = (No change)
			# medAlt = (No change)
			# highAlt = (No change)
			cruising_threshold = 394.646672 # Equal to mean from before preprocess summary
			# Decision tree classifier parameters
			# No change
		elif(i == 6):
			# Preprocessing parameters
			# No change
			
			# Decision tree classifier parameters
			# No change
			
			#Remove a attribute to see effect.
			remove_attribute = 1
			
		print("Preprocessing")
		processed_df = preprocessing(df, lowAlt, medAlt, highAlt, cruising_threshold)
		
		if(remove_attribute == 1):
			del processed_df['CruisingSpd']
		elif(remove_attribute == 2):
			del processed_df['AltCat']
		elif(remove_attribute == 3):
			del processed_df['Weekday']
		elif(remove_attribute == 4):
			del processed_df['MajorUsCarrier']
		elif(remove_attribute == 5):
			del processed_df['Cou']
			
		# Visualization after proprocessing
		print("Creating plots of processed data")
		after_preprocess_visual(processed_df, i, remove_attribute)
		
		file_name = '.\\Dataframes\\Model_' + str(i) + '.csv'
		write_json_csv('csv', file_name, processed_df)

		print("Resultant Data Set Contains " + str(processed_df.shape[0]) + " Records...")
		model_log.write("\nResultant Data Set Contains " + str(processed_df.shape[0]) + " Records...")
		model_log.flush()
		
		features = list(processed_df.columns[:(len(processed_df.columns)-1)])
		print("Features: ", features)
		
		model_log.write("\nFeatures: ")
		model_log.flush()
		
		for item in features:
			model_log.write(item + ", ")
			model_log.flush()
		
		target = processed_df.columns[(len(processed_df.columns)-1)]
		print("Target: ", target)
		model_log.write("\nTarget: " + target)
		model_log.flush()

		y = processed_df['Intl']
		x = processed_df[features]

		#Automatically split into training and test, then run classification and output accuracy numbers
		X_train, X_test, y_train, y_test = train_test_split( x, y, test_size = 0.3, random_state = 100)
		decisionTree(X_train,
					X_test,
					y_train,
					y_test,
					features,
					max_depth,
					min_samples_split,
					min_samples_leaf,
					min_weight_fraction_leaf,
					max_leaf_nodes,
					model_log, i)
		naiveBayes(X_train, X_test, y_train, y_test, model_log)
		model_log.write("\n----------------------------------------------------------------------------------\n")
		model_log.flush()
		
	model_log.close()
		
def main():
	path = '.\\Data'

	if(len(sys.argv) > 1):
		if(sys.argv[1] == 'new_data'):
			df_json = create_df(path)
			print("Storing raw data to CSV")
			output_raw_csv(df_json)
			print("Reading raw data from CSV")
			df = pd.read_csv('.\\DataFrames\\raw_df.csv')
	else:
		print("Reading raw data from CSV")
		df = pd.read_csv('.\\DataFrames\\raw_df.csv')
		
	del df['Unnamed: 0']
	
	# Summary of data
	print_summary = True
	if(print_summary):
		print("Calculating summary of raw data")
		data_summary('.\\Summary\\raw_data_summary.txt', df)
	
	# Visualiztion before preprocessing data
	before_preprocess_visual(df)
	
	print("Creating models from data")
	make_models(df)

if __name__ == '__main__':
	MAJORUS = ["JBU", "AAL", "DAL", "UAL", "ASA", "AAY", "FFT", "HAL", "SWA", "NKS", "VRD", "ENY", "ASQ", "SKW"]
	main()
