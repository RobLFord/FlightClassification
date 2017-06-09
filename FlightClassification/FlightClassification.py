import pandas as pd
import ijson

if __name__ == '__main__':
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
	]

	filename = '2016-06-20-0000Z.json'
	f= open(filename, 'r', encoding="utf8")
	objects = ijson.items(f, 'acList.item')

	#Parsing json for Flights only
	flights = (o for o in objects if o['Species'] == 1)
	
	#Parsing data based on attributes we want to use
	data = []
	for flight in flights:
		row = []
		for attribute in good_attributes:
			row.append(flight.get(attribute))

		data.append(row)

	# Creating DataFrame
	flight_DF = pd.DataFrame(data, columns=good_attributes)

	#Covert Cou to Domestic (US) and Internation Non US


	#Check other data and see if we need to covert to different types for decision tree.

	#Check first 10 row in the dataframe
	
	print(flight_DF[:10]) #If receive encoding error enter for windows $ chcp 65001

