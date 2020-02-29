import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
import sys, os.path
import process_data

def get_data(filename):
	data = pd.read_csv(filename)
	return data

def get_compact_columns(data):
	return data[['Season', 'WTeamID', 'LTeamID']]

def join_data(data1, data2):
	return



def main(argv):
	if len(argv) > 1:
		seed_filename = argv[1]
		compact_filename = argv[2]
	else:
		seed_filename = 'MNCAATourneySeeds.csv'
		compact_filename = 'MNCAATourneyCompactResults.csv'

	if os.path.exists(seed_filename) and os.path.exists(compact_filename):
		# basename, ext = filename.split('.')
		seed_data = get_data(seed_filename)
		compact_data = get_data(compact_filename)
		compact_data = get_compact_columns(compact_data)
		print()
		print(seed_data)
		print()
		print(compact_data)



		# data = join_data(seed_data, compact_data)

if __name__ == '__main__':
	main(sys.argv)