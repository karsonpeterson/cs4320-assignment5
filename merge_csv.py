import pandas as pd
import sys, os.path

def get_data(filename):
	data = pd.read_csv(filename)
	data = data.fillna(0)
	return data

def main(argv):
	if len(argv) > 2:
		first_file = argv[1]
		second_file = argv[2]
	else:
		print('Specify the filenames of the csvs you want to merge.')
		sys.exit(1)

	if os.path.exists(first_file) and os.path.exists(second_file):
		first_df = get_data(first_file)
		second_df = get_data(second_file)
		final_df = pd.concat([first_df, second_df])
		final_df.to_csv('combined_results.csv')
	else:
		print('At least one of these files does not exist.')

	return

if __name__ == '__main__':
	main(sys.argv)
