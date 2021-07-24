#!/usr/bin/env python3

import os
import glob

def count_one_file(data_file, counts):
	with open(data_file, 'r') as rf: 
		process = os.popen('cat ' + data_file + '| ' + 'wc ' + '-l ')
		response = process.read()
		file_name = data_file.split('/')[-1]
		counts[file_name] = int(response)
		process.close()

def main():
	counts = {}
	
	list_of_files = sorted(glob.glob('./rainbow/*.py'))
	for data_file in list_of_files:
		print("Processing " + data_file + "...")
		count_one_file(data_file, counts)
	print(counts)
	
if __name__ == '__main__':
	main()
