import argparse
import os
import numpy as np

def get_args():
	parser = argparse.ArgumentParser(description='Generative Model')
	parser.add_argument('--config', '-c', type=str, required=True,
		help='Config File for the particular model and data settings to be specified to be specified')
	parser.add_argument('--eval_only',  action='store_true')

	args = parser.parse_args()
	return args

def main(args):

	


if __name__ == '__main__':
	main(get_args())