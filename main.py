import argparse
import os
import numpy as np
import torch

from src.trainers import Trainer
import config

# def get_args():
# 	parser = argparse.ArgumentParser(description='Generative Model')
# 	parser.add_argument('--config', '-c', type=str, required=True,
# 		help='Config File for the particular model and data settings to be specified to be specified')
# 	parser.add_argument('--eval_only',  action='store_true')

# 	args = parser.parse_args()
# 	return args

def main(args=None):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	trainer = Trainer(config, device)
	trainer.train()
	


if __name__ == '__main__':
	main()