import os
from tqdm import tqdm
import numpy as np

import torch 
import torch.nn as nn

from .models import *
from .utils import *
from .data import *

class AppearanceGeneratorTrainer(object):

	def __init__(self, config, device):

		self.model = GeneratorAppearance(config).to(device)
		# import pdb; pdb.set_trace()

		self.data_loader = get_image_folder_data_loader(config)

		self.batch_size = config.batch_size
		self.sigma = config.sigma

		self.optimizer = torch.optim.Adam(self.model.parameters(), lr = config.lr)
		self.num_epochs = config.num_epochs
		self.save_param_step = config.save_param_step
		self.log_step = config.log_step
		self.val_step = config.val_step

		self.save_param_dir = config.save_param_dir
		self.save_results_dir = config.save_results_dir

		os.makedirs(self.save_param_dir, exist_ok=True)
		os.makedirs(self.save_results_dir, exist_ok=True)

		self.device = device

		self.loss_func = nn.MSELoss().cuda()

	def save_params(self, epoch, total_loss, is_best=False):
		torch.save({
					'model_state_dict': self.model.state_dict(),
					'optimizer_state_dict': self.optimizer.state_dict(),
					'total_loss': total_loss, 
					'is_best': is_best
					}, os.path.join(self.save_param_dir, 'epoch_{}.tar'.format(epoch))
					)

	def load_params(self, path):
		checkpoint = torch.load(PATH)
		self.model.load_state_dict(checkpoint['model_state_dict'])
		self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

	def train(self):
		# z_app = torch.randn(len(self.data_loader.dataset), self.model.z_dim)
		# z_app = z_app.to(self.device)

		for epoch in range(self.num_epochs):
			epoch_loss = 0.0
			pbar = tqdm(enumerate(self.data_loader), desc = 'training batch_loss', 
				total=np.ceil(len(self.data_loader.dataset) / self.data_loader.batch_size))

			for (nbatch, data) in pbar:

				img = data[0].to(self.device)
				# batch_idx = torch.arange(nbatch * self.batch_size, (nbatch+1) * self.batch_size)
				# z_app_batch = z_app[batch_idx]
				z_app_batch = torch.randn(img.shape[0], self.model.z_dim).to(self.device)

				try:
					img_recon = self.model(z_app_batch)
					loss = self.loss_func(img, img_recon)/self.sigma / self.sigma / 100
					self.optimizer.zero_grad()
					loss.backward()
					self.optimizer.step()
				except Exception as e:
					import pdb; pdb.set_trace()

				pbar.set_description('Epoch: {}, Batch Loss: {}'.format(epoch, loss.item()))

				epoch_loss += loss

			print('Epoch Loss:', epoch_loss.item())


	def validate(self):
		pass

	def save_results(self):
		pass
