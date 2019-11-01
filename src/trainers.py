import os
# from tqdm i

import torch 
import torch.nn as nn

from .models import *
from .utils import *
from .data import *

class AppearanceGeneratorTrainer(object):

	def __init__(self, config, device):

		self.model = GeneratorAppearance(config)
		self.data_loader = get_celeba_data_loader()

		self.batch_size = config.batch_size
		self.sigma = config.sigma

		self.optimizer = torch.optim.Adam(lr = config.lr)
		self.num_epochs = config.num_epochs
		self.save_param_step = config.save_param_step
		self.log_step = config.log_step
		self.val_step = config.val_step

		self.save_param_dir = config.save_param_dir
		self.save_results_dir = config.save_results_dir

		os.makedirs(self.save_param_dir, exists_ok=True)
		os.makedirs(self.save_results_dir, exists_ok=True)

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
		z_app = torch.random.randn(self.data_loader.dataset.__len__(), self.model.z_dim)
		z_app = z_app.to(self.device)

		for epoch in range(self.num_epochs):
			for (nbatch, data) in enumerate(self.data_loader):
				batch_idx = torch.arange(nbatch * self.batch_size, (nbatch+1) * self.batch_size)
				z_app_batch = z_app[batch_idx]

				img_recon = self.model(z_app_batch)
				loss = self.loss_func(data, img_recon)

				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

				print('Loss:', loss.item(), end='')


	def validate(self):
		pass

	def save_results(self):
		pass
