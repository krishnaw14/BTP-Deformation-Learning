import os
from tqdm import tqdm
import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torchvision.utils import save_image

from .models import *
from .utils import *
from .data import *

class InverseTrainer(object):

	def __init__(self, config, device):

		self.model = GeneratorDeform(config, device).to(device)
		self.inverse_model = GeneratorGeometry(config)
		self.data_loader = get_image_folder_data_loader(config)

		self.batch_size = config.batch_size
		self.sigma = config.sigma
		self.sampling_step = config.sampling_step
		self.step_size = config.step_size

		self.optimizer = torch.optim.Adam(self.model.parameters(), lr = config.lr)
		self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = config.lr_step_size, gamma=config.lr_gamma)
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

	def langevin_dynamics(self, img, z_app, z_geo=None):
		
		for i in range(self.sampling_step):
			noise_app = torch.randn_like(z_app)
			noise_geo = torch.randn_like(z_geo)

			gen_app, gen_geo, img_recon = self.model(z_app, z_geo)

			loss = torch.sum((gen_app-img_recon)**2)/self.sigma / self.sigma / self.batch_size
			loss += torch.sum(z_app**2) 
			loss *= 0.5

			grad_app = autograd.grad(loss, z_app, retain_graph=True)[0]
			z_app = z_app - 0.5 * self.step_size * self.step_size * grad_app + self.step_size * noise_app

		return z_app, z_geo

	def train(self):

		z_app = torch.randn(len(self.data_loader.dataset), self.model.generator_appearance.z_dim).to(self.device)
		z_app.requires_grad = True
		z_geo = torch.randn(len(self.data_loader.dataset), self.model.generator_geometry.z_dim).to(self.device)
		z_geo.requires_grad = True

		epoch_loss_values = []

		for epoch in range(self.num_epochs):
			epoch_loss = 0.0
			pbar = tqdm(enumerate(self.data_loader), desc = 'training batch_loss', 
				total=np.ceil(len(self.data_loader.dataset) / self.data_loader.batch_size))

			for (nbatch, data) in pbar:
				# import pdb; pdb.set_trace()

				img = data[0].to(self.device)
				batch_idx = torch.arange(nbatch * self.batch_size, min((nbatch+1) * self.batch_size, len(self.data_loader.dataset) ))

				if len(batch_idx) != self.batch_size:
					continue

				# Infer z
				z_app_batch = z_app[batch_idx]
				z_geo_batch = z_geo[batch_idx]

				z_app_batch_infer, z_geo_batch_infer = self.langevin_dynamics(img, z_app_batch, z_geo_batch)
				with torch.no_grad():
					z_app[batch_idx] = z_app_batch_infer
					z_geo[batch_idx] = z_geo_batch_infer

				gen_app, gen_geo, img_recon = self.model(z_app_batch_infer, z_geo_batch_infer)

				if epoch%self.log_step == 0 and nbatch == 0:
					self.save_results(epoch, img, img_recon, gen_app, gen_geo)
					self.save_params(epoch, 0.0)

				loss = self.loss_func(img, gen_app)/self.sigma / self.sigma 
				loss *= 0.5

				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

				pbar.set_description('Epoch: {}, Batch Loss: {}'.format(epoch, loss.item()))

				epoch_loss += loss

			print('Epoch Loss:', epoch_loss.item())
			self.lr_scheduler.step()
			epoch_loss_values.append(epoch_loss.cpu().item())

		import pdb; pdb.set_trace()

	def save_results(self, epoch, img, img_recon, gen_app, gen_geo):
		save_image(img, os.path.join(self.save_results_dir, 'original_epoch_{}.png'.format(epoch)), nrow=8, normalize=True)
		save_image(img_recon, os.path.join(self.save_results_dir, 'recon_epoch_{}.png'.format(epoch)), nrow=8, normalize=True)
		save_image(gen_app, os.path.join(self.save_results_dir, 'app_epoch_{}.png'.format(epoch)), nrow=8, normalize=True)

