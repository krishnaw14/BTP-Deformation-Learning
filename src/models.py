import torch 
import torch.nn as nn 
import torch.nn.functional as F

# from .utils import image_warp

class GeneratorAppearance(nn.Module):

	def __init__(self, config):
		super(GeneratorAppearance, self).__init__()
		
		self.img_size = config.img_size 
		self.z_dim = config.z_dim_app

		self.layer_fc = nn.Sequential(
			nn.Linear(self.z_dim, (self.img_size // 16) * (self.img_size // 16) * config.channels_app[0]*2),
			nn.ReLU(),
			)

		self.layer_conv = nn.ModuleList()
		for i in range(config.conv_layers_app):

			in_channels = config.channels_app[i-1] if i != 0 else config.channels_app[0]*2
			out_channels = config.channels_app[i]
			kernel_size = config.kernel_sizes_app[i]
			stride = config.strides_app[i]
			pad = config.pads_app[i]

			layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, pad)
			self.layer_conv.append(layer)
			if i == config.conv_layers_app-1:
				self.layer_conv.append(nn.Sigmoid())
			else:
				self.layer_conv.append(nn.ReLU())
		self.layer_conv = nn.Sequential(*self.layer_conv)

	def forward(self, z):
		out = self.layer_fc(z).view(-1, 80, self.img_size // 16, self.img_size // 16)
		out = self.layer_conv(out)
		return out

class GeneratorGeometry(nn.Module):

	def __init__(self, config):
		super(GeneratorGeometry, self).__init__()

		self.img_size = config.img_size
		self.z_dim = config.z_dim_geo

		self.layer_fc = nn.Sequential(
			nn.Linear(self.z_dim, (self.img_size // 16) * (self.img_size // 16) * config.channels_geo[0]*2),
			nn.ReLU(),
			)

		self.layer_conv = nn.ModuleList()
		for i in range(config.conv_layers_geo):

			in_channels = config.channels_geo[i-1] if i != 0 else config.channels_geo[0]*2
			out_channels = config.channels_geo[i]
			kernel_size = config.kernel_sizes_geo[i]
			stride = config.strides_geo[i]
			pad = config.pads_geo[i]

			layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, pad)
			self.layer_conv.append(layer)
			if i == config.conv_layers_geo-1:
				self.layer_conv.append(nn.Tanh())
			else:
				self.layer_conv.append(nn.ReLU())
		self.layer_conv = nn.Sequential(*self.layer_conv)

	def forward(self, z):
		out = self.layer_fc(z).view(-1, 128, self.img_size // 16, self.img_size // 16)
		out = self.layer_conv(out)
		return out

class GeneratorDeform(nn.Module):

	def __init__(self, config, device):
		super(GeneratorDeform, self).__init__()

		self.device = device
		self.img_size = config.img_size
		self.geo_scale = config.geo_scale

		self.generator_appearance = GeneratorAppearance(config)
		self.generator_geometry = GeneratorGeometry(config)

		self.make_grid(config.batch_size)

	def make_grid(self, batch_size):
		x_t, y_t = torch.meshgrid(torch.linspace(-1, 1, 64), torch.linspace(-1, 1, 64))
		grid = torch.cat((x_t.unsqueeze(0), y_t.unsqueeze(0)), dim=0)

		self.grid = grid.unsqueeze(0).repeat(128, 1, 1, 1).to(self.device)

	def image_warp(self, gen_app, gen_geo):

		template = gen_app

		grid_ = (self.grid+1)*self.img_size/2
		deformation = gen_geo*self.geo_scale + grid_
		deformation = torch.clamp(deformation, min=0, max=self.img_size)
		deformation = 2*(deformation/self.img_size)-1

		# import pdb; pdb.set_trace()

		img_recon = F.grid_sample(template, deformation.permute(0,2,3,1))
		return img_recon

	def forward(self, z_app, z_geo):

		gen_app = self.generator_appearance(z_app)
		gen_geo = self.generator_geometry(z_geo)

		# img_recon = image_warp(template=gen_app, deformation=gen_geo)
		img_recon = self.image_warp(gen_app, gen_geo)

		return gen_app, gen_geo, img_recon






