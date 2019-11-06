import torch 
import torch.nn as nn 
import torch.nn.functional as F

from .utils import image_warp_custom

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
			padding = config.pads_app[i]
			output_padding = config.output_pad_app[i]

			layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
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
			padding = config.pads_geo[i]
			output_padding = config.output_pad_geo[i]

			layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
			self.layer_conv.append(layer)
			if i == config.conv_layers_geo-1:
				self.layer_conv.append(nn.Tanh())
			else:
				self.layer_conv.append(nn.ReLU())
		self.layer_conv = nn.Sequential(*self.layer_conv)

	def forward(self, z):
		# import pdb; pdb.set_trace()
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

		self.grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(self.device)

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

		# import pdb; pdb.set_trace()

		img_recon = image_warp_custom(template=gen_app, deformation=self.geo_scale*gen_geo, device=self.device)
		# img_recon = self.image_warp(gen_app, gen_geo)

		return gen_app, gen_geo, img_recon

class EncoderAppearance(nn.Module):

	def __init__(self, config):
		self.img_size = config.img_size
		self.z_dim = config.z_dim_app

		self.layer_conv = nn.ModuleList()
		for i in range(config.conv_layers_app):

			in_channels = config.channels_app[config.conv_layers_app -1 -i] 
			out_channels = config.channels_app[config.conv_layers_app -2 -i] if i != config.conv_layers_app-1 else config.channels_app[-1]*2
			kernel_size = config.kernel_sizes_app[i]
			stride = config.strides_app[i]
			pad = config.pads_app[i]

			layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad)
			self.layer_conv.append(layer)
			if i == config.conv_layers_geo-1:
				self.layer_conv.append(nn.Sigmoid())
			else:
				self.layer_conv.append(nn.ReLU())
		self.layer_conv = nn.Sequential(*self.layer_conv)

		self.layer_fc = nn.Sequential(
			nn.Linear((self.img_size // 16) * (self.img_size // 16) * config.channels_app[0]*2, 2*self.z_dim),
			nn.ReLU(),
			)

	def forward(self, img):

		import pdb; pdb.set_trace()
		out = self.layer_conv(img)
		out = self.layer_fc(out.view(-1,))

		mu = out[:,0]
		logsigma = out[:,1]

		return mu, logsigma

class EncoderGeometry(nn.Module):

	def __init__(self, config):
		self.img_size = config.img_size
		self.z_dim = config.z_dim_geo

		self.layer_conv = nn.ModuleList()
		for i in range(config.conv_layers_geo):

			in_channels = config.channels_geo[config.conv_layers_geo -1 -i] 
			out_channels = config.channels_geo[config.conv_layers_geo -2 -i] if i != config.conv_layers_geo-1 else config.channels_geo[-1]*2
			kernel_size = config.kernel_sizes_geo[i]
			stride = config.strides_geo[i]
			pad = config.pads_geo[i]

			layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad)
			self.layer_conv.append(layer)
			if i == config.conv_layers_geo-1:
				self.layer_conv.append(nn.Sigmoid())
			else:
				self.layer_conv.append(nn.ReLU())
		self.layer_conv = nn.Sequential(*self.layer_conv)

		self.layer_fc = nn.Sequential(
			nn.Linear((self.img_size // 16) * (self.img_size // 16) * config.channels_geo[0]*2, 2*self.z_dim),
			nn.ReLU(),
			)

	def forward(self, img):
		out = self.layer_conv(img)
		out = self.layer_fc(out.view(-1,))

		mu = out[:,0]
		logsigma = out[:,1]

		return mu, logsigma

class VAEDeform(nn.Module):

	def __init__(self, config, device):
		super(VAEDeform, self).__init__()

		self.img_size = config.img_size

		self.decoder = GeneratorDeform(config, device)

		self.encoder_app = EncoderAppearance(config)
		self.encoder_geo = EncoderGeometry(config)

	def reparameterize(self, mu, logsigma):
		pass

	def forward(self, img):
		mu_app, logsigma_app = self.encoder_app(img)
		z_app = reparameterize(mu_app, logsigma_app)

		mu_geo, logsigma_geo = self.encoder_app(img)
		z_geo = reparameterize(mu_geo, logsigma_geo)

		gen_app, gen_geo, img_recon = self.decoder(z_app, z_geo)

		return z_app, z_geo, gen_app, gen_geo, img_recon







