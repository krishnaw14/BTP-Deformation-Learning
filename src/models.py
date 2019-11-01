import torch 
import torch.nn as nn 

class GeneratorAppearance(nn.Module):

	def __init__(self, config):
		super(GeneratorAppearance, self).__init__()
		
		self.img_size = config.img_size 
		self.z_dim = config.z_dim

		self.layer_fc = nn.Sequential(
			nn.Linear(self.z_dim, (self.img_size // 16) * (self.img_size // 16) * 80),
			nn.ReLU(),
			)
		self.layer_conv = nn.ModuleList()
		for i in range(config.conv_layers):

			in_channels = config.channels[i-1] if i != 0 else 80
			out_channels = config.channels[i]
			kernel_size = config.kernel_sizes[i]
			stride = config.strides[i]
			pad = config.pads[i]

			layer = nn.ConvTranspose2D(in_channels, out_channels, kernel_size, stride, pad)
			self.layer_conv.append(layer)
			if i == config.conv_layers-1:
				self.layer_conv.append(nn.Sigmoid())
			else:
				self.layer_conv.append(nn.ReLU())
		self.layer_conv = nn.Sequential(*self.layer_conv)

	def forward(self, z):
		out = self.layer_fc(z).view(-1, self.image_size // 16, self.image_size // 16, 80)
		out = self.layer_conv(out)
		return out