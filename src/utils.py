import torch
import torch.nn.functional as F
import numpy as np

# def make_grid(num_batch, height, width):
#         # This should be equivalent to:
#      x_t, y_t = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
#      ones = np.ones(np.prod(x_t.shape))
#      grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])

def make_grid(batch_size, height, width):
	x_t, y_t = torch.meshgrid(torch.linspace(-1.0, 1.0, width), torch.linspace(-1.0, 1.0, height))
	# ones = torch.ones(np.prod(x_t.numpy().shape))
	grid = torch.cat((x_t.flatten().unsqueeze(0), y_t.flatten().unsqueeze(0)), dim=0).unsqueeze(0)
	grid = grid.view(1, -1)
	grid = grid.repeat(batch_size, 1)
	grid = grid.view(batch_size, 2, -1)
	return grid

def repeat(x, n_repeats):
	rep = torch.ones(1, n_repeats).long()
	x = torch.matmul(x.view(-1,1), rep)
	return x.view(-1)


def interpolate(template, x, y, x_d, y_d, output_size, device):

	batch_size, num_channels, height, width = template.shape[0], template.shape[1], template.shape[2], template.shape[3]
	# x = x.float(); y = y.float()
	# x_d = x_d.float(); y_d = y_d.float()

	out_height = output_size[0]
	out_width = output_size[1]

	max_y1 = float(height-1.0)
	max_x1 = float(width-1.0)

	x = (x+1.0)*width/2.0
	y = (y+1.0)*height/2.0

	x += x_d 
	y += y_d 
	# import pdb; pdb.set_trace()
	x = torch.clamp(x, min=0.000001, max=max_x1-0.00001)
	y = torch.clamp(y, min=0.000001, max=max_y1-0.00001)

	x0 = x.long()
	x1 = x0+1
	y0 = y.long()
	y1 = y0+1

	dim2 = width
	dim1 = width * height
	base = repeat(torch.arange(0, batch_size) * dim1, out_height * out_width).to(device)
	base_y0 = base + y0 * dim2
	base_y1 = base + y1 * dim2
	idx_a = base_y0 + x0
	idx_b = base_y1 + x0
	idx_c = base_y0 + x1
	idx_d = base_y1 + x1

	# import pdb; pdb.set_trace()

	im_flat = template.view(-1, num_channels).float()
	Ia = im_flat[idx_a.long()]
	Ib = im_flat[idx_b.long()]
	Ic = im_flat[idx_c.long()]
	Id = im_flat[idx_d.long()]

	x0_f = x0.float()
	x1_f = x1.float()
	y0_f = y0.float()
	y1_f = y1.float()
	wa = ((x1_f-x)*(y1_f-y)).unsqueeze(1)
	wb = ((x1_f-x)*(y-y0_f)).unsqueeze(1)
	wc = ((x-x0_f)*(y1_f-y)).unsqueeze(1)
	wd = ((x-x0_f)*(y-y0_f)).unsqueeze(1)

	output = wa*Ia + wb*Ib + wc*Ic + wd*Id 

	return output	


def image_warp_custom(template, deformation, device):
	batch_size, num_channels, height, width = template.shape[0], template.shape[1], template.shape[2], template.shape[3]
	deformation = deformation.view(-1, height*width, 2)
	deformation = deformation.permute(0,2,1)
	grid = make_grid(batch_size, height, width)

	x_s = grid[:, 0, :]
	y_s = grid[:, 1, :]
	x_s_flat = x_s.contiguous().view(-1)
	y_s_flat = y_s.contiguous().view(-1)

	x_d_s = deformation[:, 0, :]
	y_d_s = deformation[:, 1, :]
	x_d_s_flat = x_d_s.view(-1)
	y_d_s_flat = x_d_s.view(-1)

	img_transformed = interpolate(template, x_s_flat.to(device), y_s_flat.to(device), x_d_s_flat, y_d_s_flat,
		(height, width), device)

	return img_transformed.view(batch_size, num_channels, height, width)





