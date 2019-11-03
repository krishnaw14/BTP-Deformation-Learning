import torch
import torch.nn.functional as F

# def make_grid(batch_size, height, width):
# 	x_t, y_t = torch.meshgrid(torch.linspace(-1.0, 1.0, width), torch.linspace(-1.0, 1.0, height))
# 	ones = torch.ones(torch.prod(x_t.shape))
# 	grid = torch.vstack([x_t.flatten, y_t.flatten, ones])
# 	return grid

# def interpolate(template, x, y, x_d, y_d, output_size):

# 	batch_size, num_channels, height, width = template.shape[0], template.shape[1], template.shape[2], template.shape[3]

# 	x = (x+1.0)*width/2.0
# 	y = (y+1.0)*height/2.0

# 	x += x_d 
# 	y += y_d 
# 	x = torch.clip(x, 0.000001, width-0.000001)
# 	y = torch.clip(y, 0.000001, height-0.000001)

# 	x0 = x.int()
# 	x1 = x0+1
# 	y0 = y.int()
# 	y1 = y0+1

# 	dim2 = width
# 	dim1 = width * height
# 	base = _repeat(tf.range(num_batch) * dim1, out_height * out_width)
# 	base_y0 = base + y0 * dim2
# 	base_y1 = base + y1 * dim2
# 	idx_a = base_y0 + x0
# 	idx_b = base_y1 + x0
# 	idx_c = base_y0 + x1
# 	idx_d = base_y1 + x1





	
# 	return img_transformed

# def image_warp(template, deformation):
# 	batch_size, num_channels, height, width = template.shape[0], template.shape[1], template.shape[2], template.shape[3]
# 	deformation = deformation.view(-1, height*width, 2)
# 	deformation = deformation.transpose(0,2,1)
# 	grid = make_grid(batch_size, height, width)

# 	x_s = something
# 	y_s = something
# 	x_s_flat = x_s.view(-1)
# 	y_s_flat = x_s.view(-1)

# 	x_d_s = something
# 	y_d_s = something
# 	x_d_s_flat = x_d_s.view(-1)
# 	y_d_s_flat = x_d_s.view(-1)

# 	img_transformed = interpolate(template, x_s_flat, y_s_flat, xd_s_flat, yd_s_flat, 
# 		(height, width))

# 	return img_transformed.view(batch_size, num_channels, height, width)

def image_warp(template, deformation):
	# import pdb; pdb.set_trace()
	img_transform = F.grid_sample(template, deformation.permute(0,2,3,1))
	return img_transform



