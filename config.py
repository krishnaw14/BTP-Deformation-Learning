# model
img_size = 64
sampling_step = 20
step_size = 0.1
geo_scale = 20
z_dim_app = 64
z_dim_geo = 64

conv_layers_app = 4
channels_app = [40, 20, 10, 3]
kernel_sizes_app = [4, 4, 4, 4]
strides_app = [2, 2, 2, 2]
pads_app = [1, 1, 1, 1]

conv_layers_geo = 4
channels_geo = [64, 32, 16, 2]
kernel_sizes_geo = [4, 4, 4, 4]
strides_geo = [2, 2, 2, 2]
pads_geo = [1, 1, 1, 1]

#dir
save_param_dir = "saved_models/"
save_results_dir = "reconstructions/"

#train
num_epochs = 2000
lr = 1e-5
sigma= 0.15
log_step = 5
val_step = 5
save_param_step = 5

#data
data_path = "data/"
batch_size = 128


