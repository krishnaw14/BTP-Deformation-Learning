# model
img_size = 128
z_dim = 32
conv_layers = 4
channels = [40, 20, 10, 3]
kernel_sizes = [3, 3, 5, 5]
strides = [2, 2, 2, 2]
pads = [0, 0, 1, 1]

#dir
save_param_dir = "saved_models/"
save_results_dir = "reconstructions/"

#train
num_epochs = 50
lr = 1e-3
sigma=100
log_step = 1
val_step = 1
save_param_step = 1

#data
data_path = "data/"
batch_size = 32


