# model
img_size = 64
z_dim = 64
conv_layers = 4
channels = [40, 20, 10, 3]
kernel_sizes = [4, 4, 4, 4]
strides = [2, 2, 2, 2]
pads = [1, 1, 1, 1]

#dir
save_param_dir = "saved_models/"
save_results_dir = "reconstructions/"

#train
num_epochs = 50
lr = 1e-3
sigma= 0.15
log_step = 1
val_step = 1
save_param_step = 1

#data
data_path = "data/"
batch_size = 128


