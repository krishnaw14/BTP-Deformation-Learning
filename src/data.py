from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

def get_image_folder_data_loader(config):
	data_path = config.data_path

	dataset = ImageFolder(root=data_path, transform=transforms.Compose([
		transforms.CenterCrop(96),
		transforms.Resize(config.img_size),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
		])
	)

	data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=8)
	return data_loader
