from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

def get_data_loader(config):
	data_path = config.data_path

	dataset = ImageFolder(root=data_path, transform=transforms.Compose([
		transforms.CenterCrop(config.img_size),
		transforms.Normalize((0.5*255, 0.5*255, 0.5*255), (0.5*255, 0.5*255, 0.5*255)),
		transforms.ToTensor(),
		])
	)

	data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=8)
	return data_loader
