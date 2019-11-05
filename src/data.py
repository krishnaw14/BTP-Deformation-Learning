from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

class LimitImageFolder(Dataset):
	def __init__(self, dataset, n):
		self.n = n
		self.dataset = dataset
	def __len__(self):
		return self.n
	def __getitem__(self, i):
		return self.dataset[i]

def get_image_folder_data_loader(config):
	data_path = config.data_path

	dataset = ImageFolder(root=data_path, transform=transforms.Compose([
		transforms.CenterCrop(96),
		transforms.Resize((config.img_size, config.img_size)),
		transforms.ToTensor(),
		transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
		])
	)
	limit_dataset = LimitImageFolder(dataset, 10000)

	# import pdb; pdb.set_trace()

	data_loader = DataLoader(limit_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8)
	return data_loader
