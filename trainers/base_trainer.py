import nnabla as nn 
import nnabla.funtions as F
import nnabla.solvers as S

class BaseTrainer(object):

	def __init__(self, config):
		self.config = config
		
		self.get_model()
		self.get_solver()
		self.get_dataset()

	def save_params(self, param_path):
		nn.save_parameters(os.path.join(param_path, 'param.h5'))
		self.solver.save_states(os.path.join(param_path, 'solver.h5'))

	def load_params(self, param_path):
		nn.load_parameters(os.path.join(param_path, 'param.h5'))
		self.solver.load_states(os.path.join(param_path, 'solver.h5'))

	def get_model(self):
		if config.model.name == 'GAN':
			from ..models.gan import GAN
			self.model = GAN(config)

	def get_optimizer(self):
		if config.solver.name == 'sgd':
			self.solver = S.Sgd(lr=config.solver.lr)
		elif config.solver.name == 'momentum':
			self.solver = S.Momentum(lr=config.solver.lr, config.solver.momentum)
		elif config.solver.name == 'adam':
			self.solver = S.Adam(alpha=self.solver.lr, beta1=self.solver.beta_1, beta2=self.solver.beta_2)

	def get_dataset(self):
		if config.model.name == 'MNIST':
			from ..datasets.mnist import get_mnist
			self.dataset = get_mnist()


