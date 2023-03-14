import matplotlib.pyplot as plt
import seaborn as sns
from src.os import path
sns.set_style("darkgrid")

class Graphs(object):
	"""docstring for Graphs"""
	def __init__(self, arg):
		super(Graphs, self).__init__()
		self.arg = arg
		self.path = path.ROOT_DIR # root directory of the projects

	def drawLoss(self,data):
		folder_path = self.path+"src/results/carEvaluation/trainingloss.png"
		plt.plot(data)
		plt.xlabel("Epoch")
		plt.ylabel("loss")
		plt.savefig(folder_path,dpi=300)
		plt.close()
