import torch 
import torch.nn as nn 




class Unet(nn.Module):
	def __init__(self):
		super().__init__()
		self.model = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True)
		self.model.outc = nn.Conv2d(64, 1, kernel_size=(1,1), stride = (1,1))

		#self.model.activatef = torch.nn.Sigmoid()

	def forward(self, x):
		return self.model(x)




