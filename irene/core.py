# Code from :      Information Removal at the Bottleneck of Neural Networks
# ==========================================================================


import torch
import torch.nn.functional as F
import numpy as np
from irene.utilities import Hook

class Privacy_head(torch.nn.Module):
	def __init__(self, bottleneck_layer, head_structure):
		super(Privacy_head, self).__init__()
		self.bottleneck = Hook(bottleneck_layer, backward=False)
		self.classifier = head_structure
	def forward(self):
		x = self.bottleneck.output.clone().detach()
		if len(x.size())>2:
			x = x.view(-1, np.prod((x.size())[1:]))
		x = self.classifier(x)
		return x
	def forward_attached(self):
		x = self.bottleneck.output
		if len(x.size())>2:
			x = x.view(-1, np.prod((x.size())[1:]))
		x = self.classifier(x)
		return x


class MI(torch.nn.Module):
	def __init__(self, privates = 10, device = "cpu"):
		super(MI, self).__init__()
		self.device = device
		self.privates = privates
		self.scaling = 1 / np.log(privates)

	def forward(self, private_head,yb_ethic):
		out_bias = private_head.forward_attached()
		GT =  1.0* torch.nn.functional.one_hot(yb_ethic, num_classes=self.privates)
		prob_bias = torch.nn.functional.softmax(out_bias, dim=1)
		joint = torch.clamp(torch.mm(torch.transpose(GT, 0, 1), prob_bias), min=1e-15)/ len(yb_ethic)
		marginal_bias = torch.sum(joint, dim=0, keepdim=True)
		marginal_GT = torch.sum(joint, dim=1, keepdim=True)
		marginals = torch.clamp(torch.mm(marginal_GT, marginal_bias), min=1e-15)
		return torch.sum(joint * torch.log(joint /marginals) * self.scaling)
