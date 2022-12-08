import torch
import torch.nn as nn
import torch.nn.functional as F

K = 0.5

def applyMask(scores):

    norm_scores = torch.sigmoid(scores) # 0~1

    mask = torch.zeros_like(norm_scores)
    mask[norm_scores>=K] = 1.0

    return mask

class SupermaskLinear(nn.Linear):
    def __init__(self,in_features, out_features):
        super().__init__(in_features, out_features,bias=False)

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.normal_(self.scores, std=0.01)

        # initialize the weights like this.
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu") # fixed
        # turn the gradient on the weights off 
        self.weight.requires_grad = False
        self.tmp_weight = torch.tensor(self.weight.tolist())
        self.weight = None # remove weight 

    def forward(self, x):
        subnet = applyMask(self.scores)  
        w = self.tmp_weight * subnet
        x = F.linear(x, w, self.bias)
        return x