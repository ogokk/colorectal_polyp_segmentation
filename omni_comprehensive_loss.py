
import torch.nn as nn
import numpy as np
from torch.nn import functional as F


def ncc(inputs, targets):
    c = (inputs - inputs.mean()) / (inputs.std() * len(inputs))
    d = (targets - targets.mean()) / (targets.std())
    c = c.cpu().detach().numpy()
    d = d.cpu().detach().numpy()
    ncc = np.correlate(c, d, 'valid')
    # print(ncc)
    return ncc.mean()


def TverskyIndex(inputs, targets, smooth=0.1, beta=0.1):        
    #True Positives, False Positives & False Negatives
    TP = (inputs * targets).sum()    
    FP = ((1-targets) * inputs).sum()
    FN = (targets * (1-inputs)).sum()
   
    Tversky = (TP) / (TP + beta*FP + (1-beta)*FN + smooth)  
    
    return Tversky.mean()

    
    
class omni_comprehensive_loss(nn.Module):
    def __init__(self):
        super(omni_comprehensive_loss, self).__init__()
        

    def forward(self, inputs, targets):
        alpha = 0.5
        # sigmoid activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        omni_comprehensive = (1 - (alpha*ncc(inputs,targets)+\
                        (1-alpha)*TverskyIndex(inputs, targets))) * BCE

        return omni_comprehensive
