import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoupler_head(nn.Module):
    def __init__(self, dim_f, num_f_center, num_classes):
        super().__init__()
        self.dim_f = dim_f
        self.num_f_center = num_f_center
        self.decoup_metrixs = nn.Parameter(torch.randn(dim_f, dim_f*num_f_center))
        self.classifier = nn.Parameter(torch.randn(dim_f, num_classes))

    def forward(self, x):
        # x = F.normalize(x)
        x = torch.mm(x, self.decoup_metrixs)
        x = x.view(-1, self.num_f_center, self.dim_f)
        classifier = self.classifier.unsqueeze(0).repeat(x.shape[0], 1, 1)
        x = torch.bmm(F.normalize(x, dim=-1), F.normalize(classifier, dim=1))
        return x
