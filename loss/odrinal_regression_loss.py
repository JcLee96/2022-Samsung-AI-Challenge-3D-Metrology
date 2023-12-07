import torch.nn as nn
import torch

class BaseLoss(nn.Module):

    def __init__(self, device=torch.device("cuda")):
        super(BaseLoss, self).__init__()
        self.loss = 0.0
        self.set_device(device)

    def set_device(self, device):
        self.device = device

    def __repr__(self):
        return self.__class__.__name__

class OrdinalRegressionLoss(BaseLoss):
    """
    Ordinal loss is defined as the average of pixelwise ordinal loss F(h, w, X, O)
    over the entire image domain:
    """

    def __init__(self, device=torch.device("cuda")):
        super(OrdinalRegressionLoss, self).__init__()
        self.set_device(device)
        self.loss = 0.0

    #def forward(self, ord_labels, target, weight_mask ,mask=0):
    def forward(self, ord_labels, target):
        """
        :param ord_labels: ordinal labels for each position of Image I.
        :param target:     the ground_truth discreted using SID strategy.
        :return: ordinal loss
        """
        # assert pred.dim() == target.dim()
        # invalid_mask = target < 0
        # target[invalid_mask] = 0
        ord_labels = ord_labels.to(self.device)
        target = target.to(self.device)
        #mask = mask.to(self.device)
        N, C, H, W = ord_labels.size()
        ord_num = C
        self.loss = 0.0

        K = torch.zeros((N, C, H, W), dtype=torch.int).to(self.device)
        for i in range(ord_num):
            K[:, i, :, :] = K[:, i, :, :] + i * torch.ones((N, H, W), dtype=torch.int).to(self.device)

        mask_0 = (K <= target).detach()
        mask_1 = (K > target).detach()

        one = torch.ones(ord_labels[mask_1].size()).to(self.device)

        self.loss += torch.sum(torch.log(torch.clamp(ord_labels[mask_0], min=1e-8, max=1e8))) \
                     + torch.sum(torch.log(torch.clamp(one - ord_labels[mask_1], min=1e-8, max=1e8)))

        N = N * H * W
        self.loss /= (-N)  # negative
        return self.loss