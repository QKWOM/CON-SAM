import torch
import torch.nn as nn

class BoundaryDoULoss(nn.Module):
    def __init__(self):
        super(BoundaryDoULoss, self).__init__()

    def _adaptive_size(self, score, target):
        kernel = torch.Tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        kernel = kernel.float()
        padding_out = torch.zeros((target.shape[0], target.shape[-2] + 2, target.shape[-1] + 2))
        padding_out[:, 1:-1, 1:-1] = target.squeeze(1)
        h, w = 3, 3
        Y = torch.zeros((padding_out.shape[0],padding_out.shape[1]-h+1,padding_out.shape[2]-w+1)).cuda(1)
        for i in range(Y.shape[0]):
            Y[i,:,:] = torch.conv2d(target[i].unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0).cuda(1), padding=1)
        # Y = torch.conv2d(target.unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0).cuda(), padding=1)
        Y = Y * target
        Y[Y == 5] = 0
        C = torch.count_nonzero(Y)
        S = torch.count_nonzero(target)
        smooth = 1e-5
        alpha = 1 - (C + smooth) / (S + smooth)
        alpha = 2 * alpha - 1

        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        alpha = min(alpha, 0.8)
        loss = (z_sum + y_sum - 2 * intersect + smooth) / (z_sum + y_sum - (1 + alpha) * intersect + smooth)

        return loss

    def forward(self, inputs, target):
        inputs = torch.sigmoid(inputs)
        target = target.float()  # Assuming target is a tensor of 0s and 1s
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        loss = self._adaptive_size(inputs, target)
        return loss
