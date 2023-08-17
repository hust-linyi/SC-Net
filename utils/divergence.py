import torch
import torch.nn.functional as F

def loss_CT(M1, M2):
    criterion = torch.nn.KLDivLoss(reduction='mean').cuda()
    loss = criterion(F.log_softmax(M1, dim=1), M2)
    return loss

def Epsilon(M1, M2):
    results = 0
    M1_1 = F.softmax(M1, dim=1)
    for i in range(M2.shape[0]):
        m1 = M1_1[i] - M2[i]
        m2 = M1_1[i] + M2[i]
        result = 2 * (torch.abs(m1)).sum() / m2.sum()
        result = result.item()
        results += result
    return results