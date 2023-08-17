import torch.nn.functional as F
import torch


def Combine(prediction, target):
    predict = prediction
    pseudos = torch.zeros([prediction.shape[0], 2, prediction.shape[2], prediction.shape[3]]).cuda()
    for i in range(target.shape[0]):
        pseudo_label = predict[i].squeeze(0)
        pseudo_label[target[i] == 1] = 1.0  # nuclei
        pseudo_label[target[i] == 0] = 0.0   # background
        pseudo_0 = torch.ones((target[i].shape[0], target[i].shape[1])).cuda() - pseudo_label
        pseudo1 = pseudo_label.unsqueeze(0)
        pseudo0 = pseudo_0.unsqueeze(0)
        pseudo = torch.cat([pseudo0, pseudo1.float()], dim=0)
        pseudos[i] = pseudo
    return pseudos