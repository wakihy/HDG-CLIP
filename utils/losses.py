import torch
import torch.nn as nn

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()



nINF = -100

class TwoWayLoss(nn.Module):
    def __init__(self, Tp=4., Tn=1.):
        super(TwoWayLoss, self).__init__()
        self.Tp = Tp
        self.Tn = Tn

    def forward(self, x, y):
        class_mask = (y > 0).any(dim=0)
        sample_mask = (y > 0).any(dim=1)

        # Calculate hard positive/negative logits
        pmask = y.masked_fill(y <= 0, nINF).masked_fill(y > 0, float(0.0))
        plogit_class = torch.logsumexp(-x/self.Tp + pmask, dim=0).mul(self.Tp)[class_mask]
        plogit_sample = torch.logsumexp(-x/self.Tp + pmask, dim=1).mul(self.Tp)[sample_mask]
    
        nmask = y.masked_fill(y != 0, nINF).masked_fill(y == 0, float(0.0))
        nlogit_class = torch.logsumexp(x/self.Tn + nmask, dim=0).mul(self.Tn)[class_mask]
        nlogit_sample = torch.logsumexp(x/self.Tn + nmask, dim=1).mul(self.Tn)[sample_mask]

        return torch.nn.functional.softplus(nlogit_class + plogit_class).mean() + \
                torch.nn.functional.softplus(nlogit_sample + plogit_sample).mean()




def create_loss(loss_fc):
    if loss_fc=='mlsm':    
        criterion = nn.MultiLabelSoftMarginLoss(reduction='sum')
    elif loss_fc == 'bce':
        #criterion = nn.BCEWithLogitsLoss(reduction='sum') 
        criterion = AsymmetricLoss(gamma_neg=0, gamma_pos=0, clip=0, disable_torch_grad_focal_loss=True)
    elif loss_fc == 'focal':
        criterion = AsymmetricLoss(gamma_neg=1, gamma_pos=1, clip=0, disable_torch_grad_focal_loss=True)
    elif loss_fc == 'asl':
        criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    elif loss_fc == 'halfasl':
        criterion = AsymmetricLoss(gamma_neg=1, gamma_pos=0, clip=0, disable_torch_grad_focal_loss=True)
    elif loss_fc == 'TwoWayLoss':
        criterion = TwoWayLoss(Tp=4, Tn=1)
    else:
        raise ValueError('loss not implemented')

    return criterion


from sklearn import metrics
import numpy as np
def evaluation(all_targets,all_predictions):
    # all_targets[all_targets==-1]=0
    # all_predictions = torch.sigmoid(all_predictions)
    # all_targets = all_targets.cpu().numpy()
    # all_predictions = all_predictions.cpu().numpy()
    meanAP = metrics.average_precision_score(all_targets,all_predictions, average='micro', pos_label=1)
    meanAP1 = metrics.average_precision_score(all_targets,all_predictions, average='macro', pos_label=1)
    meanAP2= metrics.average_precision_score(all_targets,all_predictions, average=None, pos_label=1)

    optimal_threshold = 0.5


    top_3rd = np.sort(all_predictions)[:,-3].reshape(-1,1)
    all_predictions_top3 = all_predictions.copy()
    all_predictions_top3[all_predictions_top3<top_3rd] = 0
    all_predictions_top3[all_predictions_top3<optimal_threshold] = 0
    all_predictions_top3[all_predictions_top3>=optimal_threshold] = 1

    CP_top3 = metrics.precision_score(all_targets, all_predictions_top3, average='macro')
    CR_top3 = metrics.recall_score(all_targets, all_predictions_top3, average='macro')
    CF1_top3 = (2*CP_top3*CR_top3)/(CP_top3+CR_top3)
    OP_top3 = metrics.precision_score(all_targets, all_predictions_top3, average='micro')
    OR_top3 = metrics.recall_score(all_targets, all_predictions_top3, average='micro')
    OF1_top3 = (2*OP_top3*OR_top3)/(OP_top3+OR_top3)

    all_predictions_thresh = all_predictions.copy()
    all_predictions_thresh[all_predictions_thresh < optimal_threshold] = 0
    all_predictions_thresh[all_predictions_thresh >= optimal_threshold] = 1
    CP = metrics.precision_score(all_targets, all_predictions_thresh, average='macro')
    CR = metrics.recall_score(all_targets, all_predictions_thresh, average='macro')
    CF1 = (2*CP*CR)/(CP+CR)
    OP = metrics.precision_score(all_targets, all_predictions_thresh, average='micro')
    OR = metrics.recall_score(all_targets, all_predictions_thresh, average='micro')
    OF1 = (2*OP*OR)/(OP+OR)

    print('meanAP-micro:', round(meanAP,6)*100)
    print('meanAP-macro',round(meanAP1,6)*100)
    print('meanAP-None',meanAP2)
    #print('000000000000000000000000000000')
    #for j in meanAP2:
    #    for i in range(10):
    #        print(round(j,2),end=",")
    #    print()
    #print('000000000000000000000000000000')
    #print("--------------macro---------------")
    #print("top3的CP,CR,CF1：", round(CP_top3, 6) * 100, round(CR_top3, 6) * 100, round(CF1_top3, 6) * 100)
    #print("--------------micro---------------")
    #print("top3的OP,OR,OF1：", round(OP_top3, 6) * 100, round(OR_top3, 6) * 100, round(OF1_top3, 6) * 100)
    print("**************macro***************")
    print("CP,CR,CF1:", round(CP, 6) * 100, round(CR, 6) * 100, round(CF1, 6) * 100)
    print("**************micro***************")
    print("OP,OR,OF1:", round(OP, 6) * 100, round(OR, 6) * 100, round(OF1, 6) * 100)
    return meanAP,meanAP1,CP_top3,CR_top3,CF1_top3,OP_top3,OR_top3,OF1_top3,CP,CR,CF1,OP,OR,OF1