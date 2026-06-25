import torch
import numpy as np
import random
import os
import codecs
import logging
import builtins
import datetime

import torch.distributed as dist
import os
from sklearn.metrics import f1_score, precision_score, recall_score
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def compute_AP(predictions, labels):
    num_class = predictions.size(1)
    ap = torch.zeros(num_class).to(predictions.device)
    empty_class = 0
    for idx_cls in range(num_class):
        prediction = predictions[:, idx_cls]
        label = labels[:, idx_cls]
        mask = label.abs() == 1
        if (label > 0).sum() == 0:
            empty_class += 1
            continue
        binary_label = torch.clamp(label[mask], min=0, max=1)
        sorted_pred, sort_idx = prediction[mask].sort(descending=True)
        sorted_label = binary_label[sort_idx]
        tmp = (sorted_label == 1).float()
        tp = tmp.cumsum(0)
        fp = (sorted_label != 1).float().cumsum(0)
        num_pos = binary_label.sum()
        rec = tp/num_pos
        prec = tp/(tp+fp)
        ap_cls = (tmp*prec).sum()/num_pos
        ap[idx_cls].copy_(ap_cls)
    return ap

def compute_F1(predictions, labels, mode_F1, k_val):
    idx = predictions.topk(dim=1, k=k_val)[1]
    predictions.fill_(0)
    predictions.scatter_(dim=1, index=idx, src=torch.ones(predictions.size(0), k_val).to(predictions.device))
    mask = predictions == 1
    TP = (labels[mask] == 1).sum().float()
    tpfp = mask.sum().float()
    tpfn = (labels == 1).sum().float()
    p = TP / tpfp
    r = TP/tpfn
    f1 = 2*p*r/(p+r)

    return f1,p,r

def one_hot_to_class_labels(one_hot_array):
    samples = []
    if isinstance(one_hot_array, np.ndarray):
        for i, s in enumerate(one_hot_array):
            idx_hot = np.where(s)[0]
            samples.append(list(idx_hot))
    return samples


def compute_F100(idxs,gt_labels, k, num_classes=None):
    TP = np.zeros(num_classes)
    FP = np.zeros(num_classes)
    class_samples = np.zeros(num_classes)
    gt_labels = one_hot_to_class_labels(gt_labels)

    num_samples = len(gt_labels)

    for i in range(num_samples):
        gt_label = gt_labels[i]
        if isinstance(gt_label, list):
            tps = [elem in idxs[i][:k] for elem in gt_label]
            for j in range(len(gt_label)):
                TP[gt_label[j]] += tps[j]
                class_samples[gt_label[j]] += 1
            fps = [elem not in gt_label for elem in idxs[i][:k]]

            for j in range(k):
                if j < FP.shape[0]:
                    FP[idxs[i][j]] += fps[j]
        else:
            raise NotImplementedError

    TP_s = np.nansum(TP)
    FP_s = np.nansum(FP)
    precision_o = TP_s / (TP_s + FP_s)

    class_samples_s = np.nansum(class_samples)
    recall_o = TP_s / class_samples_s

    if precision_o == 0 or recall_o == 0:  # avoid nan if both zero
        F1_o = 0
    else:
        F1_o = 2 * precision_o * recall_o / (precision_o + recall_o)

    return F1_o,precision_o, recall_o


def compute_F10(predictions, labels, mode_F1,k_val):

    labels = labels.cpu().numpy()
    predictions = predictions.cpu().numpy()

    mask = np.sum(labels == 1, 1) > 0
    #print("Total test samples: {} Total samples with positive labels: {}".format(predictions.shape[0], np.sum(mask)))
    predictions = predictions[mask]
    labels = labels[mask]

    idx = np.argsort(predictions, axis=1)
    for i in range(predictions.shape[0]):
        predictions[i][idx[i][-k_val:]] = 1
        predictions[i][idx[i][:-k_val]] = 0
    mask = np.sum(labels == 1, 1) > 0
    #print("Total test samples: {} Total samples with positive labels: {}".format(predictions.shape[0], np.sum(mask)))
    predictions = predictions[mask]
    labels = labels[mask]
    if mode_F1 == 'overall':
        #print('evaluation overall!! cannot decompose into classes F1 score')
        mask = predictions == 1
        TP = np.sum(labels[mask] == 1)
        p = TP / np.sum(mask)
        r = TP / np.sum(labels == 1)
        f1 = 2 * p * r / (p + r)

    else:
        num_class = predictions.shape[1]
        print('evaluation per classes')
        f1 = np.zeros(num_class)
        p = np.zeros(num_class)
        r = np.zeros(num_class)
        for idx_cls in range(num_class):
            prediction = np.squeeze(predictions[:, idx_cls])
            label = np.squeeze(labels[:, idx_cls])
            if np.sum(label > 0) == 0:
                continue
            binary_label = np.clip(label, 0, 1)
            f1[idx_cls] = f1_score(binary_label, prediction)  # AP(prediction,label,names)
            p[idx_cls] = precision_score(binary_label, prediction)
            r[idx_cls] = recall_score(binary_label, prediction)
    return torch.tensor(f1), torch.tensor(p), torch.tensor(r)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed) 
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def init_log(args, record_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_path = os.path.join(record_path, 'recording.log')

    fh = logging.FileHandler(log_path, mode='w') 
    fh.setLevel(logging.DEBUG)  
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()

def convert_models_to_half(model):
    for p in model.parameters():
        p.data = p.data.half()
        if p.grad:
            p.grad.data = p.grad.data.half()

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0

def init_distributed_mode(args):

    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.gpu = int(os.environ['LOCAL_RANK'])
    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print

def write_description_to_folder(file_name, config):
    with codecs.open(file_name, 'w') as desc_f:
        desc_f.write("- Training Parameters: \n")
        for key, value in config.__dict__.items():
            desc_f.write("  - {}: {}\n".format(key, value))
