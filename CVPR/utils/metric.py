import torch
import torch.nn.functional as F

def acc_multi_label(pred, label, index):
    ts = []
    accs = []
    for i in range(len(index)-1):
        pred_max_idx = pred[i].argmax(dim=1, keepdim=True)
        label_max_idx = label[:, index[i]:index[i+1]].argmax(dim=1, keepdim=True)
        t = pred_max_idx.eq(label_max_idx)
        ts.append(t)
        accs.append((t.sum()/label.shape[0])*100)
    ts = torch.cat(ts, dim=1).min(dim=1)[0].sum()
    acc = ts/label.shape[0]
    return acc*100, accs

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def cosine_similarity(x, y):
    x = F.normalize(x.float())
    y = F.normalize(y.float())
    cos_sim_metrix = torch.mm(x, y.T)
    mask = torch.eye(x.size(0), x.size(0)).bool()
    cos_sim = cos_sim_metrix[mask]
    return cos_sim.mean()