import numpy as np 
import torch.nn as nn 
import torch.nn.functional as F 
from sklearn.metrics import precision_recall_curve, auc, f1_score, accuracy_score
import torch


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.5, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )

class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        """class to calculate Label-Distribution-Aware Margin Loss

        Args:
            cls_num_list (_type_): _description_
            max_m (float, optional): _description_. Defaults to 0.5.
            weight (_type_, optional): _description_. Defaults to None.
            s (int, optional): _description_. Defaults to 30.
        """
        super().__init__()
        m_list = 1.0 / (torch.sqrt(torch.sqrt(cls_num_list))+ 1e-7)
        m_list = m_list * (max_m / torch.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.bool)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=30, smoothing=0.05, dim=-1):
        """class to calculate label smoothing loss

        Args:
            classes (int, optional): Defaults to 30.
            smoothing (float, optional): highly recommend to use between 0~0.1
            dim (int, optional): Defaults to -1.
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))



def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = ["no_relation", "org:dissolved", "org:founded", "org:place_of_headquarters", 
                "org:alternate_names", "org:member_of", "org:members","org:political/religious_affiliation", 
                "org:product", "org:founded_by","org:top_members/employees", "org:number_of_employees/members", 
                "per:date_of_birth", "per:date_of_death", "per:place_of_birth", "per:place_of_death", 
                "per:place_of_residence", "per:origin","per:employee_of", "per:schools_attended", "per:alternate_names", 
                "per:parents", "per:children", "per:siblings", "per:spouse", "per:other_family", "per:colleagues", 
                "per:product", "per:religion", "per:title"]
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = precision_recall_curve(targets_c, preds_c)
        score[c] = auc(recall, precision)
    return np.average(score) * 100.0

def compute_metrics(pred):
  """ validation을 위한 metrics function """
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  probs = pred.predictions

  # calculate accuracy using sklearn's function
  f1 = klue_re_micro_f1(preds, labels)
  auprc = klue_re_auprc(probs, labels)
  acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.

  return {
      'micro f1 score': f1,
      'auprc' : auprc,
      'accuracy': acc,
  }
