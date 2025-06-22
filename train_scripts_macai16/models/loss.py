import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from . import basic as basic
from . import utils
import numpy as np

def smooth_loss(logit, is_logit=True):
    """
    logit: B, T, C
    """
    if is_logit:
        logsoft = F.log_softmax(logit, dim=2)
    else:
        logsoft = logit

    loss = torch.clamp((logsoft[:, 1:] - logsoft[:, :-1])**2, min=0, max=16)
    loss = loss.mean()
    return loss

def torch_class_label_to_segment_label(label):
    segment_label = torch.zeros_like(label)
    current = label[0]
    transcript = [label[0]]
    segment_start_ids = [0]
    segment_end_ids = []
    aid = 0
    for i, l in enumerate(label):
        if l == current:
            pass
        else:
            current = l
            aid += 1
            segment_end_ids.append(i-1)
            segment_start_ids.append(i)
            transcript.append(l)
        segment_label[i] = aid

    segment_end_ids.append(len(label) - 1)
    
    transcript = torch.LongTensor(transcript).to(label.device)
    segment_start_ids = torch.LongTensor(segment_start_ids).to(label.device)
    segment_end_ids = torch.LongTensor(segment_end_ids).to(label.device)
    
    transcript_seg = torch.stack([segment_start_ids,segment_end_ids], dim=1)

    
    return transcript, segment_label, transcript_seg

def logit2prob(clogit, dim=-1, class_sep=None):
    if class_sep is None or class_sep<=0:
        cprob = torch.softmax(clogit, dim=dim)
    else:
        assert dim==-1, dim
        cprob1 = torch.softmax(clogit[..., :class_sep], dim=dim)
        cprob2 = torch.softmax(clogit[..., class_sep:], dim=dim)
        cprob = torch.cat([cprob1, cprob2], dim=dim)
    
    return cprob

@torch.jit.script
def ctr_diou_loss_1d(
    input_offsets: torch.Tensor,
    target_offsets: torch.Tensor,
    reduction: str = 'none',
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Distance-IoU Loss (Zheng et. al)
    https://arxiv.org/abs/1911.08287

    This is an implementation that assumes a 1D event is represented using
    the same center point with different offsets, e.g.,
    (t1, t2) = (c - o_1, c + o_2) with o_i >= 0

    Reference code from
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/giou_loss.py

    Args:
        input/target_offsets (Tensor): 1D offsets of size (N, 2)
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        eps (float): small number to prevent division by zero
    """
    input_offsets = input_offsets.float()
    target_offsets = target_offsets.float()
    # check all 1D events are valid
    assert (input_offsets >= 0.0).all(), "predicted offsets must be non-negative"
    assert (target_offsets >= 0.0).all(), "GT offsets must be non-negative"

    lp, rp = input_offsets[:, 0], input_offsets[:, 1]
    lg, rg = target_offsets[:, 0], target_offsets[:, 1]

    # intersection key points
    lkis = torch.min(lp, lg)
    rkis = torch.min(rp, rg)

    # iou
    intsctk = rkis + lkis
    unionk = (lp + rp) + (lg + rg) - intsctk
    iouk = intsctk / unionk.clamp(min=eps)

    # smallest enclosing box
    lc = torch.max(lp, lg)
    rc = torch.max(rp, rg)
    len_c = lc + rc

    # offset between centers
    rho = 0.5 * (rp - lp - rg + lg)

    # diou
    loss = 1.0 - iouk + torch.square(rho / len_c.clamp(min=eps))

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

class MatchCriterion():

    def __init__(self, cfg, nclasses, bg_ids=[], class_weight=None):
        self.cfg = cfg
        self.nclasses = nclasses
        self.bg_ids = bg_ids
        self._class_weight=class_weight


    def set_label(self, label):
        self.class_label = label
        self.transcript, self.seg_label, self.transcript_seg = torch_class_label_to_segment_label(label) # transcript 每段的标签； seg_label 每帧属第几段的id
        self.onehot_class_label = self._label_to_onehot(self.class_label, self.nclasses)
        self.onehot_seg_label = self._label_to_onehot(self.seg_label, len(self.transcript))
        
        # create class weight
        cweight = torch.ones(self.nclasses+1).to(label.device)
        cweight[-1] = self.cfg.Loss.nullw
        if self._class_weight is not None:
            for i in range(self.nclasses):
                cweight[i] = self._class_weight[i]
        else:
            for i in self.bg_ids:
                cweight[i] = self.cfg.Loss.bgw

        # create weight for each phase segment based on class weight
        sweight = torch.ones_like(self.transcript, dtype=torch.float32)
        if self._class_weight is not None:
            for i, t in enumerate(self.transcript.tolist()):
                sweight[i] = self._class_weight[t]
        else:
            for i in self.bg_ids:
                sweight[self.transcript==i] = self.cfg.Loss.bgw

        self.cweight=cweight
        self.sweight=sweight

    def _label_to_onehot(self, label, nclass):
        onehot_label = torch.zeros(len(label), nclass).to(label.device)
        onehot_label[torch.arange(len(label)), label] = 1
        return onehot_label

    @classmethod
    def p2f_soft_iou(self, p2f_attn, onehot_seg_label):
        """
        p2f_attn: 1, f, a, sum over a == 1
        onehot_seg_label: f, s
        """
        p2f_attn = p2f_attn[0].unsqueeze(-1) # 1, f, a -> f, a, 1
        onehot_seg_label = onehot_seg_label.unsqueeze(1) # f, s -> f, 1, s
        p2f_attn_np = utils.to_numpy(p2f_attn)
        onehot_seg_label_np = utils.to_numpy(onehot_seg_label)
        overlap = np.einsum('tax,txs->as', p2f_attn_np, onehot_seg_label_np)
        union = np.minimum(p2f_attn_np + onehot_seg_label_np, 1.0).sum(0) # a,s 
        iou = np.nan_to_num(overlap / union, nan=0.0)

        del p2f_attn_np, onehot_seg_label_np
        return iou

    def match(self, clogit, p2f_attn):
        """
        clogit: a, 1, c  
        f2p_attn: 1, a, f
        p2f_attn: 1, f, a
        """
        assert clogit.shape[1] == 1 # batch_size == 1

        match_cfg = self.cfg.Loss
        transcript = self.transcript
        onehot_seg_label = self.onehot_seg_label

        # sequential matching between tokens and groundtruth segments
        if match_cfg.match == 'seq':
            A = clogit.shape[0]
            S = onehot_seg_label.shape[-1]
            assert A >= S, (A, S)
            phase_ind = seg_ind = torch.as_tensor(list(range(S)), dtype=torch.int64)
            return phase_ind, seg_ind

        # compute matching cost 
        cost = 0
        with torch.no_grad():
            if match_cfg.pc > 0:
                prob = clogit.squeeze(1)
                prob = torch.index_select(prob, 1, transcript) # a, s
                prob = utils.to_numpy(prob)
                cost -= match_cfg.pc * prob
            
            if match_cfg.p2fc > 0:
                p2f_iou = self.p2f_soft_iou(p2f_attn, onehot_seg_label)
                p2f_iou = utils.to_numpy(p2f_iou)
                cost -= match_cfg.p2fc * p2f_iou

        cost = utils.to_numpy(cost) # a, s

        # find optimal matching
        if match_cfg.match == 'o2o': # one-to-one matching
            phase_ind, seg_ind = linear_sum_assignment(cost)
        elif match_cfg.match == 'o2m': # one-to-many matching
            phase_ind, seg_ind = self._one_to_many_match(cost)

        phase_ind = torch.as_tensor(phase_ind, dtype=torch.int64) # id of phase query token
        seg_ind    = torch.as_tensor(seg_ind, dtype=torch.int64) # groundtruth phase label id

        return phase_ind, seg_ind

    def _one_to_many_match(self, cost):
        transcript_np = utils.to_numpy(self.transcript)
        phases = np.unique(transcript_np)
        token2phase_cost = []
        for a in phases:
            where = (transcript_np == a)
            score = cost[:, where]
            score = score.sum(1)
            token2phase_cost.append(score)
        token2phase_cost = np.stack(token2phase_cost, axis=1)

        _aid, _cid = linear_sum_assignment(token2phase_cost)
        
        unassign_aid = [ a for a in range(cost.shape[0]) if a not in _aid ] 
        unassign_cid = token2phase_cost[unassign_aid].argmin(1)

        all_aid = np.array(_aid.tolist() + unassign_aid)
        all_cid = [ phases[i] for i in _cid.tolist() + unassign_cid.tolist() ]
        all_cid = np.array(all_cid)

        atoken_cid = np.zeros(cost.shape[0])
        atoken_cid[all_aid] = all_cid

        match = {}
        for a in phases:
            seg_where = np.where(transcript_np == a)[0]
            token_where = np.where(atoken_cid == a)[0]
            subset = cost[token_where][:, seg_where]
            assign = subset.argmin(0)

            for s, a in zip(seg_where, assign):
                match[s] = token_where[a]
                
        aid_new, sid_new = [], []
        for k, v in match.items():
            aid_new.append(v)
            sid_new.append(k)
        
        return aid_new, sid_new

    def phase_token_loss(self, match, phase_clogit, is_logit=True):
        aind, sind = match
        A, C = phase_clogit.shape[0], phase_clogit.shape[-1]

        # phase prediction loss
        clabel = torch.zeros(A).to(phase_clogit.device).long() + C - 1 # shape: a; default = empty_class
        
        clabel[aind] = self.transcript[sind]

        if is_logit:
            loss = F.cross_entropy(phase_clogit.squeeze(1), clabel, weight=self.cweight)
        else:
            loss = F.nll_loss(phase_clogit.squeeze(1), clabel, weight=self.cweight)
        
        return loss

    def phase_seg_loss(self, match, phase_seg):
        aind, sind = match
        
        gt_segs = self.transcript_seg[sind]
        pre_segs = phase_seg[aind]

        reg_loss = ctr_diou_loss_1d(pre_segs, gt_segs, reduction='sum')

        loss_normalizer = 0.9 * 100 + 0.1 * len(self.class_label)
        # reg_loss = reg_loss / loss_normalizer
        
        return reg_loss

    def cross_attn_loss(self, match, attn, dim=None):
        assert dim >= 1
        onehot_seg_label = self.onehot_seg_label # f, s
        aind, sind = match
        
        frame_tgt = onehot_seg_label[:, sind] # f, s
        attn = attn[0, :, aind] # f, s

        attn_logp = torch.log_softmax(attn, dim=dim-1)
        loss2 = - attn_logp * frame_tgt 
        # if self.sweight is not None:
        #     loss2 = loss2 * self.sweight
        loss2 = loss2.sum(1).sum() / self.onehot_seg_label.sum()

        return loss2

    def cross_attn_loss_tdu(self, match, attn, tdu: basic.TemporalDownsampleUpsample, dim=None):
        assert dim >= 1
        onehot_seg_label = self.onehot_seg_label
        aind, sind = match

        # f, c -> s, c
        zoomed_label = torch.zeros([tdu.num_seg, onehot_seg_label.shape[1]], dtype=onehot_seg_label.dtype).to(onehot_seg_label.device) 
        zoomed_label.index_add_(0, tdu.seg_label, onehot_seg_label)
        zoomed_label = zoomed_label / tdu.seg_lens[:, None]

        frame_tgt = zoomed_label[:, sind] # s, n
        attn = attn[0, :, aind] # s, n
        attn_logp = torch.log_softmax(attn, dim=dim-1)
        
        loss2 = - attn_logp * frame_tgt 

        # if self.sweight is not None:
        #     loss2 = loss2 * self.sweight

        loss2 = loss2.sum(1).sum() / zoomed_label.sum()

        return loss2

    def frame_loss(self, frame_clogit, is_logit=True):
        if is_logit:
            logp = torch.log_softmax(frame_clogit, dim=-1)
        else:
            logp = frame_clogit
        
        cweight = self.cweight[:frame_clogit.shape[-1]] # remove the weight for null class
        frame_loss = - logp * self.onehot_class_label
        frame_loss = frame_loss * cweight

        frame_loss = frame_loss.sum(-1).sum() / self.onehot_class_label.sum()

        return frame_loss

    def frame_loss_tdu(self, seg_clogit, tdu, is_logit=True):
        if is_logit:
            logp = torch.log_softmax(seg_clogit.squeeze(1), dim=-1)
        else:
            logp = seg_clogit.squeeze(1)


        ohl = self.onehot_class_label
        zoomed_label = torch.zeros([tdu.num_seg, ohl.shape[1]], dtype=ohl.dtype).to(ohl.device) 
        zoomed_label.index_add_(0, tdu.seg_label, ohl)
        zoomed_label = zoomed_label / tdu.seg_lens[:, None]
        seg_loss = ( - logp * zoomed_label )
        _cweight = self.cweight[:logp.shape[-1]] # remove the weight for null class
        seg_loss = (seg_loss * _cweight)

        seg_loss = seg_loss.sum(-1).sum() / zoomed_label.sum()

        return seg_loss
    