import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from . import basic, utils, state
from configs.utils import update_from
from . import loss
from .loss import MatchCriterion

class DSS(nn.Module):
    """
    Dual-Scale Selector
    """
    def __init__(self, Nw, ntoken, f_dim, n_classes):
        super().__init__()
        self.Nw = Nw
        self.ntoken = ntoken
        half_Nw = Nw // 2
        self.half_Nw = half_Nw

        self.global_key_feature = basic.MSTCN2(f_dim, f_dim, n_classes, 1, dropout=0.0, in_map=False)
        self.local_key_feature = basic.MSTCN2(f_dim, f_dim, (self.Nw + 1), 1, dropout=0.0, in_map=False)

    def forward(self, frame_feature):
        """
        Args:
            frame_feature (Tensor): Frame-level features of shape (T, B, C).

        Returns:
            Tensor: Selected phase features of shape (ntoken, B, C), where K <= ntoken.
            Tensor: Top-k indices of shape (B, ntoken).
        """
        T, B, C = frame_feature.shape

        # Global Key Feature
        g_logits = self.global_key_feature(frame_feature)

        # Local Key Feature
        l_logits = self.local_key_feature(frame_feature)

        # Two-scale aggregation
        x = F.pad(g_logits.permute(1,2,0), (self.half_Nw, self.half_Nw), mode='constant', value=0).unsqueeze(-1)
        x_size = list(x.size())  # bz, n_classes, T+Nw, 1
        x_size[-1] = self.Nw + 1  # bz, n_classes, T+Nw, Nw + 1
        x_size[-2] = x_size[-2] - self.Nw  # bz, n_classes, T, Nw + 1
        x_stride = list(x.stride())
        x_stride[-2] = x_stride[-1]
        g_logits = x.as_strided(size=x_size, stride=x_stride).permute(0, 2, 1, 3)

        l_logits = l_logits.unsqueeze(-2).permute(1, 0, 2, 3)
        pred_scores = torch.softmax(g_logits + l_logits, dim=-1)
        pred_scores = pred_scores.masked_fill(torch.isnan(pred_scores), 0)

        padded_scores = F.pad(pred_scores, (0, 0, 0, 0, self.half_Nw, self.half_Nw), mode='constant', value=0)
        pred_scores_final = torch.stack([padded_scores[:, i:i+T, :, :] for i in range(self.Nw)], dim=0).sum(dim=0).sum(dim=-1)
        pred_scores_final_phase, _ = torch.max(pred_scores_final, dim=-1)

        k = min(self.ntoken, pred_scores_final_phase.shape[-1])
        top_k_values, top_k_indices = torch.topk(pred_scores_final_phase, k, dim=-1)
        
        gathered_frame_features = frame_feature[top_k_indices.squeeze(0)]

        return gathered_frame_features, top_k_indices # Return selected features and their indices


class B2Q_Net(nn.Module):

    def __init__(self, cfg, in_dim, n_classes, opts):
        super().__init__()
        self.opts = opts
        self.cfg = cfg
        self.num_classes = n_classes
        self.ntoken = cfg.B2Q_Net.ntoken
        self.Nw = cfg.B2Q_Net.Nw

        base_cfg = cfg.Bi
        self.frame_pe = basic.PositionalEncoding(base_cfg.hid_dim, max_len=10000, empty=(not cfg.B2Q_Net.fpos) )
        self.channel_masking_dropout = nn.Dropout2d(p=cfg.B2Q_Net.cmr)

        if not cfg.B2Q_Net.trans : # when video transcript is not available at training and inference
            self.phase_query = nn.Parameter(torch.randn([cfg.B2Q_Net.ntoken, 1, base_cfg.a_dim]))
            self.state_space = nn.Parameter(torch.randn([cfg.B2Q_Net.mtoken, 1, base_cfg.a_dim]))

        self.InputBlock_frame = InputBlock_frame(cfg, opts, in_dim, n_classes)
        self.dual_scale_selector = DSS(self.Nw, self.ntoken, base_cfg.f_dim, n_classes)

        # block configuration
        block_list = []
        block_list.append(self.InputBlock_frame)
        for i, t in enumerate(cfg.B2Q_Net.block):
            if t == 'i':
                block = InputBlock_phase(cfg, opts, in_dim, n_classes)
            elif t == 'u':
                update_from(cfg.Bu, base_cfg, inplace=True)
                base_cfg = cfg.Bu
                block = UpdateBlock(cfg, opts, n_classes)
            elif t == 'U':
                update_from(cfg.BU, base_cfg, inplace=True)
                base_cfg = cfg.BU
                block = UpdateBlockTDU(cfg, opts, n_classes)

            block_list.append(block)

        self.block_list = nn.ModuleList(block_list)

        
        self.SSQ = nn.ModuleList([copy.deepcopy(
            state.state_AttentionBlock(base_cfg.a_dim, base_cfg.a_dim, base_cfg.a_ffdim, base_cfg.a_nhead)) 
            for i in range(base_cfg.a_layers)])
        self.state_outpit = nn.Linear(base_cfg.a_dim, n_classes)
        self.mcriterion = None
        self.global_phase_pe = None
        self.global_phase_feature = None

    def _forward_one_video(self, seq, transcript=None):
        # prepare frame feature
        frame_feature = seq
        frame_pe = self.frame_pe(seq)
        if self.cfg.B2Q_Net.cmr:
            frame_feature = frame_feature.permute([1, 2, 0])
            frame_feature = self.channel_masking_dropout(frame_feature)
            frame_feature = frame_feature.permute([2, 0, 1])

        if self.cfg.TM.use and self.training:
            frame_feature = basic.time_mask(frame_feature, 
                        self.cfg.TM.t, self.cfg.TM.m, self.cfg.TM.p, 
                        replace_with_zero=True)

        frame_feature = self.InputBlock_frame(frame_feature, frame_pe)

        phase_feature, top_k_indices = self.dual_scale_selector(frame_feature)
        phase_pe = self.phase_query
        if phase_feature.size(0) < phase_pe.size(0):  
            padding_size = phase_pe.size(0) - phase_feature.size(0)
            padding = torch.zeros(padding_size, phase_feature.size(1), phase_feature.size(2), device=phase_feature.device)
            phase_feature = torch.cat((phase_feature, padding), dim=0)
        
        for state_layer in self.SSQ:
            Historical_feature, phase_feature = state_layer(self.global_phase_feature, phase_feature, self.global_phase_pe, phase_pe)

        if self.training:
            Historical_feature_output = self.state_outpit(Historical_feature)
            Historical_probs = F.softmax(Historical_feature_output, dim=-1)
            Historical_dist = Categorical(Historical_probs)
            Historical_entropy = Historical_dist.entropy()
            self.state_loss = Historical_entropy.mean()

        block_output = []
        for i, block in enumerate(self.block_list):
            if i == 0:
                continue
            frame_feature, phase_feature = block(frame_feature, phase_feature, frame_pe, phase_pe)
            block_output.append([frame_feature, phase_feature])

        self.global_phase_feature = Historical_feature.detach()

        return block_output

    def _loss_one_video(self, label):
        mcriterion: MatchCriterion = self.mcriterion
        mcriterion.set_label(label)

        block : Block = self.block_list[-1]
        cprob = basic.logit2prob(block.phase_clogit, dim=-1)
        match = mcriterion.match(cprob, block.p2f_attn)
        
        ######## per block loss
        loss_list = []
        for block in self.block_list:
            loss = block.compute_loss(mcriterion, match)
            loss_list.append(loss)
        loss_list.append(self.state_loss)
        sum_block_1 = loss_list[0] + loss_list[1] + loss_list[2]
        self.loss_list = [sum_block_1] + loss_list[3:]
        final_loss = sum(self.loss_list) / len(self.loss_list)
        return final_loss

    def forward(self, seq_list, label_list, compute_loss=False):
        save_list = []
        final_loss = []
        if not isinstance(seq_list, list):
            seq_list = [seq_list]
        if not isinstance(label_list, list):    
            label_list = [label_list]
        for i, (seq, label) in enumerate(zip(seq_list, label_list)):
            if seq.dim() == 2:
                seq = seq.unsqueeze(1)
            if label.dim() == 2:
                label = label.squeeze(0)

            trans = basic.torch_class_label_to_segment_label(label)[0]

            self._forward_one_video(seq, trans)

            pred, _, _ = self.block_list[-1].eval(trans)
            save_data = {'pred': utils.to_numpy(pred)}
            save_list.append(save_data)

            if compute_loss:
                loss = self._loss_one_video(label)
                final_loss.append(loss)
                save_data['loss'] = { 'loss': loss.item() }

        if compute_loss:
            final_loss = sum(final_loss) / len(final_loss)
            return final_loss, save_list
        else:
            return 0, save_list

    def save_model(self, fname):
        torch.save(self.state_dict(), fname)
    def reset(self):
        self.global_phase_pe = self.state_space # M, B(=1), H
        self.global_phase_feature = torch.zeros_like(self.global_phase_pe)


####################################################################
# Blocks

class Block(nn.Module):
    """
    Base Block class for common functions
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        lines = f"{type(self).__name__}(\n  f:{self.frame_branch},\n  a:{self.phase_branch},\n  p2f:{self.p2f_layer if hasattr(self, 'p2f_layer') else None},\n  f2p:{self.f2p_layer if hasattr(self, 'f2p_layer') else None}\n)"
        return lines

    def __repr__(self):
        return str(self)

    def process_feature(self, feature, nclass):
        # use the last several dimension as logit of phase classes
        clogit = feature[:, :, -nclass:] # class logit
        feature = feature[:, :, :-nclass] # feature without clogit
        cprob = basic.logit2prob(clogit, dim=-1)  # apply softmax
        feature = torch.cat([feature, cprob], dim=-1)

        return feature, clogit

    def create_fbranch(self, cfg, opts=None, in_dim=None, f_inmap=False):
        if in_dim is None:
            in_dim = cfg.f_dim

        if cfg.f == 'm': # use MSTCN
            frame_branch = basic.MSTCN(in_dim, cfg.f_dim, cfg.hid_dim, cfg.f_layers, 
                                dropout=cfg.dropout, ln=cfg.f_ln, ngroup=cfg.f_ngp, in_map=f_inmap)
        elif cfg.f == 'm2': # use MSTCN++
            frame_branch = basic.MSTCN2(in_dim, cfg.f_dim, cfg.hid_dim, cfg.f_layers, 
                                dropout=cfg.dropout, ln=cfg.f_ln, ngroup=cfg.f_ngp, in_map=f_inmap)
        elif cfg.f == 'LSTM':
            frame_branch = basic.LSTMHead(in_dim, cfg.f_dim, opts.seq_len)

        return frame_branch

    def create_abranch(self, cfg):
        if cfg.a == 'sa': # self-attention layers, for update blocks
            l = basic.SALayer(cfg.a_dim, cfg.a_nhead, dim_feedforward=cfg.a_ffdim, dropout=cfg.dropout, attn_dropout=cfg.dropout)
            phase_branch = basic.SADecoder(cfg.a_dim, cfg.a_dim, cfg.hid_dim, l, cfg.a_layers, in_map=False)
        elif cfg.a == 'sca': # self+cross-attention layers, for input blocks when video transcripts are not available
            layer = basic.SCALayer(cfg.a_dim, cfg.hid_dim, cfg.a_nhead, cfg.a_ffdim, dropout=cfg.dropout, attn_dropout=cfg.dropout)
            norm = torch.nn.LayerNorm(cfg.a_dim)
            phase_branch = basic.SCADecoder(cfg.a_dim, cfg.a_dim, cfg.hid_dim, layer, cfg.a_layers, norm=norm, in_map=False)
        elif cfg.a in ['gru', 'gru_om']: # GRU, for input blocks when video transcripts are available
            assert self.cfg.B2Q_Net.trans
            out_map = (cfg.a == 'gru_om')
            phase_branch = basic.ActionUpdate_GRU(cfg.a_dim, cfg.a_dim, cfg.hid_dim, cfg.a_layers, dropout=cfg.dropout, out_map=out_map)
        else:
            raise ValueError(cfg.a)

        return phase_branch

    def create_cross_attention(self, cfg, outdim, kq_pos=True):
        # one layer of cross-attention for cross-branch communication
        layer = basic.X2Y_map(cfg.hid_dim, cfg.hid_dim, outdim, 
            head_dim=cfg.hid_dim,
            dropout=cfg.dropout, kq_pos=kq_pos)
        
        return layer

    @staticmethod
    def _eval(phase_clogit, p2f_attn, frame_clogit, weight):
        fbranch_prob = torch.softmax(frame_clogit.squeeze(1), dim=-1)

        phase_clogit = phase_clogit.squeeze(1)
        p2f_attn = p2f_attn.squeeze(0)
        qtk_cpred = phase_clogit.argmax(1) 
        null_cid = phase_clogit.shape[-1] - 1
        phase_loc = torch.where(qtk_cpred!=null_cid)[0]

        if len(phase_loc) == 0:
            return fbranch_prob, fbranch_prob, fbranch_prob

        qtk_prob = torch.softmax(phase_clogit[:, :-1], dim=1) # remove logit of null classes
        phase_pred = p2f_attn[:, phase_loc].argmax(-1) # p2f_attn.shape  [14571, 60]
        phase_pred = phase_loc[phase_pred]
        abranch_prob = qtk_prob[phase_pred]

        prob = (1-weight) * abranch_prob + weight * fbranch_prob
        return prob, abranch_prob, fbranch_prob

    @staticmethod
    def _eval_w_transcript(transcript, p2f_attn):
        N = len(transcript)
        p2f_attn = p2f_attn[0, :, :N] # 1, f, a -> f, s'
        pred = p2f_attn.argmax(1) # f
        pred = transcript[pred]
        return pred

    def eval(self, transcript=None):
        if not self.cfg.B2Q_Net.trans:
            return self._eval(self.phase_clogit, self.p2f_attn, self.frame_clogit, self.cfg.B2Q_Net.mwt)
        else:
            return self._eval_w_transcript(transcript, self.p2f_attn)

class InputBlock_frame(Block):
    def __init__(self, cfg, opts, in_dim, nclass):
        super().__init__()
        self.cfg = cfg
        self.nclass = nclass

        cfg = cfg.Bi

        self.frame_branch = self.create_fbranch(cfg, opts, in_dim, f_inmap=True)
        self.phase_branch = None

    def forward(self, frame_feature, frame_pos):
        # frame branch
        frame_feature = self.frame_branch(frame_feature)
        frame_feature, frame_clogit = self.process_feature(frame_feature, self.nclass)
                
        # save features for loss and evaluation
        self.frame_clogit = frame_clogit 

        return frame_feature

    def compute_loss(self, criterion: loss.MatchCriterion, match=None):
        
        frame_loss = criterion.frame_loss(self.frame_clogit.squeeze(1))

        frame_clogit = torch.transpose(self.frame_clogit, 0, 1) 
        smooth_loss = loss.smooth_loss(frame_clogit)

        return frame_loss + self.cfg.Loss.sw * smooth_loss

class InputBlock_phase(Block):
    def __init__(self, cfg, opts, in_dim, nclass):
        super().__init__()
        self.opts = opts
        self.cfg = cfg
        self.nclass = nclass

        cfg = cfg.Bi

        self.frame_branch = None
        self.phase_branch = self.create_abranch(cfg)

    def forward(self, frame_feature, phase_feature, frame_pos, phase_pos, phase_clogit=None):

        # phase branch
        phase_feature = self.phase_branch(phase_feature, frame_feature, pos=frame_pos, query_pos=phase_pos)
        phase_feature, phase_clogit = self.process_feature(phase_feature, self.nclass+1)
        
        # save features for loss and evaluation
        self.phase_clogit = phase_clogit

        return frame_feature, phase_feature

    def compute_loss(self, criterion: loss.MatchCriterion, match=None):
        
        atk_loss = criterion.phase_token_loss(match, self.phase_clogit)

        return atk_loss

class UpdateBlock(Block):

    def __init__(self, cfg, opts, nclass):
        super().__init__()
        self.cfg = cfg
        self.nclass = nclass

        cfg = cfg.Bu
        
        # fbranch
        self.frame_branch = self.create_fbranch(cfg)

        # f2p: query is phase
        self.f2p_layer = self.create_cross_attention(cfg, cfg.a_dim)

        # abranch
        self.phase_branch = self.create_abranch(cfg)

        # p2f: query is frame
        self.p2f_layer = self.create_cross_attention(cfg, cfg.f_dim)

    def forward(self, frame_feature, phase_feature, frame_pos, phase_pos):
        # a->f
        phase_feature = self.f2p_layer(frame_feature, phase_feature, X_pos=frame_pos, Y_pos=phase_pos)

        # a branch
        phase_feature = self.phase_branch(phase_feature, phase_pos)
        phase_feature, phase_clogit = self.process_feature(phase_feature, self.nclass+1)

        # f->a
        frame_feature = self.p2f_layer(phase_feature, frame_feature, X_pos=phase_pos, Y_pos=frame_pos)

        # f branch
        frame_feature = self.frame_branch(frame_feature)
        frame_feature, frame_clogit = self.process_feature(frame_feature, self.nclass)
        
        # save features for loss and evaluation
        self.frame_clogit = frame_clogit 
        self.phase_clogit = phase_clogit 
        self.f2p_attn = self.f2p_layer.attn[0]
        self.p2f_attn = self.p2f_layer.attn[0]
        self.f2p_attn_logit = self.f2p_layer.attn_logit[0].unsqueeze(0)
        self.p2f_attn_logit = self.p2f_layer.attn_logit[0].unsqueeze(0)
        return frame_feature, phase_feature

    def compute_loss(self, criterion: loss.MatchCriterion, match=None):
        frame_loss = criterion.frame_loss(self.frame_clogit.squeeze(1)) 
        atk_loss = criterion.phase_token_loss(match, self.phase_clogit)
        f2p_loss = criterion.cross_attn_loss(match, torch.transpose(self.f2p_attn_logit, 1, 2), dim=1)
        p2f_loss = criterion.cross_attn_loss(match, self.p2f_attn_logit, dim=2)

        # temporal smoothing loss
        al = loss.smooth_loss( self.p2f_attn_logit )
        fl = loss.smooth_loss( torch.transpose(self.f2p_attn_logit, 1, 2) )
        frame_clogit = torch.transpose(self.frame_clogit, 0, 1) # f, 1, c -> 1, f, c
        l = loss.smooth_loss( frame_clogit )
        smooth_loss = al + fl + l

        return atk_loss + f2p_loss + p2f_loss + frame_loss + self.cfg.Loss.sw * smooth_loss


class UpdateBlockTDU(Block):
    """
    Update Block with Temporal Downsampling and Upsampling
    """

    def __init__(self, cfg, opts, nclass):
        super().__init__()
        self.cfg = cfg
        self.nclass = nclass

        cfg = cfg.BU

        # fbranch
        self.frame_branch = self.create_fbranch(cfg)

        # layers for temporal downsample and upsample
        self.seg_update = nn.GRU(cfg.hid_dim, cfg.hid_dim//2, cfg.s_layers, bidirectional=True)
        self.seg_combine = nn.Linear(cfg.hid_dim, cfg.hid_dim)

        # f2p: query is phase
        self.f2p_layer = self.create_cross_attention(cfg, cfg.a_dim)

        # abranch
        self.phase_branch = self.create_abranch(cfg)

        # p2f: query is frame
        self.p2f_layer = self.create_cross_attention(cfg, cfg.f_dim)

        # layers for temporal downsample and upsample
        self.sf_merge = nn.Sequential(nn.Linear((cfg.hid_dim+cfg.f_dim), cfg.f_dim), nn.ReLU())


    def temporal_downsample(self, frame_feature):

        # get phase segments based on predictions
        cprob = frame_feature[:, :, -self.nclass:]
        _, pred = cprob[:, 0].max(dim=-1)
        pred = utils.to_numpy(pred)
        segs = utils.parse_label(pred)

        tdu = basic.TemporalDownsampleUpsample(segs)
        tdu.to(cprob.device)

        # downsample frames to segments
        seg_feature = tdu.feature_frame2seg(frame_feature)

        # refine segment features
        seg_feature, hidden = self.seg_update(seg_feature)
        seg_feature = torch.relu(seg_feature)
        seg_feature = self.seg_combine(seg_feature) # combine forward and backward features
        seg_feature, seg_clogit = self.process_feature(seg_feature, self.nclass)

        return tdu, seg_feature, seg_clogit

    def temporal_upsample(self, tdu, seg_feature, frame_feature):

        # upsample segments to frames
        s2f = tdu.feature_seg2frame(seg_feature)
        
        # merge with original framewise features to keep low-level details
        frame_feature = self.sf_merge(torch.cat([s2f, frame_feature], dim=-1))

        return frame_feature

    def forward(self, frame_feature, phase_feature, frame_pos, phase_pos):
        # downsample frame features to segment features
        tdu, seg_feature, seg_clogit = self.temporal_downsample(frame_feature) # seg_feature: S, 1, H
        # f->a
        seg_center = torch.LongTensor([ int( (s.start+s.end)/2 ) for s in tdu.segs ]).to(seg_feature.device)
        seg_pos = frame_pos[seg_center]
        phase_feature = self.f2p_layer(seg_feature, phase_feature, X_pos=seg_pos, Y_pos=phase_pos)

        # a branch
        phase_feature = self.phase_branch(phase_feature, phase_pos)
        phase_feature, phase_clogit = self.process_feature(phase_feature, self.nclass+1)

        # a->f
        seg_feature = self.p2f_layer(phase_feature, seg_feature, X_pos=phase_pos, Y_pos=seg_pos)

        # upsample segment features to frame features
        frame_feature = self.temporal_upsample(tdu, seg_feature, frame_feature)

        # f branch
        frame_feature = self.frame_branch(frame_feature)
        frame_feature, frame_clogit = self.process_feature(frame_feature, self.nclass)

        # save features for loss and evaluation       
        self.frame_clogit = frame_clogit 
        self.seg_clogit = seg_clogit
        self.tdu = tdu
        self.phase_clogit = phase_clogit 

        self.f2p_attn_logit = self.f2p_layer.attn_logit[0].unsqueeze(0)
        self.f2p_attn = tdu.attn_seg2frame(self.f2p_layer.attn[0].transpose(2, 1)).transpose(2, 1)
        self.p2f_attn_logit = self.p2f_layer.attn_logit[0].unsqueeze(0) 
        self.p2f_attn = tdu.attn_seg2frame(self.p2f_layer.attn[0])
        
        return frame_feature, phase_feature

    def compute_loss(self, criterion: MatchCriterion, match=None):
        frame_loss = criterion.frame_loss(self.frame_clogit.squeeze(1))
        seg_loss = criterion.frame_loss_tdu(self.seg_clogit, self.tdu)
        atk_loss = criterion.phase_token_loss(match, self.phase_clogit)
        f2p_loss = criterion.cross_attn_loss_tdu(match, torch.transpose(self.f2p_attn_logit, 1, 2), self.tdu, dim=1)
        p2f_loss = criterion.cross_attn_loss_tdu(match, self.p2f_attn_logit, self.tdu, dim=2)

        frame_clogit = torch.transpose(self.frame_clogit, 0, 1) 
        smooth_loss = loss.smooth_loss( frame_clogit )
        
        return (frame_loss + seg_loss)/ 2 + atk_loss + f2p_loss + p2f_loss + self.cfg.Loss.sw * smooth_loss
