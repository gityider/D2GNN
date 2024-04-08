import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from SeqContext import SeqContext
from GNN import GNN
from functions import batch_graphify_label, batch_feat
from IB import IB
import numpy as np
import utils

log = utils.get_logger()

class D2GNN(nn.Module):
    def __init__(self, args):
        super(D2GNN, self).__init__()

        # dataset_label_dict = {
        #     "iemocap": {"hap": 0, "sad": 1, "neu": 2, "ang": 3, "exc": 4, "fru": 5},
        # }
        # tag_size = len(dataset_label_dict[args.dataset])

        self.wp = args.wp
        self.wf = args.wf
        self.device = args.device

        # speaker, temporal
        edge_type_to_idx = {'00': 0, '01': 1, '10': 2, '11': 3}
        self.edge_type_to_idx = edge_type_to_idx
        log.debug(self.edge_type_to_idx)

        u_dim = 1380
        self.hidden_dim = args.hidden_size
        self.num_view = args.n_views
        self.view_list = [100, 768, 512]
        self.num_classes = args.n_classes
        self.output_dropout = 0.5

        self.beta = 10

        # dim encoder for concat and unimodal part
        self.rnn = SeqContext(u_dim, self.hidden_dim, args)
        self.encoders = []  # dim encoder
        for v in range(self.num_view):
            self.encoders.append(Encoder(self.view_list[v], self.hidden_dim).to(self.device))
        self.encoders = nn.ModuleList(self.encoders)

        self.decoder_concat = Decoder(self.hidden_dim, self.hidden_dim * 2).to(self.device) # for concat part
        self.decoders = []
        for v in range(self.num_view):
            self.decoders.append(Decoder(self.hidden_dim, self.hidden_dim * 2).to(self.device)) # (input_dim, hidden_dim)
        self.decoders = nn.ModuleList(self.decoders)

        # ------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------
        # ----------------concat part----------------
        self.IB = IB(shape_x=self.hidden_dim, shape_z=self.hidden_dim, shape_y=self.num_classes,
                     per_class=self.num_classes, device=self.device, beta=self.beta)

        # # fc layer for total feature init judgement
        self.judge_dim = self.hidden_dim
        self.proj1_init_concat = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.proj2_init_concat = nn.Linear(self.hidden_dim, self.judge_dim)
        self.out_layer_init_concat = nn.Linear(self.judge_dim, self.num_classes)

        # ----------------audio modality---------------
        self.IB_a = IB(shape_x=self.hidden_dim, shape_z=self.hidden_dim, shape_y=self.num_classes,
                       per_class=self.num_classes, device=self.device, beta=self.beta)

        # # fc layer for audio feature init judgement
        self.proj1_init_a = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.proj2_init_a = nn.Linear(self.hidden_dim, self.judge_dim)
        self.out_layer_init_a = nn.Linear(self.judge_dim, self.num_classes)

        # ----------------text modality----------------
        self.IB_l = IB(shape_x=self.hidden_dim, shape_z=self.hidden_dim, shape_y=self.num_classes,
                       per_class=self.num_classes, device=self.device, beta=self.beta)

        # # fc layer for text feature init judgement
        self.proj1_init_l = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.proj2_init_l = nn.Linear(self.hidden_dim, self.judge_dim)
        self.out_layer_init_l = nn.Linear(self.judge_dim, self.num_classes)

        # ----------------video modality----------------
        self.IB_v = IB(shape_x=self.hidden_dim, shape_z=self.hidden_dim, shape_y=self.num_classes,
                       per_class=self.num_classes, device=self.device, beta=self.beta)

        # # fc layer for video feature init judgement
        self.proj1_init_v = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.proj2_init_v = nn.Linear(self.hidden_dim, self.judge_dim)
        self.out_layer_init_v = nn.Linear(self.judge_dim, self.num_classes)

        # ------------------------------------------------------------------------------------------------
        # distill proj for judge part
        self.proj_j_fusion_concat = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.proj_j_fusion_a = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.proj_j_fusion_l = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.proj_j_fusion_v = nn.Linear(self.hidden_dim, self.hidden_dim)

        # distill proj for IB part
        self.proj_IB_fusion_concat = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.proj_IB_fusion_a = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.proj_IB_fusion_l = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.proj_IB_fusion_v = nn.Linear(self.hidden_dim, self.hidden_dim)

        # distill weight
        self.alpha = 1 / 8

        self.W_weight_h_a = nn.Linear(self.hidden_dim * 2, 1)
        self.W_weight_h_l = nn.Linear(self.hidden_dim * 2, 1)
        self.W_weight_h_v = nn.Linear(self.hidden_dim * 2, 1)

        self.W_weight_j_a = nn.Linear(self.hidden_dim * 2, 1)
        self.W_weight_j_l = nn.Linear(self.hidden_dim * 2, 1)
        self.W_weight_j_v = nn.Linear(self.hidden_dim * 2, 1)

        # ------------------------------------------------------------------------------------------------
        # graph
        g_dim = self.hidden_dim
        h1_dim = self.hidden_dim
        h2_dim = self.hidden_dim
        self.gcn_j = GNN(g_dim, h1_dim, h2_dim, args)
        self.gcn_z = GNN(g_dim, h1_dim, h2_dim, args)
        # ------------------------------------------------------------------------------------------------
        # prediction
        # fc for prediction
        self.proj1_out_pred = nn.Linear(h2_dim * 2, self.hidden_dim)
        self.proj2_out_pred = nn.Linear(self.hidden_dim, self.judge_dim)
        self.out_layer_pred = nn.Linear(self.judge_dim, self.num_classes)
        # ------------------------------------------------------------------------------------------------

        # loss for reconstruct
        self.MSE = MSE()

        # loss function for classification
        if args.class_weight:
            if args.dataset == "iemocap":
                self.loss_weights = torch.tensor(
                    [
                        1 / 0.086747,
                        1 / 0.144406,
                        1 / 0.227883,
                        1 / 0.160585,
                        1 / 0.127711,
                        1 / 0.252668,
                    ]
                ).to(args.device)
            self.nll_loss = nn.NLLLoss(self.loss_weights)
            print("*******weighted loss*******")
        else:
            self.nll_loss = nn.NLLLoss()
            self.bce_loss = nn.BCEWithLogitsLoss(reduction="sum")

        # loss for emotion prototype
        self.d_c = 32
        self.proj_c = nn.Linear(h2_dim * 2, self.d_c)
        self.cls_prototypes = nn.Parameter(torch.randn(self.num_classes, self.d_c))

    def get_rep(self, data):

        node_features = self.rnn(data["text_len_tensor"], data["input_tensor"])
        features, edge_index, edge_type, edge_index_lengths = batch_graphify_label(
            node_features,
            data["text_len_tensor"],
            data["speaker_tensor"],
            self.wp,
            self.wf,
            self.edge_type_to_idx,
            self.device,
        )
        return features, edge_index, edge_type, edge_index_lengths

    def forward(self, data):
        # get batch samples and edge info
        out_features, edge_index, edge_type, edge_index_lengths = self.get_rep(data)

        data_input = batch_feat(data["input_tensor"], data["text_len_tensor"],
                                self.device)
        feat_a = data_input[:, :100]
        feat_l = data_input[:, 100:868]
        feat_v = data_input[:, 868:]
        data_unimodal = []
        data_unimodal.append(feat_a)
        data_unimodal.append(feat_l)
        data_unimodal.append(feat_v)

        out_dims = []
        for v in range(self.num_view):
            out_dim = self.encoders[v](data_unimodal[v])
            out_dims.append(out_dim)

        # --------------------------------------two part---------------------------------------
        # ---------------------------IB part---------------------------------
        # 1.concat IB part
        z_features = self.IB.forward(x=out_features, y=data['label_tensor'])

        # 2.audio IB part
        z_a_features = self.IB_a.forward(x=out_dims[0], y=data['label_tensor'])

        # 3.text IB part
        z_l_features = self.IB_l.forward(x=out_dims[1], y=data['label_tensor'])

        # 4.video IB part
        z_v_features = self.IB_v.forward(x=out_dims[2], y=data['label_tensor'])

        # ---------------------------judge part---------------------------------
        # 5.concat judge part
        j_proj_con = self.proj2_init_concat(
            F.dropout(F.relu(self.proj1_init_concat(out_features)), p=self.output_dropout))  # (Bsz, judge_dim)

        # 6.audio judge part
        j_proj_a = self.proj2_init_a(
            F.dropout(F.relu(self.proj1_init_a(out_dims[0])), p=self.output_dropout))  # (Bsz, judge_dim)

        # 7.text judge part
        j_proj_l = self.proj2_init_l(
            F.dropout(F.relu(self.proj1_init_l(out_dims[1])), p=self.output_dropout))  # (Bsz, judge_dim)

        # 8.video judge part
        j_proj_v = self.proj2_init_v(
            F.dropout(F.relu(self.proj1_init_v(out_dims[2])), p=self.output_dropout))  # (Bsz, judge_dim)

        # --------------------------------------distill part------------------------------------------

        # -------------------------------distill judge part ---------------------------------
        j_dw_proj_concat = self.proj_j_fusion_concat(j_proj_con)
        j_dw_proj_a = self.proj_j_fusion_a(j_proj_a)
        j_dw_proj_l = self.proj_j_fusion_l(j_proj_l)
        j_dw_proj_v = self.proj_j_fusion_v(j_proj_v)

        w_j_a = self.W_weight_j_a(torch.cat([j_dw_proj_concat, j_dw_proj_a], dim=1))
        w_j_l = self.W_weight_j_l(torch.cat([j_dw_proj_concat, j_dw_proj_l], dim=1))
        w_j_v = self.W_weight_j_v(torch.cat([j_dw_proj_concat, j_dw_proj_v], dim=1))

        w_j = []
        w_j.append(w_j_a)
        w_j.append(w_j_l)
        w_j.append(w_j_v)
        w_j = torch.cat(w_j, dim=1)
        w_j = torch.tanh(w_j)
        w_j = F.softmax(w_j, dim=1).transpose(0, 1)

        j_feats = j_proj_con + torch.mul((w_j[0]).unsqueeze(1), j_proj_a) \
                  + torch.mul((w_j[1]).unsqueeze(1), j_proj_l) \
                  + torch.mul((w_j[2]).unsqueeze(1), j_proj_v)

        # -------------------------------distill IB part ---------------------------------
        z_dw_proj_concat = self.proj_IB_fusion_concat(z_features)
        z_dw_proj_a = self.proj_IB_fusion_a(z_a_features)
        z_dw_proj_l = self.proj_IB_fusion_l(z_l_features)
        z_dw_proj_v = self.proj_IB_fusion_v(z_v_features)

        w_h_a = self.W_weight_h_a(torch.cat([z_dw_proj_concat, z_dw_proj_a], dim=1))
        w_h_l = self.W_weight_h_l(torch.cat([z_dw_proj_concat, z_dw_proj_l], dim=1))
        w_h_v = self.W_weight_h_v(torch.cat([z_dw_proj_concat, z_dw_proj_v], dim=1))

        w_h = []
        w_h.append(w_h_a)
        w_h.append(w_h_l)
        w_h.append(w_h_v)
        w_h = torch.cat(w_h, dim=1)
        w_h = torch.tanh(w_h)
        w_h = F.softmax(w_h, dim=1).transpose(0, 1)

        z_feats = z_features + torch.mul((w_h[0]).unsqueeze(1), z_a_features) \
                  + torch.mul((w_h[1]).unsqueeze(1), z_l_features) \
                  + torch.mul((w_h[2]).unsqueeze(1), z_v_features)

        # --------------------------------------graph part------------------------------------------
        nsamps = out_features.shape[0]

        # graph IB part
        node_z_feats = torch.cat([out_features, z_feats], dim=0)
        graph_z_feat_out = self.gcn_z(node_z_feats, edge_index, edge_type)
        graph_z_out = graph_z_feat_out[:nsamps, :]

        # graph judge part
        node_j_feats = torch.cat([out_features, j_feats], dim=0)
        graph_j_feat_out = self.gcn_j(node_j_feats, edge_index, edge_type)
        graph_j_out = graph_j_feat_out[:nsamps, :]

        l_last_judge_proj = self.proj2_out_pred(
            F.dropout(F.relu(self.proj1_out_pred(torch.cat([graph_z_out, graph_j_out], dim=1))),
                      p=self.output_dropout, training=True))
        l_last_judge_out = self.out_layer_pred(l_last_judge_proj)
        log_prob_last_judge = F.log_softmax(l_last_judge_out, dim=1)
        out = torch.argmax(log_prob_last_judge, dim=-1)
        return out

    def get_loss(self, data):
        # get batch samples and edge info
        out_features, edge_index, edge_type, edge_index_lengths = self.get_rep(data)

        data_input = batch_feat(data["input_tensor"], data["text_len_tensor"], self.device)
        feat_a = data_input[:, :100]
        feat_l = data_input[:, 100:868]
        feat_v = data_input[:, 868:]
        data_unimodal = []
        data_unimodal.append(feat_a)
        data_unimodal.append(feat_l)
        data_unimodal.append(feat_v)

        out_dims = []
        for v in range(self.num_view):
            out_dim = self.encoders[v](data_unimodal[v])
            out_dims.append(out_dim)

        # --------------------------------------two part---------------------------------------
        # ---------------------------IB part---------------------------------
        # 1.concat IB part
        z_features = self.IB.forward(x=out_features, y=data['label_tensor'])

        # 2.audio IB part
        z_a_features = self.IB_a.forward(x=out_dims[0], y=data['label_tensor'])

        # 3.text IB part
        z_l_features = self.IB_l.forward(x=out_dims[1], y=data['label_tensor'])

        # 4.video IB part
        z_v_features = self.IB_v.forward(x=out_dims[2], y=data['label_tensor'])

        # ---------------------------judge part---------------------------------
        # 5.concat judge part
        j_proj_con = self.proj2_init_concat(
            F.dropout(F.relu(self.proj1_init_concat(out_features)), p=self.output_dropout))  # (Bsz, judge_dim)
        j_con_out = self.out_layer_init_concat(j_proj_con)  # (Bsz, num_classes)

        # 6.audio judge part
        j_proj_a = self.proj2_init_a(
            F.dropout(F.relu(self.proj1_init_a(out_dims[0])), p=self.output_dropout))  # (Bsz, judge_dim)
        j_a_out = self.out_layer_init_a(j_proj_a)

        # 7.text judge part
        j_proj_l = self.proj2_init_l(
            F.dropout(F.relu(self.proj1_init_l(out_dims[1])), p=self.output_dropout))  # (Bsz, judge_dim)
        j_l_out = self.out_layer_init_l(j_proj_l)

        # 8.video judge part
        j_proj_v = self.proj2_init_v(
            F.dropout(F.relu(self.proj1_init_v(out_dims[2])), p=self.output_dropout))  # (Bsz, judge_dim)
        j_v_out = self.out_layer_init_v(j_proj_v)

        # --------------------------------------distill part------------------------------------------
        # -------------------------------distill judge part ---------------------------------
        j_dw_proj_concat = self.proj_j_fusion_concat(j_proj_con)
        j_dw_proj_a = self.proj_j_fusion_a(j_proj_a)
        j_dw_proj_l = self.proj_j_fusion_l(j_proj_l)
        j_dw_proj_v = self.proj_j_fusion_v(j_proj_v)

        w_j_a = self.W_weight_j_a(torch.cat([j_dw_proj_concat, j_dw_proj_a], dim=1))
        w_j_l = self.W_weight_j_l(torch.cat([j_dw_proj_concat, j_dw_proj_l], dim=1))
        w_j_v = self.W_weight_j_v(torch.cat([j_dw_proj_concat, j_dw_proj_v], dim=1))

        w_j = []
        w_j.append(w_j_a)
        w_j.append(w_j_l)
        w_j.append(w_j_v)
        w_j = torch.cat(w_j, dim=1)
        w_j = torch.tanh(w_j)
        w_j = F.softmax(w_j, dim=1).transpose(0, 1)

        j_feats = j_proj_con + torch.mul((w_j[0]).unsqueeze(1), j_proj_a) \
                  + torch.mul((w_j[1]).unsqueeze(1), j_proj_l) \
                  + torch.mul((w_j[2]).unsqueeze(1), j_proj_v)

        # -------------------------------distill IB part ---------------------------------
        z_dw_proj_concat = self.proj_IB_fusion_concat(z_features)
        z_dw_proj_a = self.proj_IB_fusion_a(z_a_features)
        z_dw_proj_l = self.proj_IB_fusion_l(z_l_features)
        z_dw_proj_v = self.proj_IB_fusion_v(z_v_features)

        w_h_a = self.W_weight_h_a(torch.cat([z_dw_proj_concat, z_dw_proj_a], dim=1))
        w_h_l = self.W_weight_h_l(torch.cat([z_dw_proj_concat, z_dw_proj_l], dim=1))
        w_h_v = self.W_weight_h_v(torch.cat([z_dw_proj_concat, z_dw_proj_v], dim=1))

        w_h = []
        w_h.append(w_h_a)
        w_h.append(w_h_l)
        w_h.append(w_h_v)
        w_h = torch.cat(w_h, dim=1)
        w_h = torch.tanh(w_h)
        w_h = F.softmax(w_h, dim=1).transpose(0, 1)

        z_feats = z_features + torch.mul((w_h[0]).unsqueeze(1), z_a_features) \
                  + torch.mul((w_h[1]).unsqueeze(1), z_l_features) \
                  + torch.mul((w_h[2]).unsqueeze(1), z_v_features)

        # --------------------------------------graph part------------------------------------------
        nsamps = out_features.shape[0]

        # graph IB part
        node_z_feats = torch.cat([out_features, z_feats], dim=0)
        graph_z_feat_out = self.gcn_z(node_z_feats, edge_index, edge_type)
        graph_z_out = graph_z_feat_out[:nsamps, :]

        # graph judge part
        node_j_feats = torch.cat([out_features, j_feats], dim=0)
        graph_j_feat_out = self.gcn_j(node_j_feats, edge_index, edge_type)
        graph_j_out = graph_j_feat_out[:nsamps, :]

        l_last_judge_proj = self.proj2_out_pred(
            F.dropout(F.relu(self.proj1_out_pred(torch.cat([graph_z_out, graph_j_out], dim=1))),
                      p=self.output_dropout, training=True))
        l_last_judge_out = self.out_layer_pred(l_last_judge_proj)

        ## --------------------------------------Loss part------------------------------------------
        # loss for concat IB part (1.)
        loss_IB_concat = self.IB.get_IB_loss()
        # loss for audio IB part (2.)
        loss_IB_a = self.IB_a.get_IB_loss()
        # loss for text IB part (3.)
        loss_IB_l = self.IB_l.get_IB_loss()
        # loss for video IB part (4.)
        loss_IB_v = self.IB_v.get_IB_loss()
        loss_IB = loss_IB_concat + loss_IB_a + loss_IB_l + loss_IB_v

        # loss for concat judge part(5.)
        log_prob_judge_concat = F.log_softmax(j_con_out, dim=1)
        log_prob_judge_a = F.log_softmax(j_a_out, dim=1)
        log_prob_judge_l = F.log_softmax(j_l_out, dim=1)
        log_prob_judge_v = F.log_softmax(j_v_out, dim=1)
        loss_judge = self.nll_loss(log_prob_judge_concat, data['label_tensor']) \
                     + 0.1 * self.nll_loss(log_prob_judge_a, data['label_tensor']) \
                     + 0.1 * self.nll_loss(log_prob_judge_l, data['label_tensor']) \
                     + 0.1 * self.nll_loss(log_prob_judge_v, data['label_tensor'])

        # reconstruct Loss for two part
        recon_concat_feat = self.decoder_concat(torch.cat([j_proj_con, z_features], dim=1))
        recon_a_feat = self.decoders[0](torch.cat([j_proj_a, z_a_features], dim=1))
        recon_l_feat = self.decoders[1](torch.cat([j_proj_l, z_l_features], dim=1))
        recon_v_feat = self.decoders[2](torch.cat([j_proj_v, z_v_features], dim=1))
        loss_recon_concat = self.MSE(recon_concat_feat, out_features)
        loss_recon_a = self.MSE(recon_a_feat, out_dims[0])
        loss_recon_l = self.MSE(recon_l_feat, out_dims[1])
        loss_recon_v = self.MSE(recon_v_feat, out_dims[2])
        loss_recon = loss_recon_concat + loss_recon_a + loss_recon_l + loss_recon_v

        # emotion prototype contrastive loss
        con_feat = torch.cat([graph_z_out, graph_j_out], dim=1)
        loss_constrastive = self.CL_loss(x=con_feat, label_prototype=self.cls_prototypes, past_label_prob=l_last_judge_out, batch_label=data['label_tensor'])

        # cls loss
        log_prob_last_judge = F.log_softmax(l_last_judge_out, dim=1)
        loss_cls = self.nll_loss(log_prob_last_judge, data['label_tensor'])

        # total loss
        loss = loss_cls + 0.5 * (loss_IB + 0.1 * loss_recon + loss_judge) + 1.0 * loss_constrastive
        log.info(f"loss: {loss}, loss_cls: {loss_cls}, loss_IB: {loss_IB}, loss_recon: {loss_recon}, loss_init_judge:{loss_judge}, loss_constrastive: {loss_constrastive}")

        return loss

    def CL_loss(self, x, label_prototype=None, ini_label_prob=None, past_label_prob=None, batch_label=None, temperature=1):

        batch_size = x.shape[0]
        # prob_last_one_hot = label_prototype.index_select(0, batch_label.long())
        ones = torch.sparse.torch.eye(self.num_classes).to(self.device)
        prob_last_one_hot = ones.index_select(0, batch_label).transpose(0, 1)  # [label_num, B]

        contrast_feature = self.proj_c(x)
        masked_logits = prob_last_one_hot.unsqueeze(2) * contrast_feature.repeat(prob_last_one_hot.shape[0], 1, 1)  # [label_num, B, dc]
        logits = torch.mean(masked_logits, dim=1)  # [label_num, dc]

        '''Compute logits for all clusters.'''
        logits = torch.div(torch.matmul(logits, label_prototype.T), temperature)  # [label_num, label_num]

        '''Extract the logits to be maximised.'''
        up_logits = torch.diag(logits)  # [label_num]

        '''Compute contrastive loss.'''
        all_logits = torch.log(torch.sum(torch.exp(logits), dim=1))
        loss = (up_logits - all_logits)
        loss = - loss.mean()

        return loss


def softmax(w, t=1.0, axis=None):
  w = np.array(w) / t
  e = np.exp(w - np.amax(w, axis=axis, keepdims=True))
  dist = e / np.sum(e, axis=axis, keepdims=True)
  return dist


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2 * feature_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2 * feature_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2 * feature_dim),
            nn.ReLU(),
            nn.Linear(2 * feature_dim, input_dim),
            nn.LayerNorm(input_dim)
        )

    def forward(self, x):
        return self.decoder(x)