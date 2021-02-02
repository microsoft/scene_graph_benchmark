# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 
# written by Chu Wang during internship at Bing MM team.
import math
import os.path as op
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence
from torch.autograd import Variable
from .roi_sorter import roi_sorter, enumerate_by_image
from .decoder_rnn import DecoderRNN
from .word_vectors import obj_edge_vectors, to_onehot, arange
from maskrcnn_benchmark.layers import nms as _box_nms

def apply_nms(scores, boxes, im_inds=None, nms_thresh=0.7):
    """
    Note - this function is non-differentiable so everything is assumed to be a tensor, not
    a variable.
        """
    just_inds = im_inds is None
    if im_inds is None:
        im_inds = scores.new(scores.size(0)).fill_(0).long()

    keep = []
    im_per = []
    for i, s, e in enumerate_by_image(im_inds):
        keep_im = _box_nms(boxes[s:e], scores[s:e], nms_thresh)
        keep.append(keep_im + s)
        im_per.append(keep_im.size(0))

    inds = torch.cat(keep, 0)
    if just_inds:
        return inds
    return inds, im_per


# bbox overlap fn in FAIR repo
# from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou

# debugged 7/29


class context_encoder(nn.Module):
    """
    Module for computing the object contexts and edge contexts
    """

    def __init__(self, config, obj_classes, rel_classes, in_channels):
        super(context_encoder, self).__init__()
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_classes = len(self.obj_classes)
        self.num_rels = len(self.rel_classes)

        self.debug = config.MODEL.ROI_RELATION_HEAD.NEURAL_MOTIF.DEBUG
        # to add entries to those params in NM repo
        # define model params
        self.mode = config.MODEL.ROI_RELATION_HEAD.MODE

        self.num_layers_obj = config.MODEL.ROI_RELATION_HEAD.NEURAL_MOTIF.OBJ_LSTM_NUM_LAYERS
        self.num_layers_edge = config.MODEL.ROI_RELATION_HEAD.NEURAL_MOTIF.EDGE_LSTM_NUM_LAYERS

        self.embed_dim = config.MODEL.ROI_RELATION_HEAD.NEURAL_MOTIF.EMBED_DIM
        self.glove_path = op.join(config.DATA_DIR, config.MODEL.ROI_RELATION_HEAD.NEURAL_MOTIF.GLOVE_PATH)
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.NEURAL_MOTIF.HIDDEN_DIM

        self.obj_dim = in_channels
        self.dropout_rate = config.MODEL.ROI_RELATION_HEAD.NEURAL_MOTIF.DROPOUT

        self.pass_in_obj_feats_to_decoder = config.MODEL.ROI_RELATION_HEAD.NEURAL_MOTIF.OBJ_FEAT_TO_DECODER
        self.pass_in_obj_feats_to_edge = config.MODEL.ROI_RELATION_HEAD.NEURAL_MOTIF.OBJ_FEAT_TO_EDGE
        self.order = config.MODEL.ROI_RELATION_HEAD.NEURAL_MOTIF.ORDER
        assert self.order in ('size', 'confidence', 'random', 'leftright')

        embed_vecs = obj_edge_vectors(self.obj_classes,
                                      wv_dir=self.glove_path,
                                      wv_dim=self.embed_dim)

        self.obj_embed = nn.Embedding(self.num_classes, self.embed_dim)
        self.obj_embed.weight.data = embed_vecs.clone()

        # position embedding
        self.pos_embed_dim = config.MODEL.ROI_RELATION_HEAD.NEURAL_MOTIF.POS_EMBED_DIM
        self.pos_embed = nn.Sequential(*[
            nn.BatchNorm1d(4, momentum=config.MODEL.ROI_RELATION_HEAD.NEURAL_MOTIF.POS_BATCHNORM_MOMENTUM),
            nn.Linear(4, self.pos_embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
        ])

        # define RNNs
        # object context encoder
        # assert self.num_layers_obj > 0, "Need LSTM layer in Neural Motif!"
        if self.num_layers_obj > 0:
            self.obj_ctx_rnn = torch.nn.LSTM(
                input_size=self.obj_dim + self.embed_dim + self.pos_embed_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers_obj,
                dropout=self.dropout_rate,
                bidirectional=False
            )

            # bi LSTM's output dim is 2* hidden_dim
            # this contradicts with NM repo's variable dimensions
            # thus uses regular LSTM
            self.decoder_inputs_dim = 2 * self.hidden_dim if self.obj_ctx_rnn.bidirectional else self.hidden_dim
            # for previous embeddings during decode, line 276 decoder_rnn.py:
            # timestep_input = torch.cat((sequence_tensor[start_ind:end_ind], previous_embed), 1)
            self.decoder_inputs_dim += self.embed_dim

            if self.pass_in_obj_feats_to_decoder:
                self.decoder_inputs_dim += (self.obj_dim + self.embed_dim + self.pos_embed_dim)

            embed_vecs_decoder = obj_edge_vectors(['start'] + self.obj_classes,
                                                  wv_dir=self.glove_path,
                                                  wv_dim=self.embed_dim)

            self.decoder_rnn = DecoderRNN(
                self.obj_classes, embed_dim=self.embed_dim,
                inputs_dim=self.decoder_inputs_dim,
                hidden_dim=self.hidden_dim,
                recurrent_dropout_probability=self.dropout_rate,
                embed_vecs=embed_vecs_decoder
            )
        else:
            if self.num_layers_edge > 0:
                self.obj_ctx_trans = nn.Linear(self.obj_dim + self.embed_dim + self.pos_embed_dim, self.hidden_dim)
                self.obj_ctx_trans.weight.data.normal_(0, math.sqrt(
                    1.0 / (self.obj_dim + self.embed_dim + self.pos_embed_dim)))
                self.obj_ctx_trans.bias.data.zero_()
            self.decoder_lin = nn.Linear(self.obj_dim + self.embed_dim + self.pos_embed_dim, self.num_classes)
            self.decoder_lin.weight.data.normal_(0, math.sqrt(
                1.0 / (self.obj_dim + self.embed_dim + self.pos_embed_dim)))
            self.decoder_lin.bias.data.zero_()

        # edge context encoder
        # assert self.num_layers_edge > 0, "Need LSTM layer in Neural Motif!"
        if self.num_layers_edge > 0:
            self.obj_embed2 = nn.Embedding(self.num_classes, self.embed_dim)
            self.obj_embed2.weight.data = embed_vecs.clone()

            self.edge_rnn_input_dim = self.embed_dim
            self.edge_rnn_input_dim += self.hidden_dim
            if self.pass_in_obj_feats_to_edge:
                self.edge_rnn_input_dim += self.obj_dim

            self.edge_ctx_rnn = torch.nn.LSTM(
                input_size=self.edge_rnn_input_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers_edge,
                dropout=self.dropout_rate,
                bidirectional=False
            )

        # ~~~~~~~~~~~~~~~~
        # define a sorter class instance
        # to call, use :
        # perm, inv_perm, ls_transposed = self.roi_sorter.sort(im_inds.data, confidence, box_priors)
        # ~~~~~~~~~~~~~~~~
        self.roi_sorter = roi_sorter(order=self.order)

    def edge_context(self, obj_feats, obj_dists, im_inds, obj_preds, box_priors=None):
        """
        2nd overall phase for LSTM context encoding.
        :param box_priors:
        :param obj_preds:
        :param obj_feats: [num_obj, img_dim + object embedding0 dim]
        :param obj_dists: [num_obj, #classes]
        :param im_inds: [num_obj] the indices of the images
        :return: edge_ctx: [num_obj, #feats] For later!
        """
        # Only use hard embeddings
        obj_embed2 = self.obj_embed2(obj_preds)
        # obj_embed3 = F.softmax(obj_dists, dim=1) @ self.obj_embed3.weight
        inp_feats = torch.cat((obj_embed2, obj_feats), 1)

        # Sort by the confidence of the maximum detection.
        # there should be softmax here.
        confidence = F.softmax(obj_dists, dim=1).data.view(-1)[
            obj_preds.data + arange(obj_preds.data) * self.num_classes]
        perm, inv_perm, ls_transposed = self.roi_sorter.sort(im_inds.data, confidence, box_priors)

        edge_input_packed = PackedSequence(inp_feats[perm], ls_transposed)
        edge_reps = self.edge_ctx_rnn(edge_input_packed)[0][0]

        # now we're good! unperm
        edge_ctx = edge_reps[inv_perm]

        return edge_ctx

    def obj_context(self, obj_feats, obj_dists, im_inds,
                    obj_labels=None,
                    box_priors=None,
                    boxes_per_cls=None):
        """
        Object context and object classification.
        1st overall phase for LSTM encoding.

        :param obj_feats: [num_obj, img_dim + object embedding0 dim]
        :param obj_dists: [num_obj, #classes]
        :param im_inds: [num_obj] the indices of the images
        :param obj_labels: [num_obj] the GT labels of the image
        :param box_priors: [num_obj, 4] boxes.
        :param boxes_per_cls:
        :return: obj_dists: [num_obj, #classes] new probability distribution.
                 obj_preds: argmax of that distribution.
                 obj_final_ctx: [num_obj, #feats] For later!
        """

        # Sort by the confidence of the maximum detection.
        # there should be NO softmax
        confidence = obj_dists.data[:, 1:].max(1)[0]
        perm, inv_perm, ls_transposed = self.roi_sorter.sort(im_inds.data, confidence, box_priors)

        # Pass object features, sorted by score, into the encoder LSTM
        # ls_transposed should be Tensor: batch_sizes
        obj_inp_rep = obj_feats[perm].contiguous()
        input_packed = PackedSequence(obj_inp_rep, ls_transposed)

        encoder_rep = self.obj_ctx_rnn(input_packed)[0][0]

        # Decode in order
        if self.mode != 'predcls':
            decoder_inp = PackedSequence(torch.cat((obj_inp_rep, encoder_rep), 1)
                                         if self.pass_in_obj_feats_to_decoder else encoder_rep,
                                         ls_transposed)
            obj_dists, obj_preds = self.decoder_rnn(
                decoder_inp,
                labels=obj_labels[perm] if obj_labels is not None else None,
                boxes_for_nms=boxes_per_cls[perm] if boxes_per_cls is not None else None,
            )
            obj_preds = obj_preds[inv_perm]
            obj_dists = obj_dists[inv_perm]
        else:
            assert obj_labels is not None
            obj_preds = obj_labels
            obj_dists = Variable(to_onehot(obj_preds.data, self.num_classes))
        encoder_rep = encoder_rep[inv_perm]

        return obj_dists, obj_preds, encoder_rep

    def forward(self, obj_feats, obj_dists, im_inds,
                obj_labels=None,
                box_priors=None,
                boxes_per_cls=None):
        """
        Forward pass through the object and edge context
        Inputs:
            obj_feats: per bbox obj feature map, N by feat_dim
            obj_dists: per obj class probability distribution, output by detector.
                                                N by classes
            im_inds: per obj's corresponding image_index in a single batch N by 1
            obj_labels: per obj's ground truth label. N by 1.
            box_priors: box data per RPN output, for each object N by 4
            boxes_per_cls**: needs help on understanding this var in NM repo
            todo: for now ignored. Maybe a work item in future.
            https://github.com/rowanz/neural-motifs/blob/master/lib/rel_model.py#L197

        Outputs:
            obj_prob_dists: per obj class probability distribution result
            obj_preds: per obj class predictions
            edge_ctx: encoded feature vector for each obj
        """
        # @ is either a look up operation or matrix multiplication. Can't find answer on Google...
        # there should be no softmax here because post detection there is already a softmax
        # see box_head.py
        # obj_embed = F.softmax(obj_dists, dim=1) @ self.obj_embed.weight
        obj_embed = obj_dists @ self.obj_embed.weight
        pos_embed = self.pos_embed(Variable(self.roi_sorter.center_size(box_priors)))
        obj_pre_lstm_rep = torch.cat((obj_feats, obj_embed, pos_embed), 1)

        if self.num_layers_obj > 0:
            # get first round of lstm encoding results
            obj_dists2, obj_preds, obj_ctx = self.obj_context(
                obj_pre_lstm_rep,
                obj_dists,
                im_inds,
                obj_labels,
                box_priors,
                boxes_per_cls,
            )
        else:
            # UNSURE WHAT TO DO HERE
            if self.mode == 'predcls':
                obj_dists2 = Variable(to_onehot(obj_labels.data, self.num_classes))
            else:
                obj_dists2 = self.decoder_lin(obj_pre_lstm_rep)

            if self.mode == 'sgdet' and not self.training:
                # NMS here for baseline

                probs = F.softmax(obj_dists2, 1)
                if boxes_per_cls is not None:
                    nms_mask = obj_dists2.data.clone()
                    nms_mask.zero_()
                    for c_i in range(1, obj_dists2.size(1)):
                        scores_ci = probs.data[:, c_i]
                        boxes_ci = boxes_per_cls.data[:, c_i]

                        keep = apply_nms(scores_ci, boxes_ci, nms_thresh=0.3)
                        nms_mask[:, c_i][keep] = 1

                    obj_preds = Variable(nms_mask * probs.data)[:, 1:].max(1)[1] + 1
                else:
                    obj_preds = probs[:, 1:].max(1)[1] + 1
            else:
                obj_preds = obj_labels if obj_labels is not None else obj_dists2[:, 1:].max(1)[1] + 1
            if self.num_layers_edge > 0:
                obj_ctx = self.obj_ctx_trans(obj_pre_lstm_rep)

        # get second round of lstm encoding result
        edge_ctx = None
        if self.num_layers_edge > 0:
            edge_ctx = self.edge_context(
                torch.cat((obj_feats, obj_ctx), 1) if self.pass_in_obj_feats_to_edge else obj_ctx,
                obj_dists=obj_dists2.detach(),  # Was previously obj_logits.
                im_inds=im_inds,
                obj_preds=obj_preds,
                box_priors=box_priors,
            )

        # return obj_dists2, obj_preds, obj_ctx # for debugging
        return obj_dists2, obj_preds, edge_ctx
