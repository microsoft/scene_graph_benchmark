# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 
import numpy as np
import torch

def enumerate_by_image(im_inds):
    im_inds_np = im_inds.cpu().numpy()
    initial_ind = int(im_inds_np[0])
    s = 0
    for i, val in enumerate(im_inds_np):
        if val != initial_ind:
            yield initial_ind, s, i
            initial_ind = int(val)
            s = i
    yield initial_ind, s, len(im_inds_np)

# chuw:
# debug/test done 7/25
class roi_sorter():
    def __init__(self, order = 'confidence'):

        super(roi_sorter, self).__init__()
        self.order = order


    def transpose_packed_sequence_inds(self, lengths):
        """
        Goes from a TxB packed sequence to a BxT or vice versa. Assumes that nothing is a variable
        :param ps: PackedSequence
        :return:
        """

        new_inds = []
        new_lens = []
        cum_add = np.cumsum([0] + lengths)
        max_len = lengths[0]
        length_pointer = len(lengths) - 1
        for i in range(max_len):
            while length_pointer > 0 and lengths[length_pointer] <= i:
                length_pointer -= 1
            new_inds.append(cum_add[:(length_pointer + 1)].copy())
            cum_add[:(length_pointer + 1)] += 1
            new_lens.append(length_pointer + 1)
        new_inds = np.concatenate(new_inds, 0)
        return new_inds, new_lens

    def _sort_by_score(self, im_inds, scores):
        """
        We'll sort everything scorewise from Hi->low, BUT we need to keep images together
        and sort LSTM from l
        :param im_inds: Which im we're on
        :param scores: Goodness ranging between [0, 1]. Higher numbers come FIRST
        :return: Permutation to put everything in the right order for the LSTM
                 Inverse permutation
                 Lengths for the TxB packed sequence.
        """
        num_im = im_inds[-1] + 1
        # this is just to create a tensor of same dtype on same device
        #rois_per_image = scores.new((num_im,))
        rois_per_image = scores.new(
                        torch.Size((num_im.cpu().numpy().astype(np.int32)[0],))
                                    )

        lengths = []
        for i, s, e in enumerate_by_image(im_inds):
            rois_per_image[i] = 2 * (s - e) * num_im + i
            lengths.append(e - s)

        lengths = sorted(lengths, reverse=True)
        # B x T , num images by num rois
        # 2 x 256. eg.
        # inds is the T x B sequences indexes.
        # e.g. inds[:,0] is first batch image's sequence for lstm processing
        # ls_transposed is T length list with each element numbered as B
        inds, ls_transposed = self.transpose_packed_sequence_inds(lengths)  # move it to TxB form
        inds = torch.LongTensor(inds).cuda()


        # sort by confidence in the range (0,1)
        # but more importantly by longest img
        # very hacky since rois_per_image is defined
        # as arbitrarily large negative value, different for different image
        # by increment of 1 in value

        # then - 2*rois_per_im will again make it positive
        # thus allows us to sort scores given a same image
        roi_order = scores - 2 * rois_per_image[im_inds.long().squeeze()]
        _, perm = torch.sort(roi_order, 0, descending=True)

        # this perm makes the ordering in the packed sequence format
        perm = perm[inds]

        # reverse sort the perm indexes, to get back to pre-sort indexing
        _, inv_perm = torch.sort(perm)

        # return perm in packed seq format, inv_perm to get original ordering
        # note original ordering is NOT in packed seq format
        # ls_transposed: T length list with each element numbered as B

        # convert ls_transposed to tensor format
        ls_transposed = torch.Tensor(ls_transposed).long()

        return perm, inv_perm, ls_transposed

    def center_size(self, boxes):
        """ Convert prior_boxes to (cx, cy, w, h)
        representation for comparison to center-size form ground truth data.
        Args:
            boxes: (tensor) point_form boxes
        Return:
            boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
        """
        wh = boxes[:, 2:] - boxes[:, :2] + 1.0

        if isinstance(boxes, np.ndarray):
            return np.column_stack((boxes[:, :2] + 0.5 * wh, wh))
        return torch.cat((boxes[:, :2] + 0.5 * wh, wh), 1)

    def sort(self, batch_idx, confidence, box_priors):
        """
        :param batch_idx: tensor with what index we're on
        :param confidence: tensor with confidences between [0,1)
        :param boxes: tensor with (x1, y1, x2, y2)
        :return: Permutation, inverse permutation, and the lengths transposed (same as _sort_by_score)
        """
        cxcywh = self.center_size(box_priors)

        if self.order == 'size':
            sizes = cxcywh[:,2] * cxcywh[:, 3]
            # sizes = (box_priors[:, 2] - box_priors[:, 0] + 1) * (box_priors[:, 3] - box_priors[:, 1] + 1)
            assert sizes.min() > 0.0
            scores = sizes / (sizes.max() + 1)
        elif self.order == 'confidence':
            scores = confidence
        elif self.order == 'random':
            scores = torch.FloatTensor(np.random.rand(batch_idx.size(0))).cuda()
        elif self.order == 'leftright':
            centers = cxcywh[:,0]
            scores = centers / (centers.max() + 1)
        else:
            raise ValueError("invalid mode {}".format(self.order))
        return self._sort_by_score(batch_idx, scores)