'''
Ref: https://github.com/allegro/allRank/blob/master/allrank/models/losses/approxNDCG.py
'''

import torch
import torch.nn.functional as F
DEFAULT_EPS = 1e-10
PADDED_Y_VALUE = -1

# def approxNDCGLoss(y_pred, y_true, eps=DEFAULT_EPS, padded_value_indicator=PADDED_Y_VALUE, alpha=1.):
#     """
#     Loss based on approximate NDCG introduced in "A General Approximation Framework for Direct Optimization of
#     Information Retrieval Measures". Please note that this method does not implement any kind of truncation.
#     :param y_pred: predictions from the model, shape [batch_size, slate_length]
#     :param y_true: ground truth labels, shape [batch_size, slate_length]
#     :param eps: epsilon value, used for numerical stability
#     :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
#     :param alpha: score difference weight used in the sigmoid function
#     :return: loss value, a torch.Tensor
#     """
#     device = y_pred.device
#     y_pred = y_pred.clone()
#     y_true = y_true.clone()

#     padded_mask = y_true == padded_value_indicator
#     y_pred[padded_mask] = float("-inf")
#     y_true[padded_mask] = float("-inf")

#     # Here we sort the true and predicted relevancy scores.
#     y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
#     y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

#     # After sorting, we can mask out the pairs of indices (i, j) containing index of a padded element.
#     true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
#     true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
#     padded_pairs_mask = torch.isfinite(true_diffs)
#     padded_pairs_mask.diagonal(dim1=-2, dim2=-1).zero_()

#     # Here we clamp the -infs to get correct gains and ideal DCGs (maxDCGs)
#     true_sorted_by_preds.clamp_(min=0.)
#     y_true_sorted.clamp_(min=0.)

#     # Here we find the gains, discounts and ideal DCGs per slate.
#     pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
#     D = torch.log2(1. + pos_idxs.float())[None, :]
#     maxDCGs = torch.sum((torch.pow(2, y_true_sorted) - 1) / D, dim=-1).clamp(min=eps)
#     G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

#     # Here we approximate the ranking positions according to Eqs 19-20 and later approximate NDCG (Eq 21)
#     scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :])

#     scores_diffs[~padded_pairs_mask] = 0.
#     approx_pos = 1. + torch.sum(padded_pairs_mask.float() * (torch.sigmoid(-alpha * scores_diffs).clamp(min=eps)), dim=-1)
#     approx_D = torch.log2(1. + approx_pos)
#     approx_NDCG = torch.sum((G / approx_D), dim=-1)

#     return -torch.mean(approx_NDCG)

# def listNet(y_pred, y_true, eps=DEFAULT_EPS, padded_value_indicator=PADDED_Y_VALUE):
#     """
#     ListNet loss introduced in "Learning to Rank: From Pairwise Approach to Listwise Approach".
#     :param y_pred: predictions from the model, shape [batch_size, slate_length]
#     :param y_true: ground truth labels, shape [batch_size, slate_length]
#     :param eps: epsilon value, used for numerical stability
#     :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
#     :return: loss value, a torch.Tensor
#     """
#     y_pred = y_pred.clone()
#     y_true = y_true.clone()

#     mask = y_true == padded_value_indicator
#     y_pred[mask] = float('-inf')
#     y_true[mask] = float('-inf')

#     preds_smax = F.softmax(y_pred, dim=1)
#     true_smax = F.softmax(y_true, dim=1)

#     preds_smax = preds_smax + eps
#     preds_log = torch.log(preds_smax)

#     return torch.mean(-torch.sum(true_smax * preds_log, dim=1))

def listNet(y_pred, y_true):
    return torch.sum(-torch.sum(F.softmax(y_true, dim=1) * F.log_softmax(y_pred, dim=1), dim=1))

def get_pairwise_comp_probs(batch_preds, batch_std_labels, sigma=None):
    '''
    Get the predicted and standard probabilities p_ij which denotes d_i beats d_j
    @param batch_preds:
    @param batch_std_labels:
    @param sigma:
    @return:
    '''
    # computing pairwise differences w.r.t. predictions, i.e., s_i - s_j
    batch_s_ij = torch.unsqueeze(batch_preds, dim=2) - torch.unsqueeze(batch_preds, dim=1)
    batch_p_ij = torch.sigmoid(sigma * batch_s_ij)

    # computing pairwise differences w.r.t. standard labels, i.e., S_{ij}
    batch_std_diffs = torch.unsqueeze(batch_std_labels, dim=2) - torch.unsqueeze(batch_std_labels, dim=1)
    # ensuring S_{ij} \in {-1, 0, 1}
    batch_Sij = torch.clamp(batch_std_diffs, min=-1.0, max=1.0)
    batch_std_p_ij = 0.5 * (1.0 + batch_Sij)

    return batch_p_ij, batch_std_p_ij

def rankNet(y_pred, y_true):
    batch_p_ij, batch_std_p_ij = get_pairwise_comp_probs(batch_preds=y_pred, batch_std_labels=y_true,
                                                            sigma=1.0)
    _batch_loss = F.binary_cross_entropy(input=torch.triu(batch_p_ij, diagonal=1),
                                            target=torch.triu(batch_std_p_ij, diagonal=1), reduction='none')
    batch_loss = torch.sum(torch.sum(_batch_loss, dim=(2, 1)))
    return batch_loss

class Robust_Sigmoid(torch.autograd.Function):
    ''' Aiming for a stable sigmoid operator with specified sigma '''

    @staticmethod
    def forward(ctx, input, sigma=1.0, device='cpu'):
        '''
        :param ctx:
        :param input: the input tensor
        :param sigma: the scaling constant
        :return:
        '''
        x = input if 1.0==sigma else sigma * input

        torch_half = torch.tensor([0.5], dtype=torch.float, device=device)
        sigmoid_x_pos = torch.where(input>0, 1./(1. + torch.exp(-x)), torch_half)

        exp_x = torch.exp(x)
        sigmoid_x = torch.where(input<0, exp_x/(1.+exp_x), sigmoid_x_pos)

        grad = sigmoid_x * (1. - sigmoid_x) if 1.0==sigma else sigma * sigmoid_x * (1. - sigmoid_x)
        ctx.save_for_backward(grad)

        return sigmoid_x

    @staticmethod
    def backward(ctx, grad_output):
        '''
        :param ctx:
        :param grad_output: backpropagated gradients from upper module(s)
        :return:
        '''
        grad = ctx.saved_tensors[0]

        bg = grad_output * grad # chain rule

        return bg, None, None
robust_sigmoid = Robust_Sigmoid.apply

def torch_dcg_at_k(batch_rankings, cutoff=None, device='cpu'):
    '''
    ICML-nDCG, which places stronger emphasis on retrieving relevant documents
    :param batch_rankings: [batch_size, ranking_size] rankings of labels (either standard or predicted by a system)
    :param cutoff: the cutoff position
    :param label_type: either the case of multi-level relevance or the case of listwise int-value, e.g., MQ2007-list
    :return: [batch_size, 1] cumulative gains for each rank position
    '''
    if cutoff is None: # using whole list
        cutoff = batch_rankings.size(1)

    batch_numerators = torch.pow(2.0, batch_rankings[:, 0:cutoff]) - 1.0    # MultiLabel, the common case with multi-level labels
    # batch_numerators = batch_rankings[:, 0:cutoff]      # Permutation, the case like listwise ltr_adhoc, where the relevance is labeled as (n-rank_position)

    # no expanding should also be OK due to the default broadcasting
    batch_discounts = torch.log2(torch.arange(cutoff, dtype=torch.float, device=device).expand_as(batch_numerators) + 2.0)
    batch_dcg_at_k = torch.sum(batch_numerators/batch_discounts, dim=1, keepdim=True)
    return batch_dcg_at_k


def get_approx_ranks(input, alpha=10, device=None):
    ''' get approximated rank positions: Equation-11 in the paper'''
    batch_pred_diffs = torch.unsqueeze(input, dim=2) - torch.unsqueeze(input, dim=1)  # computing pairwise differences, i.e., Sij or Sxy    

    batch_indicators = robust_sigmoid(torch.transpose(batch_pred_diffs, dim0=1, dim1=2), alpha, device) # using {-1.0*} may lead to a poor performance when compared with the above way;

    batch_hat_pis = torch.sum(batch_indicators, dim=2) + 0.5  # get approximated rank positions, i.e., hat_pi(x)

    return batch_hat_pis

def approxNDCGLoss(y_pred=None, y_true=None, alpha=1):     # alpha=10

    device = y_pred.device
    batch_ideal_rankings, batch_ideal_desc_inds = torch.sort(y_true, dim=1, descending=True)
    batch_preds = torch.gather(y_pred, dim=1, index=batch_ideal_desc_inds)

    batch_hat_pis = get_approx_ranks(batch_preds, alpha=alpha, device=device)

    # ideal dcg given optimally ordered labels
    batch_idcgs = torch_dcg_at_k(batch_rankings=batch_ideal_rankings, cutoff=None, device=device)

    batch_gains = torch.pow(2.0, batch_ideal_rankings) - 1.0    # MultiLabel
    # batch_gains = batch_ideal_rankings      # Permutation

    batch_dcg = torch.sum(torch.div(batch_gains, torch.log2(batch_hat_pis + 1)), dim=1)
    batch_approx_nDCG = torch.div(batch_dcg, batch_idcgs)

    batch_loss = -torch.sum(batch_approx_nDCG)
    # batch_loss = -torch.mean(batch_approx_nDCG)
    return batch_loss