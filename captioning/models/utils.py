import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from torch.nn import Parameter
import math

EPS = 1e-6

def repeat_tensors(n, x):
    """
    For a tensor of size Bx..., we repeat it n times, and make it Bnx...
    For collections, do nested repeat
    """
    if torch.is_tensor(x):
        x = x.unsqueeze(1) # Bx1x...
        x = x.expand(-1, n, *([-1]*len(x.shape[2:]))) # Bxnx...
        x = x.reshape(x.shape[0]*n, *x.shape[2:]) # Bnx...
    elif type(x) is list or type(x) is tuple:
        x = [repeat_tensors(n, _) for _ in x]
    return x


def split_tensors(n, x):
    if torch.is_tensor(x):
        assert x.shape[0] % n == 0
        x = x.reshape(x.shape[0] // n, n, *x.shape[1:]).unbind(1)
    elif type(x) is list or type(x) is tuple:
        x = [split_tensors(n, _) for _ in x]
    elif x is None:
        x = [None] * n
    return x

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.shape[-2] + tensor.shape[-1]))
        tensor.data.uniform_(-stdv, stdv)

class GaussianNet(nn.Module):
    def __init__(self,
                opt,
                in_channels=1000,
                out_channels=100,
                dim=4,
                kernel_size=10
                ):
        super(GaussianNet, self).__init__()
        self.in_channels = in_channels  # in_channels=512
        self.out_channels = out_channels  # out_channel=512
        self.dim = dim  # dim=2
        self.kernel_size = kernel_size  # kth gaussian kernel k=4
        self.opt = opt
        self.mu = Parameter(torch.Tensor(kernel_size, dim))
        self.sigma = Parameter(torch.Tensor(kernel_size, dim))
        self.lin = nn.Linear(out_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.mu)
        glorot(self.sigma)

    def forward(self, x, adj, pseudo):
        # print(pseudo.size())
        pseudo_flatten = pseudo.reshape(-1, pseudo.size(-1))
        batch_size = adj.shape[0]
        num_node = adj.shape[1]
        (E, D), K = pseudo_flatten.size(), self.mu.shape[0]

        gaussian = -0.5 * (pseudo_flatten.unsqueeze(1) - self.mu.unsqueeze(0)) ** 2
        gaussian = gaussian / (EPS + self.sigma.unsqueeze(0) ** 2)
        gaussian = torch.exp(gaussian.sum(dim=-1, keepdim=True))  # [E, K, 1]
        gaussian_tensor = gaussian.reshape(batch_size, num_node, num_node, K, -1)
        # clip
        adj_gaussian_tensor = gaussian_tensor.squeeze() * adj.unsqueeze(-1) # [bs, n, n, 8]
        x = self.lin(x.reshape(batch_size, num_node, self.kernel_size, -1)) # [bs, n, 8, 125]
        res = torch.matmul(adj_gaussian_tensor.permute(0, 3, 1, 2).contiguous().unsqueeze(2), x.permute(0, 2, 3, 1).contiguous().unsqueeze(-1))

        return res.reshape(batch_size, -1 ,num_node)


class Feature2Adj(nn.Module):
    def __init__(self, input_dims=2048, out_dims=512):
        super(Feature2Adj, self).__init__()
        self.linear = nn.Linear(input_dims, out_dims)

    def forward(self, image_features):
        # transform image features to latent space
        batch_size = image_features.shape[0]
        latent_z = self.linear(image_features)
        adj = torch.bmm(latent_z, latent_z.permute(0, 2, 1))
        return adj


def self_attention(query, key, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.shape[-1]
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return p_attn

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def sort_pack_padded_sequence(input, lengths):
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(input[indices], sorted_lengths, batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0,len(indices)).type_as(inv_ix)
    return tmp, inv_ix

def pad_unsort_packed_sequence(input, inv_ix):
    tmp, _ = pad_packed_sequence(input, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp

def pack_wrapper(module, att_feats, att_masks):
    if att_masks is not None:
        packed, inv_ix = sort_pack_padded_sequence(att_feats, att_masks.data.long().sum(1))
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)
    else:
        return module(att_feats)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        a = self.a_2 * (x - mean)
        b = std + self.eps
        c = a/b
        # print(std)
        return  c  + self.b_2
