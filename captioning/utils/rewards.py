from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
from collections import OrderedDict
import torch

import sys
try:
    sys.path.append("cider")
    from pyciderevalcap.ciderD.ciderD import CiderD
    from pyciderevalcap.cider.cider import Cider
    sys.path.append("coco-caption")
    from pycocoevalcap.bleu.bleu import Bleu
    from pyciderevalcap.NKRE_D.nkpe_D import Nkpe_D
except:
    print('cider or coco-caption missing')

CiderD_scorer = None
Cider_scorer = None
Bleu_scorer = None
Nkpe_scorer = None
#CiderD_scorer = CiderD(df='corpus')

def init_scorer(cached_tokens):
    global CiderD_scorer
    CiderD_scorer = CiderD_scorer or CiderD(df=cached_tokens)
    global Cider_scorer
    Cider_scorer = Cider_scorer or Cider(df=cached_tokens)
    global Bleu_scorer
    Bleu_scorer = Bleu_scorer or Bleu(4)
    global Nkpe_scorer
    Nkpe_scorer = Nkpe_scorer or Nkpe_D()

def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0 :
            break
    return out.strip()

def get_self_critical_reward(greedy_res, data_gts, gen_result, opt):
    batch_size = len(data_gts) 
    gen_result_size = gen_result.shape[0]
    seq_per_img = gen_result_size // len(data_gts) # gen_result_size  = batch_size * seq_per_img
    assert greedy_res.shape[0] == batch_size

    res = OrderedDict()
    gen_result = gen_result.data.cpu().numpy()
    greedy_res = greedy_res.data.cpu().numpy()
    for i in range(gen_result_size):
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        res[gen_result_size + i] = [array_to_str(greedy_res[i])]

    gts = OrderedDict()
    for i in range(len(data_gts)):
        gts[i] = [array_to_str(data_gts[i][j]) for j in range(len(data_gts[i]))]

    res_ = [{'image_id':i, 'caption': res[i]} for i in range(len(res))]
    res__ = {i: res[i] for i in range(len(res_))}
    gts_ = {i: gts[i // seq_per_img] for i in range(gen_result_size)}
    gts_.update({i+gen_result_size: gts[i] for i in range(batch_size)})
    if opt.cider_reward_weight > 0:
        _, cider_scores = CiderD_scorer.compute_score(gts_, res_)
        print('Cider scores:', _)
    else:
        cider_scores = 0
    if opt.nkpe_reward_weight > 0:
        _, nkpe_scores = Nkpe_scorer.compute_score(gts_, res_)
        print('Nkpe scores:', _)
    else:
        nkpe_scores = 0
    if opt.bleu_reward_weight > 0:
        _, bleu_scores = Bleu_scorer.compute_score(gts_, res__)
        bleu_scores = np.array(bleu_scores[3])
        print('Bleu scores:', _[3])
    else:
        bleu_scores = 0
    scores = opt.cider_reward_weight * cider_scores + opt.bleu_reward_weight * bleu_scores + opt.nkpe_reward_weight * nkpe_scores
    # scores = cider_scores * nkpe_scores * 3
    scores = scores[:gen_result_size].reshape(batch_size, seq_per_img) - scores[-batch_size:][:, np.newaxis]
    scores = scores.reshape(gen_result_size)

    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)

    return rewards


def get_self_critical_reward_2(data_gts, gen_result, monte_carlo_count):
    global Nkpe_scorer
    Nkpe_scorer = Nkpe_scorer or Nkpe_D()
    # reward = np.zeros((gen_result.shape[0], 1))

    gen_result_size = gen_result.shape[0]
    seq_per_img = gen_result_size // len(data_gts) // monte_carlo_count   # gen_result_size  = batch_size * seq_per_img
    batch_size = gen_result_size // monte_carlo_count
    res = OrderedDict()
    gen_result = gen_result.data.cpu().numpy()
    for i in range(gen_result_size):
        # print(gen_result[i])
        res[i] = [array_to_str(gen_result[i])]

    gts = OrderedDict()
    for i in range(len(data_gts)):
        gts[i] = [array_to_str(data_gts[i][j]) for j in range(len(data_gts[i]))]

    res_ = [{'image_id': i, 'caption': res[i]} for i in range(len(res))]
    gts_ = {int(gen_result_size//monte_carlo_count) * i + j: gts[j // seq_per_img] for i in range(monte_carlo_count) for j in range(int(gen_result_size//monte_carlo_count))}

    _, nkpe_scores = Nkpe_scorer.compute_score(gts_, res_)
    # print('Nkpe scores:', _)
    reward = torch.from_numpy(nkpe_scores).cuda()
    reward = reward.view(batch_size, monte_carlo_count, -1).sum(1)
    return reward


def get_self_critical_reward_3(greedy_res, data_gts, gen_result, current_generated, opt, monte_carlo_count=2):
    batch_size = len(data_gts)
    gen_result_size = gen_result.shape[0]
    seq_length = gen_result.shape[1]
    seq_per_img = gen_result_size // len(data_gts) # gen_result_size  = batch_size * seq_per_img
    assert greedy_res.shape[0] == batch_size
    current_generated_size = current_generated.size(0)
    t = current_generated_size // gen_result_size

    res = OrderedDict()
    gen_result = gen_result.data.cpu().numpy()
    greedy_res = greedy_res.data.cpu().numpy()

    for i in range(gen_result_size):
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        res[gen_result_size + i] = [array_to_str(greedy_res[i])]


    cur_res = OrderedDict()
    current_generated = current_generated.data.cpu().numpy()
    for i in range(current_generated_size):
        cur_res[i] = [array_to_str(current_generated[i])]

    # gen_result = gen_result.data.cpu().numpy()
    # greedy_res = greedy_res.data.cpu().numpy()

    gts = OrderedDict()
    for i in range(len(data_gts)):
        gts[i] = [array_to_str(data_gts[i][j]) for j in range(len(data_gts[i]))]

    cur_res_ = [{'image_id':i, 'caption': cur_res[i]} for i in range(len(cur_res))]
    gts_ = {i: gts[i // seq_per_img] for i in range(gen_result_size)}

    gts_cur_ = {j*gen_result_size + i: gts_[i] for j in range(t) for i in range(len(gts_)) }
    # start = time.time()
    _, nkpe_scores = Nkpe_scorer.compute_score(gts_cur_, cur_res_)
    # print('scores time {}'.format(time.time() - start))
    print('Nkpe scores:', _)
    nkpe_scores_list = np.split(nkpe_scores, t/monte_carlo_count, axis=0)
    result = np.zeros((gen_result_size, seq_length), dtype=nkpe_scores.dtype)
    for t, item in enumerate(nkpe_scores_list):
        item_list = np.split(item, monte_carlo_count, axis=0)
        res_scores = np.zeros((gen_result_size,), dtype=nkpe_scores.dtype)
        for item_i in item_list:
            res_scores += item_i
        result[:,t*6: t*6+6] = np.repeat((res_scores/monte_carlo_count).reshape(-1,1),6, axis=1)


    # scores = scores[:gen_result_size].reshape(batch_size, seq_per_img) - scores[-batch_size:][:, np.newaxis]
    # scores = scores.reshape(gen_result_size)
    #
    # rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)
    rewards = result
    return rewards


def get_scores(data_gts, gen_result, opt):
    batch_size = gen_result.size(0)# batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data_gts)

    res = OrderedDict()
    
    gen_result = gen_result.data.cpu().numpy()
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]

    gts = OrderedDict()
    for i in range(len(data_gts)):
        gts[i] = [array_to_str(data_gts[i][j]) for j in range(len(data_gts[i]))]

    res_ = [{'image_id':i, 'caption': res[i]} for i in range(batch_size)]
    res__ = {i: res[i] for i in range(batch_size)}
    gts = {i: gts[i // seq_per_img] for i in range(batch_size)}
    if opt.cider_reward_weight > 0:
        _, cider_scores = CiderD_scorer.compute_score(gts, res_)
        print('Cider scores:', _)
    else:
        cider_scores = 0
    if opt.bleu_reward_weight > 0:
        _, bleu_scores = Bleu_scorer.compute_score(gts, res__)
        bleu_scores = np.array(bleu_scores[3])
        print('Bleu scores:', _[3])
    else:
        bleu_scores = 0

    scores = opt.cider_reward_weight * cider_scores + opt.bleu_reward_weight * bleu_scores

    return scores

def get_self_cider_scores(data_gts, gen_result, opt):
    batch_size = gen_result.size(0)# batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data_gts)

    res = []
    
    gen_result = gen_result.data.cpu().numpy()
    for i in range(batch_size):
        res.append(array_to_str(gen_result[i]))

    scores = []
    for i in range(len(data_gts)):
        tmp = Cider_scorer.my_self_cider([res[i*seq_per_img:(i+1)*seq_per_img]])
        def get_div(eigvals):
            eigvals = np.clip(eigvals, 0, None)
            return -np.log(np.sqrt(eigvals[-1]) / (np.sqrt(eigvals).sum())) / np.log(len(eigvals))
        scores.append(get_div(np.linalg.eigvalsh(tmp[0]/10)))

    scores = np.array(scores)

    return scores