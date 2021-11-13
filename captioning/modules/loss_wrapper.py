import torch
from . import losses
from ..utils.rewards import init_scorer, get_self_critical_reward, get_self_critical_reward_3
import time
class LossWrapper(torch.nn.Module):
    def __init__(self, model, opt):
        super(LossWrapper, self).__init__()
        self.opt = opt
        self.model = model
        if opt.label_smoothing > 0:
            self.crit = losses.LabelSmoothing(smoothing=opt.label_smoothing)

        elif opt.add_news_type > 0:
            self.crit = losses.LanguageModelWithNewsCriterion()
        else:
            self.crit = losses.LanguageModelCriterion()
        self.rl_crit = losses.RewardCriterion()
        self.struc_crit = losses.StructureLosses(opt)

    def forward(self, fc_feats, att_feats, spacial_feats, labels, masks, att_masks,  news_type, gts, gt_indices,
                sc_flag, struc_flag):
        opt = self.opt
        loss_lang = None
        loss_type = None
        out = {}
        # print(struc_flag, opt.rollout_flag)
        if struc_flag:
            if opt.structure_loss_weight < 1:
                lm_loss = self.crit(self.model(fc_feats, att_feats, labels[..., :-1], att_masks), labels[..., 1:], masks[..., 1:])
            else:
                lm_loss = torch.tensor(0).type_as(fc_feats)
            if opt.structure_loss_weight > 0:
                gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks,
                    opt={'sample_method':opt.train_sample_method,
                        'beam_size':opt.train_beam_size,
                        'output_logsoftmax': opt.struc_use_logsoftmax or opt.structure_loss_type == 'softmax_margin'\
                            or not 'margin' in opt.structure_loss_type,
                        'sample_n': opt.train_sample_n},
                    mode='sample')
                gts = [gts[_] for _ in gt_indices.tolist()]
                struc_loss = self.struc_crit(sample_logprobs, gen_result, gts)
            else:
                struc_loss = {'loss': torch.tensor(0).type_as(fc_feats),
                              'reward': torch.tensor(0).type_as(fc_feats)}
            loss = (1-opt.structure_loss_weight) * lm_loss + opt.structure_loss_weight * struc_loss['loss']
            out['lm_loss'] = lm_loss
            out['struc_loss'] = struc_loss['loss']
            out['reward'] = struc_loss['reward']
        elif not sc_flag and  opt.rollout_flag != 1:
            loss, loss_lang, loss_type = self.crit(self.model(fc_feats, att_feats, spacial_feats, labels[..., :-1], att_masks, news_type), news_type, labels[..., 1:], masks[..., 1:])
        elif opt.rollout_flag == 1:
            self.model.eval()
            gts = [gts[_] for _ in gt_indices.tolist()]
            with torch.no_grad():
                greedy_res, _, _ = self.model(fc_feats, att_feats, spacial_feats, att_masks,
                                              mode='sample',
                                              opt={'sample_method': opt.sc_sample_method,
                                                   'beam_size': opt.sc_beam_size})
            self.model.train()
            # start = time.time()
            gen_result, sample_logprobs, _, current_result = self.model(fc_feats, att_feats, spacial_feats, gts, att_masks,
                                                        opt={'sample_method': opt.train_sample_method,
                                                             'beam_size': opt.train_beam_size,
                                                             'sample_n': opt.train_sample_n},
                                                        mode='reward_sample')
            # print('spend {} s'.format(time.time() - start))
            gts = [gts[_] for _ in gt_indices.tolist()]
            reward = get_self_critical_reward_3(greedy_res, gts, gen_result,current_result, self.opt)
            reward = torch.from_numpy(reward).to(sample_logprobs)
            loss = self.rl_crit(sample_logprobs, gen_result.data, reward)
            out['reward'] = reward[:,:].mean()
        else:
            self.model.eval()
            with torch.no_grad():
                greedy_res, _, _ = self.model(fc_feats, att_feats, spacial_feats, att_masks,
                    news_type,
                    mode='sample',
                    opt={'sample_method': opt.sc_sample_method,
                         'beam_size': opt.sc_beam_size})
            self.model.train()
            gen_result, sample_logprobs,_ = self.model(fc_feats, att_feats, spacial_feats, att_masks, news_type,
                    opt={'sample_method':opt.train_sample_method,
                        'beam_size':opt.train_beam_size,
                        'sample_n': opt.train_sample_n},
                    mode='sample')
            gts = [gts[_] for _ in gt_indices.tolist()]
            reward = get_self_critical_reward(greedy_res, gts, gen_result, self.opt)
            reward = torch.from_numpy(reward).to(sample_logprobs)
            loss = self.rl_crit(sample_logprobs, gen_result.data, reward)
            out['reward'] = reward[:,0].mean()
        out['loss'] = loss

        out['loss_lang'] = loss_lang
        out['loss_type'] = loss_type
        return out
