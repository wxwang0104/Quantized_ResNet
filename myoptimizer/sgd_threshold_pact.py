import torch
from torch.optim.optimizer import Optimizer, required
from torch.autograd import Variable
import math
import torch.nn.functional as F
import copy
import numpy as np

class SGD_THRESHOLD_PACT(Optimizer):

    def __init__(self, params, significant_bit=2, bit_threshold=20, lr=0.1, param_list=None, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD_THRESHOLD_PACT, self).__init__(params, defaults)
        self.significant_bit = significant_bit
        self.bit_threshold = bit_threshold
        self.lr = lr
        self.bp = copy.deepcopy(self.param_groups)
        self.param_list = param_list
        
        self.weight_thres_lr = 0
        self.weight_thres_regu = 1e-2

    def __setstate__(self, state):
        super(SGD_THRESHOLD_PACT, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)



    def step(self, weight_thres, closure=None):
        """Performs a single optimization step.

        Arguments:
            weight_thres: a weight threshold in hysteresis loop to stabilize the training curve
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        cnt = 0
        weight_thres_grad = 0

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            bp1 = self.bp[cnt]
            cnt += 1
            inner_cnt = 0
            for p in group['params']:
                bp2 = bp1['params'][inner_cnt]
                inner_cnt += 1
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                bp3 = bp2.clone()
                bp2.data.add_(-group['lr'],d_p)
                # weight from negative (in last iter) to positive (in current iter)
                ind_neg2pos = (bp3.data<0) & (bp2.data>0)
                # weight from positive (in last iter) to negative (in current iter)
                ind_pos2neg = (bp3.data>0) & (bp2.data<0)
                # weight that keeps the same sign
                ind_otherwise = ((bp3.data>0) & (bp2.data>0)) | ((bp3.data<0) & (bp2.data<0)) 

                if self.param_list[inner_cnt-1][0:2]=='co' and inner_cnt is not 1 and inner_cnt is not len(self.param_list)-1 and inner_cnt is not len(self.param_list):
                    # Quantize the weight parameter
                    p.data.mul_(0).add_(bp2.data)
 
                    # Calculate scaling_factor
                    scaling_factor = torch.sum(torch.abs(p.data))
                    num = 1
                    for a in p.data.size():
                        num *= a
                    scaling_factor /= (1.0*num)
                
                    # Sign function
                    p.data[(p.data>0) & ind_otherwise] = 1
                    p.data[(p.data<0) & ind_otherwise] = -1
                    p.data[(p.data>=weight_thres) & ind_neg2pos] = 1
                    p.data[(p.data<=-weight_thres) & ind_pos2neg] = -1
                    p.data.mul_(scaling_factor)
                else:
                    p.data.mul_(0).add_(bp2.data)

                #TODO: consider adding this weight threshold into forward training, not implemented yet, weight_thres_lr is 0 now
                weight_thres_grad += torch.nonzero(ind_neg2pos).size(0)
                weight_thres_grad += torch.nonzero(ind_pos2neg).size(0)
                weight_thres_grad += self.weight_thres_regu * 2 * weight_thres
                weight_thres += -self.weight_thres_lr * weight_thres_grad

        return weight_thres


    def set_params(self, target):
        loss = None
        cnt = 0
        for group in self.param_groups:
            t1 = target[cnt]['params']
            cnt+=1
            cnt_inner = 0
            for p in group['params']:
                t2 = t1[cnt_inner]
                cnt_inner += 1
                p.data.mul_(0).add_(t2.data)




