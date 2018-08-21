import torch
from torch.optim.optimizer import Optimizer, required
from torch.autograd import Variable
import math
import torch.nn.functional as F
import copy
import numpy as np

class SGD_CUSTOMIZED(Optimizer):

    def __init__(self, params, significant_bit=2, bit_threshold=20, lr=0.1, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD_CUSTOMIZED, self).__init__(params, defaults)
        self.significant_bit = significant_bit
        self.bit_threshold = bit_threshold
        self.lr = lr

    def __setstate__(self, state):
        super(SGD_CUSTOMIZED, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)



    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            for p in group['params']:
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
                # Assign the continuous params to p
                p.data.add_(-group['lr'],d_p)
        return loss


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




