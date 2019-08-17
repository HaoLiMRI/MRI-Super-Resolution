import math
import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    """
    AdamW, Adam with decoupled Weight decay regularization, "2017. Decoupled Weight Decay Regularization 
    (https://arxiv.org/pdf/1711.05101.pdf), implements actual Adam with weight decay regularization algorithm which 
    decouples "learning rate" and "weight decay". It means this is the "Adam with actual weight decay regularization".
    - The normal existing Adam optimizer in Pyotrch has coupling between "learning rate" and "weight decay", 
    that somehow cancels the "regularization" effect from "weight decay".
    - AdamW implementation is straightforward and does not differ much from existing Adam implementation for PyTorch, 
    except that it separates weight decaying from batch gradient calculations,

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_

    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        #super(AdamW, self).__init__(params, defaults)
        super().__init__(params, defaults)

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
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # The 1st change compared to existing Adam optimizer from Pytorch lib
                # Delete the following code:
                # if group['weight_decay'] != 0:
                #     # It means "grad + group['weight_decay']*p.data"
                #     grad.add_(group['weight_decay'], p.data)
                # in existing Adam optimizer from Pytorch lib

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                
                # # The 2nd change compared to existing Adam optimizer from Pytorch lib
                # Change the code 
                # # It means "p.data + (-step_size)*(exp_avg/denom)"
                # p.data.addcdiv_(-step_size, exp_avg, denom)
                # in existing Adam optimizer in Pytorch lib to the following code

                # It means "p.data(1 - group['weight_decay']) + (-step_size)*(exp_avg/denom)", on the other word, is,
                # "p.data + (-step_size)*(exp_avg/denom) - group['weight_decay']*p.data"
                p.data.mul_(1 - group['weight_decay']).addcdiv_(-step_size, exp_avg, denom)

        return loss





"""
Example of using adamw optimizer
"""
"""
    batch_size = 32
    epoch_size = 1024
    model = resnet()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    for epoch in range(100):
        optimizer.step()
        train_for_every_batch(...)
            ...
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer.batch_step()
        validate(...)
"""
