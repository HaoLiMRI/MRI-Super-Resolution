"""
This is the pytorch version lookahead optimization algorithms(optimizer), which could be combined with any other optimizer, e.g. SGD, adam, etc.
For more details of lookahead optimization algorithm, please read:
2019. Lookahead Optimizer: k steps forward, 1 step back. https://arxiv.org/abs/1907.08610
Acoording to the authors of lookahead optimizer, lookahead optimization algorithm could converge very fast to reach the global min/max without too much/complicated tunning,
it is the best optimization algorithm so far(20199728) for deep learning. They claim "We empirically demonstrate Lookahead can significantly improve the performance of SGD 
and Adam, even with their default hyperparameter settings on ImageNet, CIFAR-10/100, neural machine translation, and Penn Treebank."
"""

from collections import defaultdict
from itertools import chain
from torch.optim import Optimizer
import torch
import warnings


class Lookahead(Optimizer):
    """
    Lookahead Optimizer:
    Lookahead update iteratively two sets of weights, which are 
        1) outer slow weights phi, φ, and 
        2) inner fast weights theta, θ. 
    The outer slow weights phi update once as inner fast weights theta update k times. The fast weights update can be achieved by using any optimizer, e.g. SGD, adam, etc. 

    After the inner fast weights update k times by using particular optimizer, e.g. SGD, lookahead update outer slow weights once by using weighted (θ − φ) with using
    same direction as the last update of fast weights. Every time slow weights update, the current slow weights will be used as fast weights. See "algorithm 1" in the original paper
    for more detail.
    使用优化器 A 经过 k 次内部优化器更新后，Lookahead 通过在权重空间 θ − φ 中执行线性插值的方式更新 slow weights，方向为最后一个 fast weights。
    slow weights 每更新一次，fast weights 将被重置为目前的 slow weights 值。Lookahead 的伪代码见"algorithm 1" in the original paper.
    """
    def __init__(self, optimizer, k=5, alpha=0.5):
        """
        optimizer: selected optimizer to update fast weights, e.g. SGD
        k: how many steps to be used to perform "inner fast weights update" before performing "outer slow weights update" once, e.g. k = 5
        alpha: weight to be used to update outer slow weights once by using φ = φ + alpha*(θ − φ)
        """
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group["counter"] = 0
    
    def update(self, group):
        """
        Update the slow weights by using φ = φ + alpha*(θ − φ)
        group: 
        """
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha # φ = φ + alpha*(θ − φ)
            fast.data.copy_(slow) # after everytime update of slow weights, reset the fast weights by using current slow weights
    
    def update_lookahead(self):
        """
        call update(self, group) to update the slow weights
        """
        for group in self.param_groups:
            self.update(group)

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update(group)
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
        return loss

    def state_dict(self):
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "fast_state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict["param_groups"],
        }
        fast_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state

    def add_param_group(self, param_group):
        param_group["counter"] = 0
        self.optimizer.add_param_group(param_group)
