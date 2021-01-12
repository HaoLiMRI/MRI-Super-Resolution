from torch.utils.data import DataLoader
from importlib import import_module
import os
import numpy as np
import h5py
import torch as tc
import math


def get_dataloader(args):

    if (args.dataset == 'CUFED'):
        ### import module
        m = import_module('dataset.' + args.dataset.lower())
        data_train = getattr(m, 'TrainSet')(args)   # Set up the training dataset.
        dataloader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        dataloader_test = {}
        for i in range(5):
            data_test = getattr(m, 'TestSet')(args=args, ref_level=str(i+1))    # Set up the evaluation dataset.
            dataloader_test[str(i+1)] = DataLoader(data_test, batch_size=1, shuffle=False, num_workers=args.num_workers)
        dataloader = {'train': dataloader_train, 'test': dataloader_test}

    elif (args.dataset == 'MRI_SR'):
        ### import module
        m = import_module('dataset.' + args.dataset.lower())
        data_train = getattr(m, 'TrainSet')(args)   # Set up the training dataset.
        dataloader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        data_test = getattr(m, 'EvaluationSet')(args=args)    # Set up the evaluation dataset.
        dataloader_test = DataLoader(data_test, batch_size=1, shuffle=False, num_workers=args.num_workers)
        data_final_test = getattr(m, 'FinalTestSet')(args=args)    # Set up the final test dataset.
        dataloader_final_test = DataLoader(data_final_test, batch_size=1, shuffle=False, num_workers=args.num_workers)
        dataloader = {'train': dataloader_train, 'test': dataloader_test, 'final test': dataloader_final_test}

    else:
        raise SystemExit('Error: no such type of dataset!')

    return dataloader