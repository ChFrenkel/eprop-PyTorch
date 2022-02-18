# -*- coding: utf-8 -*-

"""
------------------------------------------------------------------------------

Copyright (C) 2020-2022 University of Zurich

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

------------------------------------------------------------------------------

 "setup.py" - Setup configuration and dataset loading.
 
 Project: PyTorch e-prop

 Author:  C. Frenkel, Institute of Neuroinformatics, University of Zurich and ETH Zurich

 Cite this code: BibTeX/APA citation formats auto-converted from the CITATION.cff file in the repository are available 
       through the "Cite this repository" link in the root GitHub repo https://github.com/ChFrenkel/eprop-PyTorch/

------------------------------------------------------------------------------
"""


import torch
import numpy as np
import numpy.random as rd
import sys


class CueAccumulationDataset(torch.utils.data.Dataset):
    """Adapted from the original TensorFlow e-prop implemation from TU Graz, available at https://github.com/IGITUGraz/eligibility_propagation"""

    def __init__(self, args, type):
      
        n_cues     = 7
        f0         = 40
        t_cue      = 100
        t_wait     = 1200
        n_symbols  = 4
        p_group    = 0.3
        
        self.dt         = 1e-3
        self.t_interval = 150
        self.seq_len    = n_cues*self.t_interval + t_wait
        self.n_in       = 40
        self.n_out      = 2    # This is a binary classification task, so using two output units with a softmax activation redundant
        n_channel       = self.n_in // n_symbols
        prob0           = f0 * self.dt
        t_silent        = self.t_interval - t_cue
        
        if (type == 'train'):
            length = args.train_len
        else:
            length = args.test_len
            
    
        # Randomly assign group A and B
        prob_choices = np.array([p_group, 1 - p_group], dtype=np.float32)
        idx = rd.choice([0, 1], length)
        probs = np.zeros((length, 2), dtype=np.float32)
        # Assign input spike probabilities
        probs[:, 0] = prob_choices[idx]
        probs[:, 1] = prob_choices[1 - idx]
    
        cue_assignments = np.zeros((length, n_cues), dtype=np.int)
        # For each example in batch, draw which cues are going to be active (left or right)
        for b in range(length):
            cue_assignments[b, :] = rd.choice([0, 1], n_cues, p=probs[b])
    
        # Generate input spikes
        input_spike_prob = np.zeros((length, self.seq_len, self.n_in))
        t_silent = self.t_interval - t_cue
        for b in range(length):
            for k in range(n_cues):
                # Input channels only fire when they are selected (left or right)
                c = cue_assignments[b, k]
                input_spike_prob[b, t_silent+k*self.t_interval:t_silent+k*self.t_interval+t_cue, c*n_channel:(c+1)*n_channel] = prob0
    
        # Recall cue and background noise
        input_spike_prob[:, -self.t_interval:, 2*n_channel:3*n_channel] = prob0
        input_spike_prob[:, :, 3*n_channel:] = prob0/4.
        input_spikes = generate_poisson_noise_np(input_spike_prob)
        self.x = torch.tensor(input_spikes).float()
    
        # Generate targets
        target_nums = np.zeros((length, self.seq_len), dtype=np.int)
        target_nums[:, :] = np.transpose(np.tile(np.sum(cue_assignments, axis=1) > int(n_cues/2), (self.seq_len, 1)))
        self.y = torch.tensor(target_nums).long()
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]



def setup(args):
    args.cuda = not args.cpu and torch.cuda.is_available()
    if args.cuda:
        print("=== The available CUDA GPU will be used for computations.")
        device = torch.cuda.current_device()
    else:
        device = torch.device('cpu')
    
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

    if args.dataset == "cue_accumulation":
        print("=== Loading cue evidence accumulation dataset...")
        (train_loader, traintest_loader, test_loader) = load_dataset_cue_accumulation(args, kwargs)
    else:
        print("=== ERROR - Unsupported dataset ===")
        sys.exit(1)
        
    print("Training set length: "+str(args.full_train_len))
    print("Test set length: "+str(args.full_test_len))
    
    return (device, train_loader, traintest_loader, test_loader)


def load_dataset_cue_accumulation(args, kwargs):

    trainset = CueAccumulationDataset(args,"train")
    testset  = CueAccumulationDataset(args,"test")

    train_loader     = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,      shuffle=args.shuffle, **kwargs)
    traintest_loader = torch.utils.data.DataLoader(trainset, batch_size=args.test_batch_size, shuffle=False       , **kwargs)
    test_loader      = torch.utils.data.DataLoader(testset , batch_size=args.test_batch_size, shuffle=False       , **kwargs)
    
    args.n_classes      = trainset.n_out
    args.n_steps        = trainset.seq_len
    args.n_inputs       = trainset.n_in
    args.dt             = trainset.dt
    args.classif        = True
    args.full_train_len = len(trainset)
    args.full_test_len  = len(testset)
    args.delay_targets  = trainset.t_interval
    args.skip_test      = False
    
    return (train_loader, traintest_loader, test_loader)


def generate_poisson_noise_np(prob_pattern, freezing_seed=None):
    if isinstance(prob_pattern, list):
        return [generate_poisson_noise_np(pb, freezing_seed=freezing_seed) for pb in prob_pattern]

    shp = prob_pattern.shape

    if not(freezing_seed is None): rng = rd.RandomState(freezing_seed)
    else: rng = rd.RandomState()

    spikes = prob_pattern > rng.rand(prob_pattern.size).reshape(shp)
    return spikes
