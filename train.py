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

 "train.py" - Initializing the network and proceeding to training and test epochs.
 
 Project: PyTorch e-prop

 Author:  C. Frenkel, Institute of Neuroinformatics, University of Zurich and ETH Zurich

 Cite this code: BibTeX/APA citation formats auto-converted from the CITATION.cff file in the repository are available 
       through the "Cite this repository" link in the root GitHub repo https://github.com/ChFrenkel/eprop-PyTorch/
       
------------------------------------------------------------------------------
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import models


def train(args, device, train_loader, traintest_loader, test_loader):
    torch.manual_seed(42)
    
    for trial in range(1,args.trials+1):
        
        # Network topology
        model = models.SRNN(n_in=args.n_inputs,
                            n_rec=args.n_rec,
                            n_out=args.n_classes,
                            n_t=args.n_steps,
                            thr=args.threshold,
                            tau_m=args.tau_mem,
                            tau_o=args.tau_out,
                            b_o=args.bias_out,
                            gamma=args.gamma,
                            dt=args.dt,
                            model=args.model,
                            classif=args.classif,
                            w_init_gain=args.w_init_gain,
                            lr_layer=args.lr_layer_norm,
                            t_crop=args.delay_targets,
                            visualize=args.visualize,
                            visualize_light=args.visualize_light,
                            device=device)

        # Use CUDA for GPU-based computation if enabled
        if args.cuda:
            model.cuda()
        
        # Initial monitoring
        if (args.trials > 1):
            print('\nIn trial {} of {}'.format(trial,args.trials))
        if (trial == 1):
            print("=== Model ===" )
            print(model)
        
        # Optimizer
        if args.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=args.lr)
        elif args.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
        elif args.optimizer == 'NAG':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
        elif args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
        else:
            raise NameError("=== ERROR: optimizer " + str(args.optimizer) + " not supported")
        
        # Loss function (only for performance monitoring purposes, does not influence learning as e-prop learning is hardcoded)
        if args.loss == 'MSE':
            loss = (F.mse_loss, (lambda l : l))
        elif args.loss == 'BCE':
            loss = (F.binary_cross_entropy, (lambda l : l))
        elif args.loss == 'CE':
            loss = (F.cross_entropy, (lambda l : torch.max(l, 1)[1]))
        else:
            raise NameError("=== ERROR: loss " + str(args.loss) + " not supported")
        
        # Training and performance monitoring
        print("\n=== Starting model training with %d epochs:\n" % (args.epochs,))        
        for epoch in range(1, args.epochs + 1):
            print("\t Epoch "+str(epoch)+"...")
            #Training:
            do_epoch(args, True, model, device, train_loader, optimizer, loss, 'train')           # Will display the average accuracy on the training set during the epoch (changing weights)
            #Check performance on the training set and on the test set:
            if not args.skip_test:
                #do_epoch(args, False, model, device, traintest_loader, optimizer, loss, 'train') # Uncomment to display the final accuracy on the training set after the epoch (fixed weights)
                do_epoch(args, False, model, device, test_loader, optimizer, loss, 'test')


def do_epoch(args, do_training, model, device, loader, optimizer, loss_fct, benchType):
    model.eval()    # This implementation does not rely on autograd, learning update rules are hardcoded
    score = 0
    loss = 0
    batch = args.batch_size if (benchType == 'train') else args.test_batch_size
    length = args.full_train_len if (benchType == 'train') else args.full_test_len
    with torch.no_grad():   # Same here, we make sure autograd is disabled
        
        # For each batch
        for batch_idx, (data, label) in enumerate(loader):

            data, label = data.to(device), label.to(device)
            if args.classif:    # Do a one-hot encoding for classification
                targets = torch.zeros(label.shape, device=device).unsqueeze(-1).expand(-1,-1,args.n_classes).scatter(2, label.unsqueeze(-1), 1.0).permute(1,0,2)
            else:
                targets = label.permute(1,0,2)

            # Evaluate the model for all the time steps of the input data, then either do the weight updates on a per-timestep basis, or on a per-sample basis (sum of all per-timestep updates).
            optimizer.zero_grad()
            output = model(data.permute(1,0,2), targets, do_training)
            if do_training:
                optimizer.step()
                
            # Compute the loss function, inference and score
            if args.delay_targets:
                loss += loss_fct[0](output[-args.delay_targets:], loss_fct[1](targets[-args.delay_targets:]), reduction='mean')
            else:
                loss += loss_fct[0](output, loss_fct[1](targets), reduction='mean')
            if args.classif:
                if args.delay_targets:
                    inference = torch.argmax(torch.sum(output[-args.delay_targets:],axis=0),axis=1)
                    score += torch.sum(torch.eq(inference,label[:,0]))
                else:
                    inference = torch.argmax(torch.sum(output,axis=0),axis=1)
                    score += torch.sum(torch.eq(inference,label[:,0]))
        
    if benchType == "train" and do_training:
        info = "on training set (while training): "
    elif benchType == "train":
        info = "on training set                 : "
    elif benchType == "test":
        info = "on test set                     : "

    if args.classif:
        print("\t\t Score "+info+str(score.item())+'/'+str(length)+' ('+str(score.item()/length*100)+'%), loss: '+str(loss.item()))
    else:
        print("\t\t Loss "+info+str(loss.item()))
            
