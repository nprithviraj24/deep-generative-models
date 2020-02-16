#import python dependencies
import argparse
import numpy as np
import matplotlib
import os

#import utilities
import utilities as utility

#importing Generator AND Discriminator for high-to-low
from High-to-Low import Generator as G_H2L  
from High-to-Low import Discriminator as D_H2L

#Similarly importing Generator AND Discriminator for low to high l2h
from Low-to-High import GEN_DEEP as G_L2H  
from Low-to-High import Discriminator as D_L2H

#import dataloader
import dataloader

#loss functions
import lossFunctions as LossF

#training loop
import training-loop

#torch
import torch
from torch import nn



import time
import pylab as pl
from IPython import display
from torch import nn



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--tensorboard', type=str, default='runs/')
parser.add_argument('--pixelWeight', type=float, default=1.00)
parser.add_argument('--ganWeight', type=float, default=0.05)


# parser.add_argument('--loss', type=str, default='hinge')
# parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')

# parser.add_argument('--model', type=str, default='resnet')

args = parser.parse_args()

#number of updates to discriminator for every update to generator 
disc_iters = 5

# discriminator = torch.nn.DataParallel(Discriminator()).cuda() # TODO: try out multi-gpu training

h2l_g = G_H2L().to(device)
h2l_d = D_H2L().to(device)

l2h_d = D_L2H().to(device)
l2h_g = G_L2H().to(device)
l2h_g.load_state_dict(torch.load('model.pkl'))
# net_G_low2high = net_G_low2high.eval()

#-----------HYPERPARAMETERS FOR OPTIMIZATION---------

# because the spectral normalization module creates parameters that don't require gradients (u and v), we don't want to 
# optimize these using sgd. We only let the optimizer operate on parameters that _do_ require gradients
# TODO: replace Parameters with buffers, which aren't returned from .parameters() method.
optim_discH2L = optim.Adam(filter(lambda p: p.requires_grad, h2l_d.parameters()), lr=args.lr, betas=(0.0,0.9))
optim_discL2H = optim.Adam(filter(lambda p: p.requires_grad, l2h_d.parameters()), lr=args.lr, betas=(0.0,0.9))


optim_genH2L  = optim.Adam(h2l_g.parameters(), lr=args.lr, betas=(0.0,0.9))
optim_genL2H  = optim.Adam(l2h_g.parameters(), lr=args.lr, betas=(0.0,0.9))

# use an exponentially decaying learning rate
scheduler_d = optim.lr_scheduler.ExponentialLR(optim_discH2L, gamma=0.99)
scheduler_d = optim.lr_scheduler.ExponentialLR(optim_genH2L, gamma=0.99)
scheduler_d = optim.lr_scheduler.ExponentialLR(optim_discL2H, gamma=0.99)
scheduler_d = optim.lr_scheduler.ExponentialLR(optim_genL2H, gamma=0.99)


real = 1
fake = 0

def training_loop(dataloader_X, dataloader_Y, test_dataloader_X, test_dataloader_Y,
                  n_epochs=200):

    print_every=10

    # keep track of losses over time
    losses = []

    test_iter_X = iter(test_dataloader_X)
    test_iter_Y = iter(test_dataloader_Y)

    # Get some fixed data from domains X and Y for sampling. These are images that are held
    # constant throughout training, that allow us to inspect the model's performance.
    fixed_X = test_iter_X.next()[0]
    fixed_Y = test_iter_Y.next()[0]
    # make sure to scale to a range -1 to 1
    # fixed_X = scale(fixed_X) 
    # fixed_Y = scale(fixed_Y)

    # batches per epoch
    
    # n_epochs = 2
    for epoch in range(pretrain_epoch, n_epochs+1):

      epochG_loss = 0
      runningG_loss = 0
      runningDX_loss = 0
      runningDY_loss = 0
      LOG_INTERVAL = 25

      bn = 0 #mini batches per epoch

      for batch_id, (images_X, _) in tqdm_notebook(enumerate(dataloader_X), total=len(dataloader_X)):
        #  with torch.no_grad():
           bn += 1
           images_Y, a = next(iter(dataloader_Y))
           # move images to GPU if available (otherwise stay on CPU)
           
           images_X = images_X.to(device)
           images_Y = images_Y.to(device)
          
          #5:1 ratio
          # discriminator updates  
           for _ in range(disc_iters):
                
                real_label = torch.full((args.batch_size,), real, device=device)
                fake_label = real_label.fill_(fake)

                optim_discH2L.zero_grad()
                discH2L_loss = nn.BCELoss()( h2l_d(images_Y), real_label).mean() + nn.BCELoss()(h2l_d(h2l_g(images_X)), fake_label).mean()
                discH2L_loss.backward()
                optim_discH2L.step()

                optim_discL2H.zero_grad()
                discL2H_loss = nn.BCELoss()( l2h_d(images_X), real_label).mean() + nn.BCELoss()(l2h_d(l2h_g(images_Y)), fake_label).mean()
                discL2H_loss.backward()
                optim_discL2H.step()

#generator updates
          
          # high to low
           optim_genH2L.zero_grad()

           fakeY = h2l_g(images_X)
           fakeY_d = h2l_d(fakeY)
           ganLoss = LossF.GANloss( real=h2l_d(images_Y), fake=fakeY_d )
           pixelLoss = LossF.pixelLoss(real=images_Y, fake=fakeY)
            
           lossH2L = args.ganWeight*ganLoss + args.pixelWeight*pixelLoss

           lossH2L.backward()
           optim_genH2L.step()
           del fakeY_d

           # low to high
           optim_genL2H.zero_grad()

           fakeX = l2h_g(fakeY)
           fakeX_d = l2h_d(fakeX)
           ganLoss = LossF.GANloss( real=l2h_d(images_X), fake=fakeX_d )
           pixelLoss = LossF.pixelLoss(real=images_X, fake=fakeX)
            
           lossH2L = args.ganWeight*ganLoss + args.pixelWeight*pixelLoss

           lossH2L.backward()
           optim_genH2L.step()
           del fakeX_d, fakeX, fakeY
           

           if bn % LOG_INTERVAL == 0:  
             with torch.no_grad():
              h2l_g.eval() # set generators to eval mode for sample generation
              fakeY = h2l_g(fixed_X.to(device))
              utility.imshow(torchvision.utils.make_grid(fakeY.cpu()))
              # utility.imshow(torchvision.utils.make_grid(fakeY.cpu()))
              h2l_g.train() 

              l2h_g.eval() # set generators to eval mode for sample generation
              fakeX = l2h_g(fakeY.to(device))
              utility.imshow(torchvision.utils.make_grid(fakeX.cpu()))
              l2h_g.train()          
              # print('Mini-batch no: {}, at epoch [{:3d}/{:3d}] | d_X_loss: {:6.4f} | d_Y_loss: {:6.4f} | g_total_loss: {:6.4f}'.format(mbps, epoch, n_epochs,  runningDX_loss/mbps ,  runningDY_loss/mbps,  runningG_loss/mbps ))

      with torch.no_grad():
        h2l_g.eval() # set generators to eval mode for sample generation
        fakeY = h2l_g(fixed_X.to(device))
        utility.imshow(torchvision.utils.make_grid(fakeY.cpu()))
              # utility.imshow(torchvision.utils.make_grid(fakeY.cpu()))
        h2l_g.train() 

        l2h_g.eval() # set generators to eval mode for sample generation
        fakeX = l2h_g(fakeY.to(device))
        utility.imshow(torchvision.utils.make_grid(fakeX.cpu()))
        l2h_g.train()  
        # print("Epoch loss:  ", epochG_loss/)
      losses.append((runningDX_loss/bn, runningDY_loss/bn, runningG_loss/bn))
      print('Epoch [{:5d}/{:5d}] | d_X_loss: {:6.4f} | d_Y_loss: {:6.4f} | g_total_loss: {:6.4f}'.format(epoch, n_epochs, runningDX_loss/mbps ,  runningDY_loss/mbps,  runningG_loss/mbps ))
              
      
    return losses

