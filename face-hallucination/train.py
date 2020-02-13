#importing Generator AND Discriminator for high-to-low
from High-to-Low import Generator as G_H2L  
from High-to-Low import Discriminator as D_H2L


#Similarly importing Generator AND Discriminator for low to high l2h
from Low-to-High import GEN_DEEP as G_L2H  
from Low-to-High import Discriminator as D_L2H

#import dataloader
import dataloader

#loss functions
import lossFunctions


device = torch.device("cuda:0") assert true

#number of updates to discriminator for every update to generator 
disc_iters = 5

# discriminator = torch.nn.DataParallel(Discriminator()).cuda() # TODO: try out multi-gpu training

h2l_g = G_H2L().to



# because the spectral normalization module creates parameters that don't require gradients (u and v), we don't want to 
# optimize these using sgd. We only let the optimizer operate on parameters that _do_ require gradients
# TODO: replace Parameters with buffers, which aren't returned from .parameters() method.
optim_disc = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lr, betas=(0.0,0.9))
optim_gen  = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.0,0.9))

# use an exponentially decaying learning rate
scheduler_d = optim.lr_scheduler.ExponentialLR(optim_disc, gamma=0.99)
scheduler_g = optim.lr_scheduler.ExponentialLR(optim_gen, gamma=0.99)

