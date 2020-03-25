### Helper:

#### Discriminators have two variants:

If we you want to use W-GAN and WGAN-GP loss: We can enforce 1-Lipschitz constraint to D by using [Spectral Normalisation](https://openreview.net/forum?id=B1QRgziT-)
- One with Spectral Normalisation (SpecNorm GAN)

###### Using conventional loss function:
- Without Spectral Normalisation: LS Loss.

#### Note: We have to adjust optimizers accordingly, if you are using different Discriminators.

### Generators: 
- Still looking for good G-Two network. 

           Synthesized images are not adequate.
           
- G-One can do better, however in this model we are using EDSR. 

##### TODOs:
- [ ] Find best variant for G-Two. 
- [ ] FIDs and Inception scores.
- [x] Spectral Normalisation.

### Losses
- [ ] Hinge Loss.
- [x] Total Variation
- [x] Perceptual Distance.
- [x] MSE   