# Generative Adversarial Networks (GAN)
My understanding of different GANs. 

## NOTE: 
Prerequisite for GAN are CNN and basic machine learning concepts.

## Definition
By the name itself GAN doesn't giveaway the meaning of what it does. So to understand better, let's breakdown what comprises GAN.
#### Generative
A generative model is said to generate a probability distribution that mimicks an original probability distribution.

#### Adversarial
<i>Literal definition:</i> Conflict or opposition
<i>In context of GAN:</i> GANs are comprised of are deep neural net architectures comprised of two neural networks, competing one agaisnt the other.

#### Networks
A networks can be deep neural networks, CNN, or simple vanilla networks.


GAN are nerual neetworks that are trained in an adversarial manner to generate data mimicking some distribution.

### Two classes of models in machine learning
* Discriminative model: It is the one that discriminiates between two different classes of data.

Examples: Classification problems.

* Generative model: A generative models <b>G</b> to be trained on training data X sample from some true distribution D is the one whch gives some standard random distribution Z produces a distribution D which is close to D according to some closeness metric. 