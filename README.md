# Deep Generative Models

 - Autoregressive models.
 - Variational Autoencoders
 - Generative Adversarial Networks

## Generative Adversarial Networks (GAN)
#### Prerequisites 
GAN are CNN and basic machine learning concepts.

### Definition
By the name itself GAN doesn't giveaway the meaning of what it does. So to understand better, let's breakdown what comprises GAN.
#### Generative
A generative model is said to generate a probability distribution that mimicks an original probability distribution.

#### Adversarial
<i>Literal definition:</i> Conflict or opposition.

<i>In context of GAN:</i> GANs are comprised of two deep neural networks that are competing one against each other.

#### Networks
A networks can be deep neural networks, CNN, or simple vanilla networks.

GAN are neural networks that are trained in an adversarial manner to generate data mimicking some distribution.

### Two classes of models in machine learning
* Discriminative model: It is the one that discriminiates between two different classes of data.

Examples: Classification problems.

* Generative model: A generative models <b>  &theta  </b> to be trained on training data <b>X</b> sampled from some true distribution <b>D</b> is the one which, given some standard random distribution <b>Z</b> produces a distribution <b>$\hat{a}$</b> which is close to <b>D</b> according to some closeness metric. 

![alt text](https://openai.com/content/images/2017/02/gen_models_diag_2.svg)
<i>Source: OpenAI blog</i>


### Face Hallucination
##### This repo is built on grounds of developing different generative models to perform super-resolution (SR) on face images with different domain.

1. Image-Degrade: [To learn image super-resolution, use a GAN to learn how to do image degradation first](https://arxiv.org/pdf/1807.11458.pdf)
2. CycleGAN with EDSR. [CycleGAN](https://junyanz.github.io/CycleGAN/) and [EDSR](https://arxiv.org/pdf/1707.02921.pdf)
3. StyleVAE: Style basedVAE for Real-World SR. [StyleVAE + SR Network](https://arxiv.org/abs/1912.10227)


## References

- [NYU](https://cs.nyu.edu/courses/spring18/CSCI-GA.3033-022/)
- [Deep Learning Drizzle](https://deep-learning-drizzle.github.io/index.html)
- [Stanford](https://deepgenerativemodels.github.io/syllabus.html)
