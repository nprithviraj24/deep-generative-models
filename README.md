# Deep Generative Models

 - Autoregressive models.
 - Variational Autoencoders
 - Generative Adversarial Networks

## Generative Adversarial Networks (GAN)
My understanding of different GANs. 

## NOTE: 
Prerequisite for GAN are CNN and basic machine learning concepts.

## Definition
By the name itself GAN doesn't giveaway the meaning of what it does. So to understand better, let's breakdown what comprises GAN.
#### Generative
A generative model is said to generate a probability distribution that mimicks an original probability distribution.

#### Adversarial
<i>Literal definition:</i> Conflict or opposition.


<i>In context of GAN:</i> GANs are comprised of two deep neural networks that are competing one against each other.

#### Networks
A networks can be deep neural networks, CNN, or simple vanilla networks.


GAN are nerual neetworks that are trained in an adversarial manner to generate data mimicking some distribution.

### Two classes of models in machine learning
* Discriminative model: It is the one that discriminiates between two different classes of data.

Examples: Classification problems.

* Generative model: A generative models <b>  &theta  </b> to be trained on training data <b>X</b> sampled from some true distribution <b>D</b> is the one which, given some standard random distribution <b>Z</b> produces a distribution <b>$\hat{a}$</b> which is close to <b>D</b> according to some closeness metric. 

![alt text](https://openai.com/content/images/2017/02/gen_models_diag_2.svg)
<i>Source: OpenAI blog</i>


## References

- [NYU](https://cs.nyu.edu/courses/spring18/CSCI-GA.3033-022/)
- [Deep Learning Drizzle](https://deep-learning-drizzle.github.io/index.html)
- [Stanford](https://deepgenerativemodels.github.io/syllabus.html)
