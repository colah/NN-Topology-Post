---
title: Neural Networks, Manifolds, and Topology
date: 2014-03-25
author: colah
mathjax: on
---

Visualization of tanh Layers
-----------------------------

As you know, a tanh layer $σ(Ax+b)$ consists of:

(1) A linear transformation by the matrix $A$
(2) A translation by the vector $b$
(3) Point-wise application of tanh.

We can visualize this as a continuous transformation, as follows:


<div class="centerimgcontainer">
<img src="img/1layer.gif" alt="Gradually applying a neural network layer" style="">
</div>
<div class="spaceafterimg"></div>

The story is much the same for other activation functions.

Topology of tanh Layers
------------------------

The layer stretches and squishes space, but it never cuts, breaks, or folds it. Intuitively, we can see that it preserves topological properties. For example, a set will be connected afterwards if it was before (and vice versa).

Transformation like this, which don't affect topology, are called homeomorphisms in topology. Formally, they are bijections that are continuous functions both ways.

**Theorem**: Layers with $N$ inputs and $N$ outputs are homeomorphisms, if $W$ is non-singular. (Though one needs to be careful about domain and range.)

**Proof**: Let's consider this step by step:

(1) Let's assume $W$ has a non-zero determinant. Then it is a bijective linear function with a linear inverse. Linear functions are continuous. So, multiplying by $W$ is a homeomorphism.
(2) Translations are homeomorphisms
(3) tanh (and sigmoid and softplus but not ReLU) are continuous functions with continuous inverses. They are bijections if we are careful about the domain and range we consider. Applying them pointwise is a homemorphism

Thus, if W has a non-zero determinant, our layer is a homeomorphism. ∎

It's actually a bit stronger than this. As our animation suggests, we can continuously deform from the identity function to the neural network layer. (Though we may need to permute two of the output dimensions.) This is very significant and we will discuss it later.

These results also hold if we compose arbitrarily many of these layers together.


Topology and Classification
---------------------------

<div class="floatrightimgcontainer">
<img src="img/topology_base.png" alt="" style="">
<div class="caption">A is red, B is blue</div>
</div>
<div class="spaceafterimg"></div>


Consider a two dimensional dataset with two classes $A, B \subset \mathbb{R}^2$:

$$A = \{x | d(x,0) < 1/3\}$$
$$B = \{x | 2/3 < d(x,0) < 1\}$$

**Claim**: It is impossible for a neural network to classify this dataset without having a layer that has 3 or more hidden units, regardless of depth.

Classification with a sigmoid unit or a softmax layer would be equivalent to trying to find a hyperplane (or in this case a line) that separates $A$ and $B$. 

Unfortunately, with only two hidden units, a network is topologically doomed to failure on this dataset. We can watch it struggle and try to learn a way to do this:

<div class="centerimgcontainer">
<img src="img/topology_2D-2D_train.gif" alt="" style="">
</div>
<div class="spaceafterimg"></div>

(It's actually able to achieve ~80% classification accuracy.)

This example only had one hidden layer, but it would fail regardless.

**Proof**: Either each layer is a homeomorphism, or the layer has determinant 0. If it is a homemorphism, A is still surrounded by B, and a line can't separate them. But suppose it has a determinant of 0: then the dataset gets collapsed on some axis. Since we're dealing with something homeomorphic to the original dataset, A is surrounded by B, and collapsing on any axis means we will have some points of A and B mix and become impossible to distinguish between.

If we add a third hidden unit, the problem becomes trivial. The neural network learns the following representation:

<div class="centerimgcontainer">
<img src="img/topology_3d.png" alt="" style="">
</div>
<div class="spaceafterimg"></div>


With this representation, we can separate the datasets with a hyperplane.

To get a better sense of what's going on, let's consider an even simpler dataset that's 1-dimensional:

$$A = [-\frac{1}{3}, \frac{1}{3}]$$
$$B = [-1, -\frac{2}{3}] \cup [\frac{2}{3}, 1]$$

Without using a layer of two or more hidden units, you can't classify this dataset. But if you use one with two units, we learn to represent the data as a nice curve that allows us to separate the data:

<div class="centerimgcontainer">
<img src="img/topology_1D-2D_train.gif" alt="" style="">
</div>
<div class="spaceafterimg"></div>


What's happening? One hidden unit learns to fire when $x > -\frac{1}{2}$ and one learns to fire when $x > \frac{1}{2}$. When the first one fires, but not the second, we know that we are in A. 


The Manifold Hypothesis
------------------------

Is this relevant to real world data sets, like image data? If you take the manifold hypothesis really seriously, I think it bares consideration.

The manifold hypothesis is that natural data forms lower-dimensional manifolds in its embedding space. If you believe this, then the task of a classification algorithm is fundamentally to separate a bunch of tangled manifolds.

In the previous examples, one class completely surrounded another. It doesn't seem very likely that the dog image manifold is completely surrounded by the cat image manifold. But there are other, more plausible topological situations that could still pose an issue, as we will see in the next section.

Links And Homotopy
------------------

Another interesting dataset to consider is two linked tori, $A$ and $B$.

<div class="centerimgcontainer">
<img src="img/link.png" alt="" style="">
</div>
<div class="spaceafterimg"></div>


Much like the previous datasets we considered, this dataset can't be separated without using $n+1$ dimensions, namely a $4$th dimension.

Links are studied in knot theory, an area of topology. I don't know much about it, but there are higher dimensional links. It doesn't seem impossible that we could have some sort of similar situation in real world data.

Sometimes when you see a link, it isn't immediately obvious whether it's an unlink (a bunch of things that are tangled together, but can be separated by continuous deformation) or not. 

We observed earlier, in the first animation, that we can imagine continually deforming from the identity function to the neural net layer function. This is now of significant interest.

In topology, we would say that the layer is homotopic to the identity function. Formally, two continuous functions $f_0, f_1: X \to Y$ are homotopic if there exists a continuous function $F: X \times [0,1] \to Y$ such that $F(x, 0) = f_0(x)$, $F(x, 1) = f_1(x)$. That is, $F(x, t)$ continuously transitions from $f_0(x)$ to $f_1(x)$.

**Theorem**: A network layer is homotopic to the identity function if: a) W isn't singular, b) you are willing to permute the neurons in the hidden layer, c) there is more than 1 hidden unit, and d) you are careful about domains and ranges.

**Proof**: Again, we consider each stage of the network individually:

(1) The linear transformation is, in fact, the hardest part. In order for this to be possible, we need $W$ to have a positive determinant. Our premise is that it isn't zero, and we can flip the sign if it is negative by switching two of the hidden neurons, so we can guarantee the determinant is positive. The space of positive determinant matrices is path-connected, so there exists $p: [0,1] -> GL_n(\mathbb{R})$ such that $p(0) = Id$ and $p(1) = W$. We can continually transition from the identity function to the $W$ transformation with the function $x \to p(t)x$
(2) We can continually transition from the identity function to the b translation with the function $x \to x + tb$
(3) We can continually transition from the identity function to the pointwise use of σ with the function: $x \to (1-t)x + tσ(x)$. ∎

Homotopies are used in knot theory to study braids, links, and knots because a particular kind of homotopy, an ambient isotopy, demonstrates the equivalence of one of these to another. I imagine there is probably interest in programs automatically discovering such isotopies and automatically proving the equivalence of certain links, or that certain links are separable. It would be interesting to know if neural networks can beat whatever the state of the art is there.

The Easy Way Out
----------------

The natural thing for a neural net to do, the very easy route, is to try and pull the manifolds apart naively and stretch the parts that are tangled as thin as possible. While this won't be anywhere close to a genuine solution, it can achieve relatively low classification accuracy and be a tempting local minimum.

It would present itself as very high derivatives on the regions it is trying to stretch, and sharp near-discontinuities. We know these things happen. Contractive penalties, penalizing the derivatives of the layers on data points, are the natural way to fight this.

Since these sort of local minima are absolutely useless from the perspective of trying to solve topological problems, topological problems may provide a nice motivation to explore fighting these issues.

Better Layers for Manipulating Manifolds?
-----------------------------------------

The more I think about standard neural network layers -- that is, with an affine transformation followed by a point-wise activation function -- the more disenchanted I feel. It's hard to imagine that these are really very good for manipulating manifolds.

Perhaps it might make sense to have a very different kind of layer that we use in composition with more traditional ones?

The thing that feels natural to me is to learn a vector field with the direction you want to shift the manifold:

<div class="centerimgcontainer">
<img src="img/grid_vec.png" alt="" style="">
</div>
<div class="spaceafterimg"></div>

And then deform space based on it:

<div class="centerimgcontainer">
<img src="img/grid_bubble.png" alt="" style="">
</div>
<div class="spaceafterimg"></div>

My intuition is that we should learn the vector field at fixed points (just take some fixed points from the training set to use as anchors) and interpolate in some manner. The vector field above is of the form:

$$F(x) = \frac{v_0f_0(x) + v_1f_1(x)}{1+f_0(x)+f_1(x)}$$

Where $v_0$ and $v_1$ are vectors and $f_0(x)$ and $f_1(x)$ are n-dimensional gaussians. This is inspired a bit by RBFs.

K-Nearest Neighbor Layers
-------------------------

The more I think about it, the more persuaded I become that linear separability may be a huge, and possibly unreasonable amount to demand of a neural network. Really, the natural feeling thing would be to use nearest neighbors. However, one clearly needs a good representation before k-NN can work well.

As a first experiment, I trained some ~1% test error MNIST networks (two layer conv nets, no dropout). I then dropped the final softmax layer and used the k-NN algorithm. I was able to consistently achieve a reduction in test error of 0.1-0.2%.

Still, this doesn't quite feel like the right thing. The network is still trying to do linear classification, but since we use k-NN at test time, it's able to recover a bit from mistakes it made.

k-NN is differentiable with respect to the representation it's acting on, because of the 1/distance weighting. As such, we can train a network directly for k-NN classification. This can be thought of as a kind of "nearest neighbor" layer that acts as an alternative to softmax.

Clearly, we don't want to feedforward our entire training set for each mini-batch. I think a nice approach is to classify each element of the mini-batch based on the classes of other elements of the mini-batch, giving each one a weight of 1/(distance from classification target). (I used a slightly less elegant, but roughly equivalent algorithm because it was more practical to implement in Theano: feedforward two different batches at the same time, and classify them based on each other.)

Sadly, this only gets down to 5-4% test error. Though I've put very little effort into playing with hyper-parameters. Using simpler networks gets worse results.

Still, I really aesthetically like this approach, because it seems like what we're "asking" the network to do is much more reasonable. We want points of the same manifold to be closer than points of others. This should correspond to inflating the space between manifolds for different datatypes and contracting the individual manifolds. It feels kind of like simplification. 



