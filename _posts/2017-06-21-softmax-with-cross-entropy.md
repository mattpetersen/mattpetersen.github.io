---
layout: post
title: "Softmax with cross-entropy"
date: 2017-06-21
header: true
footer: true
comments: true
tags: backpropogation, matrix calculus, softmax, cross-entropy, neural networks, deep learning
---

A matrix-calculus approach to deriving the sensitivity of cross-entropy cost to the weighted input to a softmax output layer. _We use row vectors and row gradients_, since typical neural network formulations let columns correspond to features, and rows correspond to examples. This means that the input to our softmax layer is a row vector with a column for each class.

<br>
## Softmax

Softmax is a vector-to-vector transformation that turns a row vector

$$ \mathbf x = \begin{bmatrix} x_1,\ x_2,\ . . . \ ,\  x_n \end{bmatrix} $$

into a normalized row vector

$$ \mathbf s( \mathbf x ) = \begin{bmatrix} s(\mathbf  x)_1,\ s(\mathbf  x)_2,\ . . . \, \ s(\mathbf x)_n \end{bmatrix}. $$

The transformation is described element-wise, where the $$i$$th output $$s(\mathbf x)_i$$ is a function of the entire input $$\mathbf x$$, and is given by

$$ s(\mathbf x)_i = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}. $$

Softmax is nice because it turns $$\mathbf x$$ into a probability distribution.

* Each element $$s(\mathbf x)_i$$ is between $$0$$ and $$1$$.
* The elements $$s(\mathbf x)_i$$ sum to $$1$$.



<br>
## Jacobian

Since softmax is a vector-to-vector transformation, its derivative is a Jacobian matrix. The Jacobian has a row for each output element $$s(\mathbf x)_i$$, and a column for each input element $$x_j$$. The entries of the Jacobian take two forms, one for the main diagonal entry, and one for every off-diagonal entry. We'll compute row $$i$$ of the Jacobian, which is the gradient of output element $$s(\mathbf x)_i$$ with respect to each of its input elements $$x_j$$.

First compute the diagonal entry of row $$i$$ of the Jacobian, that is, compute the derivative of the $$i$$'th output of softmax, $$s(\mathbf x)_i$$, with respect to its $$i$$'th input, $$x_i$$:

<script type="math/tex; mode=display">
\begin{aligned}
\frac{\partial s(\mathbf x)_i}{\partial x_{i}} &= \frac{\sum_{j=1}^{n} e^{x_j} e^{x_i} - e^{x_i} e^{x_i}}{(\sum_{j=1}^{n} e^{x_j})^2} \\[1.6em]
&= s(\mathbf x)_i - s(\mathbf x)_i^2 \qquad \text{diagonal entry}
\end{aligned}
</script>

Now compute every off-diagonal entry of row $$i$$ of the Jacobian, that is, compute the derivative of the $$i$$'th output of softmax, $$s(\mathbf x)_i$$, with respect to its $$j$$'th input, $$x_j$$, where $$j \neq i$$.

<script type="math/tex; mode=display">
\begin{aligned}
\frac{\partial s(\mathbf x)_i}{\partial x_j} &= \frac{\sum_{j=1}^{n} e^{x_j} \cdot 0 - e^{x_i} e^{x_j}}{(\sum_{j=1}^{n} e^{x_j})^2} \\[1.6em]
&= - s(\mathbf x)_i s(\mathbf x)_j \qquad \text{off-diagonal entry}
\end{aligned}
</script>

The form of the off-diagonals tells us that the Jacobian of softmax is a symmetric matrix. This is nice because symmetric matrices have great numeric and analytic properties. We expand it below. Each row is a gradient of one output element $$s(\mathbf x)_i$$ with respect to each of its input elements $$x_j$$.

<script type="math/tex; mode=display">
\mathbf J_{\mathbf x} =
\begin{bmatrix}
    \nabla s(\mathbf x)_0 \\[0.5em]
    \nabla s(\mathbf x)_1 \\[0.5em]
    . . . \\
    \nabla s(\mathbf x)_n
\end{bmatrix}
=
\begin{bmatrix}
    s(\mathbf x)_0 - s(\mathbf x)_0^2 & -s(\mathbf x)_0 s(\mathbf x)_1 & . . .  & -s(\mathbf x)_0 s(\mathbf x)_n \\[0.5em]
    -s(\mathbf x)_1 s(\mathbf x)_0 & s(\mathbf x)_1 - s(\mathbf x)_1^2 & . . .   & -s(\mathbf x)_1 s(\mathbf x)_n \\[0.5em]
    . . . & . . .  & . . . & . . . \\[0.5em]
    -s(\mathbf x)_n s(\mathbf x)_0 & -s(\mathbf x)_n s(\mathbf x)_1 & . . .  & s(\mathbf x)_n - s(\mathbf x)_n^2
\end{bmatrix}
</script>

Notice that we can express this matrix as

$$\mathbf J_{\mathbf x} = \text{diag} \big(\mathbf s(\mathbf x)\big) - \mathbf s(\mathbf x)^\top \mathbf s(\mathbf x)$$

where the second term is the $$n \times n$$ outer product.



<br>
## Cross-entropy

Cross-entropy measures the difference between two probability distributions. We saw that $$\mathbf s(\mathbf x)$$ is a distribution. The correct class is also a distribution, that is, assuming we encode it as a one-hot vector:

$$\mathbf y = \begin{bmatrix} y_1,\ y_2,\ . . . \ ,\ y_n \end{bmatrix} = \begin{bmatrix} 0,\ 0,\ . . . \ , \ 1,\ . . . \ , \ 0 \end{bmatrix},$$

where the $$1$$ appears at the index of the correct class.

The cross-entropy between our predicted distribution over classes, $$\mathbf s( \mathbf x)$$, and the true distribution over classes, $$\mathbf y$$, is a scalar measure of their difference, which is perfect for a cost function. It'll drive our softmax distribution toward the one-hot distribution. We can write this cost function as

<script type="math/tex; mode=display">
\begin{aligned}
H(\mathbf y, \mathbf s(\mathbf x))
&= -\sum_{i=1}^n y_i \log s(\mathbf x)_i \\[1.6em] 
&= -\mathbf y \log \mathbf s(\mathbf x)^\top,
\end{aligned}
</script>

which is the dot product since we're using row vectors. This formula comes from information theory. It measures the information gained about our softmax distribution when we sample from our one-hot distribution.



<br>
## Gradient

Since our $$\mathbf y$$ is given and fixed, cross-entropy is a vector-to-scalar function of only our softmax distribution. That means it will have a gradient with respect to our softmax distribution. This vector-to-scalar cost function is actually made up of two steps: (1) a vector-to-vector element-wise $$\log$$ and (2) a vector-to-scalar dot product. The vector-to-vector logarithm will have a Jacobian, but since it's applied element-wise, the Jacobian will be diagonal, holding each elementwise derivative. The gradient of the dot product operation is matrix multiplied on the left of the Jacobian of the elementwise logarithm in the part below:

<script type="math/tex; mode=display">
\begin{aligned}
\nabla_{\mathbf s(\mathbf x)} H(\mathbf y, \mathbf s(\mathbf x)) 
&= -\nabla_{\mathbf s(\mathbf x)} \mathbf y \log \mathbf s(\mathbf x)^\top \\[1.6em]
&= -\mathbf y \nabla_{\mathbf s(\mathbf x)} \log \mathbf s(\mathbf x) \\[1.6em]
&= -\mathbf y \ \text{diag}\left(\frac{\mathbf 1}{\mathbf s(\mathbf x)}\right) \\[1.6em]
&= -\frac{\mathbf y}{\mathbf s(\mathbf x)},
\end{aligned}
</script>

where we used equation (69) of [the matrix cookbook](http://www2.imm.dtu.dk/pubdb/views/edoc_download.php/3274/pdf/imm3274.pdf) for the derivative of the dot product.


<br>
## Chain rule

By the chain rule, the sensitivity of cost $$H(\mathbf y, \mathbf s(\mathbf x))$$ to the input to the softmax layer $$\mathbf x$$ is given by a simple gradient-Jacobian product, each of which we've already comptued:

$$\nabla_{\mathbf x} H = \nabla_{\mathbf s(\mathbf x)} H(\mathbf y, \mathbf s(\mathbf x)) \ \mathbf J_{\mathbf x}.$$

The first term is the gradient of cross-entropy cost to softmax activation. Remember that we're using row gradients. The second term is the Jacobian of softmax activation to softmax input. Expanding and simplifying, we get


<script type="math/tex; mode=display">
\begin{aligned}
\nabla_{\mathbf x} H
&= -\frac{\mathbf y}{\mathbf s(\mathbf x)} \bigg[ \text{diag} \big(\mathbf s(\mathbf x)\big) - \mathbf s(\mathbf x)^\top \mathbf s(\mathbf x) \bigg] \\[1.6em]
&= \frac{\mathbf y}{\mathbf s(\mathbf x)}\mathbf s(\mathbf x)^\top \mathbf s(\mathbf x)  - \frac{\mathbf y}{\mathbf s(\mathbf x)} \ \text{diag} \big(\mathbf s(\mathbf x)\big) \\[1.6em]
&= \mathbf y \ \mathbf S(\mathbf x)^{\text{repeated row}} - \mathbf y \ \text{diag} \big(\mathbf 1\big) \\[1.6em]
&= \mathbf s(\mathbf x) - \mathbf{y}.
\end{aligned}
</script>

The last line follows from the fact that $$\mathbf y$$ was one-hot and applied to a matrix whose rows are identically our softmax distribution. But actually, any $$\mathbf y$$ whose elements sum to $$1$$ would satisfy the same property. To be more specific, the equation above would hold not just for one-hot $$\mathbf y$$, but for any $$\mathbf y$$ specifying a distribution over classes. $$\square$$
