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

From now on, to keep things clear, we won't write dependence on $$\mathbf x$$. Instead we'll write $$\mathbf s(\mathbf x)$$ as $$\mathbf s$$ and $$s(\mathbf x)_i$$ as $$s_i$$, understanding that $$\mathbf s$$ and $$s_i$$ are each a function of the entire vector $$\mathbf x$$.


<br>
## Jacobian of softmax

Since softmax is a vector-to-vector transformation, its derivative is a Jacobian matrix. The Jacobian has a row for each output element $$s_i$$, and a column for each input element $$x_j$$. The entries of the Jacobian take two forms, one for the main diagonal entry, and one for every off-diagonal entry. We'll compute row $$i$$ of the Jacobian, which is the gradient of output element $$s_i$$ with respect to each of its input elements $$x_j$$.

First compute the diagonal entry of row $$i$$ of the Jacobian, that is, compute the derivative of the $$i$$'th output of softmax, $$s_i$$, with respect to its $$i$$'th input, $$x_i$$:

<script type="math/tex; mode=display">
\begin{aligned}
\frac{\partial s_i}{\partial x_{i}} &= \frac{\sum_{j=1}^{n} e^{x_j} e^{x_i} - e^{x_i} e^{x_i}}{(\sum_{j=1}^{n} e^{x_j})^2} \\[1.6em]
&= s_i - s_i^2 \qquad \text{diagonal entry}
\end{aligned}
</script>

Now compute every off-diagonal entry of row $$i$$ of the Jacobian, that is, compute the derivative of the $$i$$'th output of softmax, $$s_i$$, with respect to its $$j$$'th input, $$x_j$$, where $$j \neq i$$.

<script type="math/tex; mode=display">
\begin{aligned}
\frac{\partial s_i}{\partial x_j} &= \frac{\sum_{j=1}^{n} e^{x_j} \cdot 0 - e^{x_i} e^{x_j}}{(\sum_{j=1}^{n} e^{x_j})^2} \\[1.6em]
&= - s_i s_j \qquad \text{off-diagonal entry}
\end{aligned}
</script>

The form of the off-diagonals tells us that the Jacobian of softmax is a symmetric matrix. This is nice because symmetric matrices have great numeric and analytic properties. We expand it below. Each row is a gradient of one output element $$s_i$$ with respect to each of its input elements $$x_j$$.

$$
\mathbf J_{\mathbf x}(\mathbf s) =
\begin{bmatrix}
    \nabla s_0 \\[0.5em]
    \nabla s_1 \\[0.5em]
    . . . \\
    \nabla s_n
\end{bmatrix}
$$

<script type="math/tex; mode=display">
= \begin{bmatrix}
    s_0 - s_0^2 & -s_0 s_1 & . . .  & -s_0 s_n \\[0.5em]
    -s_1 s_0 & s_1 - s_1^2 & . . .   & -s_1 s_n \\[0.5em]
    . . . & . . .  & . . . & . . . \\[0.5em]
    -s_n s_0 & -s_n s_1 & . . .  & s_n - s_n^2
\end{bmatrix}
</script>

Notice that we can express this matrix as

$$\mathbf J_{\mathbf x}(\mathbf s) = \text{diag} (\mathbf s) - \mathbf s^\top \mathbf s$$

where the second term is the $$n \times n$$ outer product.


<br>
## Cross-entropy

Cross-entropy measures the difference between two probability distributions. We saw that $$\mathbf s$$ is a distribution. The correct class is also a distribution, that is, assuming we encode it as a one-hot vector:

<script type="math/tex; mode=display">
\begin{aligned}
\mathbf y &= \begin{bmatrix} y_1,\ y_2,\ . . . \ ,\ y_n \end{bmatrix} \\[1.6em]
          &= \begin{bmatrix} 0,\ 0,\ . . . \ , \ 1,\ . . . \ , \ 0 \end{bmatrix}
\end{aligned}
</script>

where the $$1$$ appears at the index of the correct class.

The cross-entropy between our predicted distribution over classes, $$\mathbf s( \mathbf x)$$, and the true distribution over classes, $$\mathbf y$$, is a scalar measure of their difference, which is perfect for a cost function. It'll drive our softmax distribution toward the one-hot distribution. We can write this cost function as

<script type="math/tex; mode=display">
\begin{aligned}
H(\mathbf y, \mathbf s)
&= -\sum_{i=1}^n y_i \log s_i \\[1.6em] 
&= -\mathbf y \log \mathbf s^\top,
\end{aligned}
</script>

which is the dot product since we're using row vectors. This formula comes from information theory. It measures the information gained about our softmax distribution when we sample from our one-hot distribution.


<br>
## Gradient of cross-entropy

Since our $$\mathbf y$$ is given and fixed, cross-entropy is a vector-to-scalar function of only our softmax distribution. That means it will have a gradient with respect to our softmax distribution. This vector-to-scalar cost function is actually made up of two steps: (1) a vector-to-vector element-wise $$\log$$ and (2) a vector-to-scalar dot product. The vector-to-vector logarithm will have a Jacobian, but since it's applied element-wise, the Jacobian will be diagonal, holding each elementwise derivative. The gradient of the dot product operation is matrix multiplied on the left of the Jacobian of the elementwise logarithm in the part below:

<script type="math/tex; mode=display">
\begin{aligned}
\nabla_{\mathbf s} H 
&= -\nabla_{\mathbf s} \mathbf y \log \mathbf s^\top \\[1.6em]
&= -\mathbf y \nabla_{\mathbf s} \log \mathbf s \\[1.6em]
&= -\mathbf y \ \text{diag}\left(\frac{\mathbf 1}{\mathbf s}\right) \\[1.6em]
&= -\frac{\mathbf y}{\mathbf s},
\end{aligned}
</script>

where we used equation (69) of [the matrix cookbook](http://www2.imm.dtu.dk/pubdb/views/edoc_download.php/3274/pdf/imm3274.pdf) for the derivative of the dot product.


<br>
## Combining

By the chain rule, the sensitivity of cost $$H$$ to the input to the softmax layer $$\mathbf x$$ is given by a simple gradient-Jacobian product, each of which we've already comptued:

$$\nabla_{\mathbf x} H = \nabla_{\mathbf s} H \ \mathbf J_{\mathbf x}(\mathbf s).$$

The first term is the gradient of cross-entropy cost to softmax activation. Remember that we're using row gradients. The second term is the Jacobian of softmax activation to softmax input. Expanding and simplifying, we get


<script type="math/tex; mode=display">
\begin{aligned}
\nabla_{\mathbf x} H
&= -\frac{\mathbf y}{\mathbf s} \bigg[ \text{diag} \big(\mathbf s\big) - \mathbf s^\top \mathbf s \bigg] \\[1.6em]
&= \frac{\mathbf y}{\mathbf s}\mathbf s^\top \mathbf s  - \frac{\mathbf y}{\mathbf s} \ \text{diag} \big(\mathbf s\big) \\[1.6em]
&= \mathbf y \ \mathbf S^{\text{repeated row}} - \mathbf y \ \text{diag} \big(\mathbf 1\big) \\[1.6em]
&= \mathbf s - \mathbf{y}.
\end{aligned}
</script>

The last line follows from the fact that $$\mathbf y$$ was one-hot and applied to a matrix whose rows are identically our softmax distribution. But actually, any $$\mathbf y$$ whose elements sum to $$1$$ would satisfy the same property. To be more specific, the equation above would hold not just for one-hot $$\mathbf y$$, but for any $$\mathbf y$$ specifying a distribution over classes.


<br>
## Batch of examples

Our work thus far considered a single example. Hence $$\mathbf x$$, our input to the softmax layer, was a row vector. Alternatively, if we feed forward a batch of $$m$$ examples, then $$\mathbf X$$ contains a row vector for each example in the minibatch.

$$ \mathbf X
= \begin{bmatrix}
\mathbf x_1 \\[0.4em]
\mathbf x_2 \\[0.4em]
    . . .     \\[0.4em]
\mathbf x_m
\end{bmatrix} = m \times n $$

Softmax is still a vector-to-vector transformation, but it's applied independently to each row of $$\mathbf X$$.

$$ \mathbf S
= \begin{bmatrix}
\mathbf s(\mathbf x_1) \\[0.4em]
\mathbf s(\mathbf x_2) \\[0.4em]
    . . .                \\[0.4em]
\mathbf s(\mathbf x_m)
\end{bmatrix} = m \times n $$

Since we do one vector-to-vector softmax on each row of $$\mathbf X$$, we have $$m$$ Jacobian matrices. That is, we have one Jacobian matrix for each example in our minibatch. We can line these $$m$$ Jacobian matrices up as a vector, noting that each Jacobian itself is $$n \times n$$.

$$ \mathbf J_{\mathbf X}(\mathbf S)
= \begin{bmatrix}
\mathbf J_{\mathbf x_1}(\mathbf s_1) \\[0.4em]
\mathbf J_{\mathbf x_2}(\mathbf s_2) \\[0.4em]
...                                 \\[0.4em]
\mathbf J_{\mathbf x_m}(\mathbf s_m) 
\end{bmatrix} = m \times (n \times n) $$

It's important to note that we can only do this because our rows are independently softmaxed. If not, we'd have a $$4$$-dimensional Jacobian running around, because we'd need the derivative of each output element of a matrix, with respect to each input element of a matrix. What a mess.

Luckily, we can exploit the same trick for our cross-entropy, because cross entropy applies independently to each row of $$\mathbf S$$. First we let each row of $$\mathbf Y$$ be a one-hot label for an example.

$$ \mathbf Y
= \begin{bmatrix}
\mathbf y_1 \\[0.4em]
\mathbf y_2 \\[0.4em]
    . . .     \\[0.4em]
\mathbf y_m
\end{bmatrix} = m \times n $$

Then we compute the mean cross-entropy by averaging the cross-entropy of each pair of rows:

$$ H(\mathbf Y, \mathbf S) = \frac{1}{m} \sum_{i=1}^m \mathbf y_i \log \mathbf s_i $$

Since mean cross-entropy maps a matrix to a scalar row-wise, its Jacobian with respect to $$\mathbf S$$ will be a matrix whose rows are our familiar gradient vectors from before:

$$ \mathbf J_{\mathbf S}(H)
= \frac{1}{m} \begin{bmatrix}
-\mathbf y_1 / \mathbf s_1 \\[0.4em]
-\mathbf y_2 / \mathbf s_2 \\[0.4em]
    ...                          \\[0.4em]
-\mathbf y_m / \mathbf s_m
\end{bmatrix} = m \times n $$

Now we combine with our chain rule just as before. The only difference is that our gradient-Jacobian product is now a matrix-tensor product.

$$\nabla_{\mathbf X} H = \mathbf J_{\mathbf S}(H) \ \mathbf J_{\mathbf X}(\mathbf S).$$

<script type="math/tex; mode=display">
\begin{aligned}
&= \frac{1}{m} \begin{bmatrix}
-\mathbf y_1 / \mathbf s_1 \\[0.4em]
-\mathbf y_2 / \mathbf s_2 \\[0.4em]
    ...                          \\[0.4em]
-\mathbf y_m / \mathbf s_m
\end{bmatrix}
\begin{bmatrix}
\mathbf J_{\mathbf x_1}(\mathbf s_1) \\[0.4em]
\mathbf J_{\mathbf x_2}(\mathbf s_2) \\[0.4em]
...                                  \\[0.4em]
\mathbf J_{\mathbf x_m}(\mathbf s_m)
\end{bmatrix} \\[1.6em]
\\[0.4em] &= (m \times n) \cdot (m \times n) \times n \\[1.6em]
&= 1 \times n
\end{aligned}
</script>

This looks confusing, but if we break it down, we simply dot, for each of our $$m$$ examples, the $$m$$'th row of $$\mathbf J_{\mathbf S}(H)$$, against the $$m$$'th matrix of $$\mathbf J_{\mathbf X}(\mathbf S)$$, and then sum the resulting row vectors. We saw previously that each of these individual gradient-Jacobian products is given by

$$\mathbf s_i - \mathbf y_i$$

Summing the resulting vectors and remembering our scalar of $$\frac{1}{m}$$ we get

<script type="math/tex; mode=display">
\begin{aligned}
&\frac{1}{m} \sum_{i=1}^m \mathbf s_i - \mathbf y_i \\[1.6em]
= &\frac{1}{m} \sum_{\text{rows}} \mathbf S - \mathbf Y \\[1.6em]
= &1 \times n
\end{aligned}
</script>

So when we use average cross-entropy cost after a softmax output layer, the sensitivity of cost with respect to the weighted input to the softmax layer is just the row-average of the difference of our softmax output $$\mathbf S$$ from the true output $$\mathbf Y$$. Since each row corresponds to one example in our batch, we're simply averaging the individual gradients of the examples. We get this nice result thanks to the fact that both softmax and cross-entropy are row-to-row operations. $$\square$$
