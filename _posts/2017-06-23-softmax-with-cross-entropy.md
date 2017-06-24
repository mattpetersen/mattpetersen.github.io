---
layout: post
title: "Softmax with cross-entropy"
date: 2017-06-23
header: true
footer: true
comments: true
tags: backpropogation, matrix calculus, softmax, cross-entropy, neural networks, deep learning
---


A matrix-calculus approach to deriving the sensitivity of cross-entropy cost to the weighted input to a softmax output layer. _We use row vectors and row gradients_, since typical neural network formulations let columns correspond to features, and rows correspond to examples. This means that the input to our softmax layer is a row vector with a column for each class.


<br>
## Softmax

Softmax is a vector-to-vector transformation that turns a row vector

<script type="math/tex; mode=display">
\mathbf x = \begin{bmatrix}
x_1 \ \ \, 
x_2 \ \ \,
... \ \ \,
x_n \end{bmatrix}
</script>

into a normalized row vector

<script type="math/tex; mode=display">
\mathbf s(\mathbf x) = \begin{bmatrix}
s(\mathbf x)_1 \ \ \,
s(\mathbf x)_2 \ \ \,
...            \ \ \,
s(\mathbf x)_n \end{bmatrix}
</script>

The transformation is described element-wise, where the $$i$$th output $$s(\mathbf x)_i$$ is a function of the entire input $$\mathbf x$$, and is given by

<script type="math/tex; mode=display">
s(\mathbf x)_i = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}
</script>

Softmax is nice because it turns $$\mathbf x$$ into a probability distribution.

* Each element $$s(\mathbf x)_i$$ is between $$0$$ and $$1$$.
* The elements $$s(\mathbf x)_i$$ sum to $$1$$.


<br>
## Jacobian of softmax

Since softmax is a vector-to-vector transformation, its derivative is a Jacobian matrix. The Jacobian has a row for each output element $$s_i$$, and a column for each input element $$x_j$$. The entries of the Jacobian take two forms, one for the main diagonal entry, and one for every off-diagonal entry. We'll compute row $$i$$ of the Jacobian, which is the gradient of output element $$s_i$$ with respect to each of its input elements $$x_j$$.

First compute the diagonal entry of row $$i$$ of the Jacobian, that is, compute the derivative of the $$i$$'th output of softmax, $$s_i$$, with respect to its $$i$$'th input, $$x_i$$:

<script type="math/tex; mode=display">
\begin{aligned}
\frac{\partial s(\mathbf x)_i}{\partial x_i}
&= \frac{\sum_{j=1}^n e^{x_j} e^{x_i} - e^{x_i} e^{x_i}}{(\sum_{j=1}^n e^{x_j})^2} \\[1.6em]
&= s(\mathbf x)_i - s(\mathbf x)_i^2 \qquad \text{diagonal entry}
\end{aligned}
</script>

Now compute every off-diagonal entry of row $$i$$ of the Jacobian, that is, compute the derivative of the $$i$$'th output of softmax, $$s_i$$, with respect to its $$j$$'th input, $$x_j$$, where $$j \neq i$$.

<script type="math/tex; mode=display">
\begin{aligned}
\frac{\partial s(\mathbf x)_i}{\partial x_j}
&= \frac{\sum_{j=1}^n e^{x_j} \cdot 0 - e^{x_i} e^{x_j}}{(\sum_{j=1}^n e^{x_j})^2} \\[1.6em]
&= - s(\mathbf x)_i s(\mathbf x)_j \qquad \text{off-diagonal entry}
\end{aligned}
</script>

From now on, to keep things clear, we won't write dependence on $$\mathbf x$$. Instead we'll write $$\mathbf s(\mathbf x)$$ as $$\mathbf s$$ and $$s(\mathbf x)_i$$ as $$s_i$$, understanding that $$\mathbf s$$ and $$s_i$$ are each a function of the entire vector $$\mathbf x$$.

The form of the off-diagonals tells us that the Jacobian of softmax is a symmetric matrix. This is nice because symmetric matrices have great numeric and analytic properties. We expand it below. Each row is a gradient of one output element $$s_i$$ with respect to each of its input elements $$x_j$$.

<script type="math/tex; mode=display">
\mathbf J_{\mathbf x}(\mathbf s) = \begin{bmatrix}
s_0 - s_0^2 &  -s_0 s_1   &    ...   &  -s_0 s_n     \\[0.5em]
 -s_1 s_0   & s_1 - s_1^2 &    ...   &  -s_1 s_n     \\[0.5em]
   ...      &    ...      &    ...   &    ...        \\[0.5em]
 -s_n s_0   &  -s_n s_1   &    ...   & s_n - s_n^2   \end{bmatrix}
</script>

Notice that we can express this matrix as

<script type="math/tex; mode=display">
\mathbf J_{\mathbf x}(\mathbf s)
= \text{diag}(\mathbf s) - \mathbf s^\top \mathbf s
</script>

where the second term is the $$n \times n$$ outer product, because we defined $$\mathbf s$$ as a row vector.


<br>
## Cross-entropy

Cross-entropy measures the difference between two probability distributions. We saw that $$\mathbf s$$ is a distribution. The correct class is also a distribution if we encode it as a one-hot vector:

<script type="math/tex; mode=display"> \begin{aligned}
\mathbf y &= \begin{bmatrix}
        y_1  \ \ \,
        y_2  \ \ \,
        ...  \ \ \,
        y_n  \end{bmatrix} \\[1.1em]
&=  \begin{bmatrix}
0   \ \ \,
0   \ \ \,
... \ \ \,
1   \ \ \,
... \ \ \,
0   \end{bmatrix}
\end{aligned} </script>

where the $$1$$ appears at the index of the correct class of this single example.

The cross-entropy between our predicted distribution over classes, $$\mathbf s$$, and the true distribution over classes, $$\mathbf y$$, is a scalar measure of their difference, which is perfect for a cost function. It'll drive our softmax distribution toward the one-hot distribution. We can write this cost function as

<script type="math/tex; mode=display">
\begin{aligned}
H(\mathbf y, \mathbf s)
&= -\sum_{i=1}^n y_i \log s_i \\[1.6em] 
&= -\mathbf y \log \mathbf s^\top
\end{aligned}
</script>

which is the dot product since we're using row vectors. This formula comes from information theory. It measures the information gained about our softmax distribution when we sample from our one-hot distribution.


<br>
## Gradient of cross-entropy

Since our $$\mathbf y$$ is given and fixed, cross-entropy is a vector-to-scalar function of only our softmax distribution. That means it will have a gradient with respect to our softmax distribution. This vector-to-scalar cost function is actually made up of two steps: (1) a vector-to-vector element-wise $$\log$$ and (2) a vector-to-scalar dot product. The vector-to-vector logarithm will have a Jacobian, but since it's applied element-wise, the Jacobian will be diagonal, holding each elementwise derivative. The gradient of a dot product, being a linear operation, is just the vector $$\mathbf y$$.

<script type="math/tex; mode=display">
\begin{aligned}
\nabla_{\mathbf s} H 
&= -\nabla_{\mathbf s} \mathbf y \log \mathbf s^\top                \\[1.6em]
&= -\mathbf y \nabla_{\mathbf s} \log \mathbf s                     \\[1.1em]
&= -\mathbf y \ \text{diag}\left(\frac{\mathbf 1}{\mathbf s}\right) \\[1.6em]
&= -\frac{\mathbf y}{\mathbf s}
\end{aligned}
</script>

where we used equation (69) of [the matrix cookbook](http://www2.imm.dtu.dk/pubdb/views/edoc_download.php/3274/pdf/imm3274.pdf) for the derivative of the dot product.


<br>
## Finishing up with the chain rule

By the chain rule, the sensitivity of cost $$H$$ to the input to the softmax layer, $$\mathbf x$$, is given by a gradient-Jacobian product, each of which we've already computed:

<script type="math/tex; mode=display">
\nabla_{\mathbf x} H = \nabla_{\mathbf s} H \ \mathbf J_{\mathbf x}(\mathbf s)
</script>

The first term is the gradient of cross-entropy cost to softmax activation. The second term is the Jacobian of softmax activation to softmax input. Remember that we're using row gradients - this is a row vector times a matrix, resulting in a row vector. Expanding and simplifying, we get

<script type="math/tex; mode=display"> \begin{aligned}
\nabla_{\mathbf x} H
&= -\frac{\mathbf y}{\mathbf s} 
    \bigg[\text{diag}(\mathbf s)
   -\mathbf s^\top \mathbf s \bigg]                         \\[1.6em]
&= \frac{\mathbf y}{\mathbf s} \mathbf s^\top \mathbf s
 - \frac{\mathbf y}{\mathbf s} \ \text{diag}(\mathbf s)     \\[1.6em]
&= \mathbf y \ \mathbf S^{\text{repeated row}} - \mathbf y  \\[1.6em]
&= \mathbf s - \mathbf y
\end{aligned} </script>

The last line follows from the fact that $$\mathbf y$$ was one-hot and applied to a matrix whose rows are identically our softmax distribution. But actually, any $$\mathbf y$$ whose elements sum to $$1$$ would satisfy the same property. To be more specific, the equation above would hold not just for one-hot $$\mathbf y$$, but for any $$\mathbf y$$ specifying a distribution over classes.


<br>
# Now with a batch of examples
---
Our work thus far considered a single example. Hence $$\mathbf x$$, our input to the softmax layer, was a row vector. Alternatively, if we feed forward a batch of $$m$$ examples, then $$\mathbf X$$ contains a row vector for each example in the minibatch.

<script type="math/tex; mode=display">
\mathbf X =   \begin{bmatrix}
\mathbf x_1   \\[0.4em]
\mathbf x_2   \\[0.4em]
        ...   \\[0.4em]
\mathbf x_m   \end{bmatrix} \sim m \times n
</script>


<br>
## Softmax

Softmax is still a vector-to-vector transformation, but it's applied independently to each row of $$\mathbf X$$.

<script type="math/tex; mode=display">
\mathbf S =              \begin{bmatrix}
\mathbf s(\mathbf x_1)   \\[0.6em]
\mathbf s(\mathbf x_2)   \\[0.6em]
        ...              \\[0.6em]
\mathbf s(\mathbf x_m)   \end{bmatrix} \sim m \times n
</script>


<br>
## Jacobian of softmax
Because rows are independently mapped, the Jacobian of row $$i$$ of $$\mathbf S$$ with respect to row $$j \neq i$$ of $$\mathbf X$$ is a zero matrix.

<script type="math/tex; mode=display">
\mathbf J_{\mathbf x_{j \neq i}}(\mathbf s_i) = \mathbf 0
</script>

and the Jacobian of row $$i$$ of $$\mathbf S$$ with respect to row $$i$$ of $$\mathbf X$$ is

<script type="math/tex; mode=display">
\mathbf J_{\mathbf x_{i}}(\mathbf s_i) = \text{diag}(\mathbf s_i) - \mathbf s_i^\top \mathbf s_i
</script>

That means our Jacobian of $$\mathbf S$$ with respect to $$\mathbf X$$ is a diagonal $$m \times m$$ matrix of $$n \times n$$ matrices.

<script type="math/tex; mode=display">
\mathbf J_{\mathbf X}(\mathbf S) = \begin{bmatrix}
\mathbf J_{\mathbf x_1}(\mathbf s_1) & \mathbf 0 & ... & \mathbf 0 \\[1.1em]
\mathbf 0 & \mathbf J_{\mathbf x_2}(\mathbf s_2) & ... & \mathbf 0 \\[1.1em]
... & ... & ... & ...                                              \\[1.1em]
\mathbf 0 & \mathbf 0 & ... & \mathbf J_{\mathbf x_m}(\mathbf s_m)
\end{bmatrix} </script>

<br>
## Cross-entropy

Let each row of $$\mathbf Y$$ be a one-hot label for an example:

<script type="math/tex; mode=display">
\mathbf Y =   \begin{bmatrix}
\mathbf y_1   \\[0.4em]
\mathbf y_2   \\[0.4em]
        ...   \\[0.4em]
\mathbf y_m   \end{bmatrix} \sim m \times n
</script>

Then we compute the _mean cross-entropy_ by averaging the cross-entropy of every matching pair of rows of $$\mathbf Y$$ and $$\mathbf S$$. That is, we average over examples, the cross-entropy of each example:

<script type="math/tex; mode=display"> \begin{aligned}
H(\mathbf Y, \mathbf S)
&= -\frac{1}{m} \sum_{i=1}^m \mathbf y_i \log \mathbf s_i^\top  \\[1.6em]
&= -\frac{1}{m} \text{Tr}(\mathbf Y \log \mathbf S^\top)
\end{aligned} </script>

The above simplification works because each row of $$\mathbf S$$ is $$\mathbf s_i$$. So each column of $$\mathbf S^\top$$ is $$\mathbf s_i$$. So the matrix product $$\mathbf Y \log \mathbf S^\top$$ dots rows of $$\mathbf Y$$ with columns of $$(\log \mathbf S^\top)$$, which is exactly what we want for cross-entropy. Now, we only care about entries where the row index equals the column index. That's because cross-entropy sums the dot products of _matching_ rows of $$\mathbf Y$$ and $$\mathbf S$$. We can sum over matching dot products by using a trace.

__Note:__ this formulation is computationally wasteful. We shouldn't implement batch cross-entropy this way in a computer. We're only using it for its analytic simplicity to work out the backpropogating error. However, the end analytic result is actually computationally efficient.

<br>
## Gradient of cross-entropy

Since mean cross-entropy maps a matrix to a scalar, its Jacobian with respect to $$\mathbf S$$ will be a matrix.

<script type="math/tex; mode=display"> \begin{aligned}
\mathbf J_{\mathbf S}(H) 
&= \frac{\partial}{\partial \mathbf S} H                              \\[1.6em]
&= -\frac{\partial}{\partial \mathbf S}
   \bigg(\frac{1}{m} \text{Tr}(\mathbf Y \log \mathbf S^\top) \bigg)  \\[1.6em]
&= -\frac{1}{m} \frac{\partial}{\partial \mathbf S}
   \Big(\text{Tr}(\mathbf Y \log \mathbf S^\top) \Big)                \\[1.6em]
&= -\frac{1}{m} \mathbf Y
   \frac{\partial}{\partial \mathbf S} \Big(\log \mathbf S \Big)      \\[1.6em]
&= -\frac{1}{m} \mathbf Y \odot \frac{1}{\mathbf S}                   \\[1.6em]
&= -\frac{1}{m} \frac{\mathbf Y}{\mathbf S}
\end{aligned} </script>

Since $$\log \mathbf S$$ is an element-wise operation mapping a matrix to a matrix, its Jacobian is a matrix of element-wise derivatives which we chain rule by a hadamard product, rather than a dot product.

<br>
#### Why this works

This procedure is always true for any element-wise operations. We can see this by concatenating the rows of $$\mathbf S$$. 

<script type="math/tex; mode=display">
\mathbf s = \begin{bmatrix}
\mathbf s_1   \ \,
\mathbf s_2   \ \,
        ...   \ \,
\mathbf s_m   \end{bmatrix}
</script>

such that $$\mathbf s$$ is a row vector of length $$m \cdot n$$. Then $$\log$$ is an element-wise vector-to-vector transformation again. So it has an $$m \cdot n \times m \cdot n$$ diagonal Jacobian matrix.

<script type="math/tex; mode=display">
\mathbf J_{\mathbf s}(\log \mathbf s) =
\text{diag}\left(\frac{\mathbf 1}{\mathbf s}\right)
</script>

If we flatten $$\mathbf Y$$ in the same way

<script type="math/tex; mode=display">
\mathbf y = \begin{bmatrix}
\mathbf y_1   \ \,
\mathbf y_2   \ \,
        ...   \ \,
\mathbf y_m   \end{bmatrix}
</script>

then we get

<script type="math/tex; mode=display">
H(\mathbf y, \mathbf s) = - \frac{1}{m} \mathbf y \log \mathbf s^\top
</script>

and so

<script type="math/tex; mode=display"> \begin{aligned}
\mathbf J_{\mathbf s}(H)
&= -\frac{1}{m} \mathbf y
   \frac{\partial}{\partial \mathbf s} \Big(\log \mathbf s \Big)      \\[1.6em]
&= -\frac{1}{m} \mathbf y
   \text{diag}\left(\frac{\mathbf 1}{\mathbf s}\right)                \\[1.6em]
&= -\frac{1}{m} \frac{\mathbf y}{\mathbf s}
\end{aligned} </script>

Now since $$\mathbf y$$ and $$\mathbf s$$ are each of length $$m \cdot n$$, we can reshape this formulation back into matrices, understanding that in both cases the division is element-wise:

<script type="math/tex; mode=display">
\mathbf J_{\mathbf s}(H) = -\frac{1}{m} \frac{\mathbf Y}{\mathbf S}
</script>

and we have our result. $$\square$$

<br>
## Finishing up with the chain rule

We apply the chain rule just as before. The only difference is that our gradient-Jacobian product is now a matrix-tensor product.

<script type="math/tex; mode=display"> \begin{aligned}
\nabla_{\mathbf X} H
&= \mathbf J_{\mathbf S}(H) \ \mathbf J_{\mathbf X}(\mathbf S)    \\[1.6em]
&= \bigg(-\frac{1}{m} \frac{\mathbf Y}{\mathbf S}\bigg)
   \bigg(\mathbf J_{\mathbf X}(\mathbf S) \bigg)                  \\[2.4em]
&= \frac{1}{m} \odot \begin{bmatrix}
-\mathbf y_1 / \mathbf s_1  \\[0.4em]
-\mathbf y_2 / \mathbf s_2  \\[0.4em]
...                        \\[0.4em]
-\mathbf y_m / \mathbf s_m
\end{bmatrix}
\begin{bmatrix}
\mathbf J_{\mathbf x_1}(\mathbf s_1) & \mathbf 0 & ... & \mathbf 0    \\[0.4em]
\mathbf 0 & \mathbf J_{\mathbf x_2}(\mathbf s_2) & ... & \mathbf 0    \\[0.4em]
... & ... & ... & ...                                                 \\[0.4em]
\mathbf 0 & \mathbf 0 & ... & \mathbf J_{\mathbf x_m}(\mathbf s_m)
\end{bmatrix}                                                         \\[0.4em]
\\[1.1em]
&= \frac{1}{m} \odot \begin{bmatrix}
-\frac{\mathbf y_1}{\mathbf s_1} \mathbf J_{\mathbf x_1}(\mathbf s_1)  \\[1.4em]
-\frac{\mathbf y_2}{\mathbf s_2} \mathbf J_{\mathbf x_2}(\mathbf s_2)  \\[1.4em]
...                                                                   \\[1.4em]
-\frac{\mathbf y_m}{\mathbf s_m} \mathbf J_{\mathbf x_m}(\mathbf s_m)
\end{bmatrix}                                                         \\[1.4em]
\\[1.1em]
&= \frac{1}{m} \odot       \begin{bmatrix}
\mathbf s_1 - \mathbf y_1  \\[0.4em]
\mathbf s_2 - \mathbf y_2  \\[0.4em]
...                        \\[0.4em]
\mathbf s_m - \mathbf y_m  \end{bmatrix}                              \\[1.4em]
\\[1.1em]
&= \frac{1}{m} \odot \Big(\mathbf S - \mathbf Y\Big)
\end{aligned} </script>

So the sensitivity of cost to the weighted input to our softmax layer is just the difference of our softmax matrix and our matrix of one-hot labels, where every element is divided by the number of examples in the batch. $$\quad \square$$