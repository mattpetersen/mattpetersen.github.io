---
layout: post
title: "Linear algebra"
date: 2017-03-15
header: true
footer: true
comments: true
tags: linear algebra, vectors, dot product, matrix multiplication
---

This post explains the basics of linear algebra, just enough to get going with neural networks. If you'd like a more thorough cover, Goodfellow, Bengio, and Courville's [chapter](http://www.deeplearningbook.org/contents/linear_algebra.html)[^fn1] on the topic can be read in two or three hours, and Terence Tao's [course](http://www.math.ucla.edu/~tao/resource/general/115a.3.02f/) can be finished in a few weeks.

## Scalars, vectors, and matrices

We say that a variable $$a$$ is a _scalar_ if it takes on a single value. Most likely, all of the math you've encountered in your life so far has used scalars: from arithmatic, to algebra, and even most of calculus. The number $3$ is a scalar, as is the variable $x$ (whose value we simply don't know).

However, in linear algebra and neural networks, we usually work with _vectors_. A vector is just a list of scalars. We use bold font for vectors, to distinguish them. We can write a vector of length three as 

$$ \mathbf v = \left[\begin{matrix} a, \ b, \ c \end{matrix}\right] $$

where $$a$$, $$b$$, and $$c$$ are scalars. We can reference each entry of $$\mathbf{v}$$ by using a subscript,

$$ v_1 = a, \quad v_2 = b, \quad v_3 = c. $$

This subscript notation lets us rewrite $$\mathbf v$$ as

$$ \mathbf v = \left[\begin{matrix} v_1, \ v_2, \ v_3 \end{matrix}\right]. $$

Notice how we don't use bold font for each element, because the elements alone are scalars.

Now that we've defined scalars and vectors, we can define matrices. A matrix is just a list of vectors. We use capital letters for matrices. We can write a matrix as

$$ W = \left[\begin{matrix} a, \  b, \  c \\ d, \ e, \ f \end{matrix}\right] $$

Now, you might wonder whether $$W$$
---

### Citations

[^fn1]: Goodfellow, Ian; Bengio, Yoshua; and Courville, Aaron; _Deep Learning_, MIT Press, 2017, [deeplearningbook.org](http://www.deeplearningbook.org)

