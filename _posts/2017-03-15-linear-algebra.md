---
layout: post
title: "Linear algebra"
date: 2017-03-15
header: true
footer: true
comments: true
tags: linear algebra, vectors, dot product, matrix multiplication
---

This post explains the basics of linear algebra, just enough to get going with neural networks. If you'd like a more thorough introduction, check out Goodfellow, Bengio, and Courville's [chapter](http://www.deeplearningbook.org/contents/linear_algebra.html) on the topic[^fn1], or Terence Tao's [course](http://www.math.ucla.edu/~tao/resource/general/115a.3.02f/). 

We start by introducing a _scalar_. We say that a variable $$a$$ is a scalar if its value is some fixed real number. Most likely, all of the math you've encountered in your life so far has used scalars: from arithmatic, to algebra, and even most of calculus.

Next we define _vectors_. A vector is just a list of scalars. We usually use bold font to distinguish vectors from scalars. We can write a vector of length three as 

$$ \mathbf v = \left[\begin{matrix} a, \ b, \ c \end{matrix}\right] $$

where $$a$$, $$b$$, and $$c$$ are each a scalar. We can reference each entry of $$\mathbf{v}$$ by using a subscript

$$ v_1 = a, \quad v_2 = b, \quad v_3 = c $$

Subscript notation gives us an alternative way to express $$\mathbf v$$,

$$ \mathbf v = \left[\begin{matrix} v_1, \ v_2, \ v_3 \end{matrix}\right] $$

Notice how we don't use bold font for each element, because the elements are scalars.

Now that we've defined scalars and vectors, we can define matrices.

---

### Citations

[^fn1]: Goodfellow, Ian; Bengio, Yoshua; and Courville, Aaron; _Deep Learning_, MIT Press, 2017, [deeplearningbook.org](http://www.deeplearningbook.org)

