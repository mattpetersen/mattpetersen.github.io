---
layout: post
title: "Linear algebra"
date: 2017-03-15
header: true
footer: false
comments: true
tags: linear algebra, vectors, dot product, matrix multiplication
---

This post explains the basics of linear algebra, just enough to get going with neural networks. If you'd like a more thorough introduction, check out Goodfellow, Bengio, and Courville's [chapter](http://www.deeplearningbook.org/contents/linear_algebra.html) on the topic[^fn1].

We start by introducing a _scalar_. We say that a variable $$a$$ is a scalar if its value is some fixed real number. Most likely, all of the math you've encountered in your life so far has used scalars variables: from arithmatic, to algebra, and even most of calculus. We can define a scalar like

<dtex> a = 5 </dtex>

Next we define _vectors_. A vector is just a list of scalars. We usually use bold font to distinguish vectors from scalars. We can write a vector of length three as 

<dtex> \mathbf v = \left[\begin{matrix} a, \ b, \ c \end{matrix}\right] </dtex>

where $$a$$, $$b$$, and $$c$$ are each a scalar. Importantly, the order of the entries matters. We can reference each entry of $$\mathbf{v}$$ by using a subscript

<dtex> v_1 = a, \quad v_2 = b, \quad v_3 = c </dtex>

or rather

<dtex> \mathbf v = \left[\begin{matrix} v_1, \ v_2, \ v_3 \end{matrix}\right] </dtex>

Notice how we don't use bold here. That's because each element alone is a scalar :)

Now that we've defined scalars and vectors

---

### Citations

[^fn1]: Goodfellow, Ian; Bengio, Yoshua; and Courville, Aaron; _Deep Learning_, MIT Press, 2017, [deeplearningbook.org](http://www.deeplearningbook.org)

