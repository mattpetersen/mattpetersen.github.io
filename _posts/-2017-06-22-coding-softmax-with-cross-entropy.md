---
layout: post
title: "Coding softmax with cross-entropy"
date: 2017-06-22
header: true
footer: true
comments: true
tags: backpropogation, softmax, cross-entropy, neural networks, deep learning, python, coding
---


<br>
## Information

Information theory is built upon one intuition: that the information conveyed about a distribution $$P(x)$$ by an observation $$x$$ is inversely proportional to the probability of that observation.

$$ I (x) := \frac{1}{P(x)}. $$

**Problem:** We'd like the information from two independent observations to be additive. That is, if we observe $$x$$ and observe $$y$$, the information gained, $$I(x, y)$$, should equal $$I(x) + I(y)$$, but notice that

$$ I(x) + I(y) = \frac{1}{P(x)} + \frac{1}{P(y)} \neq \frac{1}{P(x)P(y)} = I(x, y). $$

**Solution:** Use logarithms in our definition

$$ I_P(x) := \log \frac{1}{P(x)}. $$

Then information is additive.

$$ I(x) + I(y) = \log \frac{1}{P(x)} + \log \frac{1}{P(y)} = \log \frac{1}{P(x)P(y)} = I(x, y) $$

where we used (in order), the definition of information, the property of logarithms, and the fact that $$x$$ and $$y$$ were assumed to be independent, such that $$P(x)P(y) = P(x, y)$$.



<br>
## Entropy

The entropy of a discrete distribution $$P$$ is just the expected information of one sample $$x$$ from that distribution

<script type="math/tex; mode=display">
\begin{aligned}
H(P) :&= \mathbb{E} \left[ I(x) \right] \\
&= \mathbb{E} \left[\log \frac{1}{P(x)}\right] \\
&= -\mathbb{E} \left[ \log P(x) \right] \\
&= - \sum_{x \in P} P(x) \log P(x)
\end{aligned}
</script>


<br>
## Cross-entropy

If we

$$\mathbf y = \begin{bmatrix} y_0,\ y_1,\ . . . \ ,\ y_9 \end{bmatrix}$$


### Invariance to scaling

Softmax is invariant to additively scaling every element of $$x$$ by the same constant $$c$$.

<script type="math/tex; mode=display">
\begin{aligned}
s(x + c)_i &= \frac{e^{x_i + c}}{\sum_{j=0}^{9} e^{x_j + c}}    \\[1.2em]
           &= \frac{e^{x_i}e^c}{\sum_{j=0}^{9} e^{x_j}e^c}      \\[1.2em]
           &= \frac{e^c e^{x_i}}{e^c \sum_{j=0}^{9} e^{x_j}}    \\[1.2em]
           &= \frac{e^{x_i}}{\sum_{j=0}^{9} e^{x_j}} = s(x)_i
\end{aligned}
</script>

That means we can protect softmax from numerical overflow by subtracting the maximum entry of $x$ from every element of $x$

$$ s(x - \max(x))_i = \frac{e^{x_i - \max(x)}}{\sum_{j=1}^{10} e^{x_j - \max(x)}} = \frac{e^{x_i}}{\sum_{j=1}^{10} e^{x_j}} = s(x)_i $$
