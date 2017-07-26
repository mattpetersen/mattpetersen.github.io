---
layout: post
title: "Probability theory"
date: 2017-07-26
header: true
footer: true
comments: true
tags: probability theory, expectation, variance, definition of a random variable
---

This is mostly for my own review. I'll be updating it periodically. Eventually I'd like it to be a one-stop shop for brushing up on theoretical probability.

<br>
## Basics

A state space $$\mathcal{S}$$ is a set of outcomes.

<script type="math/tex; mode=display">
\mathcal{S} = \{\text{heads},\ \text{tails} \}
</script>

A random variable $$X$$ is a deterministic function from the state space to numbers.

<script type="math/tex; mode=display">
X(\text{heads}) = 1, \quad X(\text{tails}) = 0
</script>

A probability function assigns probabilities to those numbers

<script type="math/tex; mode=display">
p(1) = 0.5, \quad p(0) = 0.5
</script>



<br>
## Expectation

Expectation is a number, the center of mass of a probability function. It's a convex sum over all possible values $$x$$ of $$X$$, each weighted by its probability $$p(x)$$.

<script type="math/tex; mode=display">
\mathbb{E}[X] = \sum_{x \in X} p(x) x
</script>

The expectation of a constant is the same constant.

<script type="math/tex; mode=display">
\mathbb{E}[c] = \sum_{x \in X} p(x) c = c \sum_{x \in X} p(x) = c
</script>

Expectation is linear.

<script type="math/tex; mode=display">
\begin{aligned}
\mathbb{E}[aX + b]
&= \sum_{x \in X} p(x) (aX + b) \\[2.6em]
&= \sum_{x \in X} p(x) aX + \sum_{x \in X} p(x) b \\[2.6em]
&= a \sum_{x \in X} p(x) X + b \sum_{x \in X} p(x) \\[2.6em]
&= a \mathbb{E}[X] + b
\end{aligned}
</script>



<br>
## Variance

Variance is a number, the weighted average distance of any $$x \in X$$ from the mean $$\mathbb{E}[X]$$. The weights are $$p(x)$$ and the distance metric is squared $$L^2$$, or rather, squared Euclidean distance.

<script type="math/tex; mode=display">
\begin{aligned}
\text{Var}[X]
&= \mathbb{E} \Big[ \big( X - \mathbb{E}[X] \big)^2 \Big] \\[2.6em]
&= \mathbb{E} \Big[ X^2 - 2 \mathbb{E}[X] X + \mathbb{E}[X]^2 \Big] \\[2.6em]
&= \mathbb{E}[X^2] - 2 \mathbb{E}[X] \mathbb{E}[X] + \mathbb{E}[X]^2 \\[2.6em]
&= \mathbb{E}[X^2] - \mathbb{E}[X]^2
\end{aligned}
</script>

The variance of a constant is zero.

<script type="math/tex; mode=display">
\text{Var}[c] = \mathbb{E}[c^2] - \mathbb{E}[c]^2 = c^2 - [c]^2
</script>

Variance is **not** linear, but what follows is a useful property:

<script type="math/tex; mode=display">
\begin{aligned}
\text{Var}[aX + b]
&= \mathbb{E}[(aX + b)^2] - \mathbb{E}[aX + b]^2 \\[2.6em]
&= \mathbb{E}[a^2X^2 + 2abX + b^2] - [a\mathbb{E}[X] + b]^2 \\[2.6em]
&= a^2\mathbb{E}[X^2] + 2ab\mathbb{E}[X] + b^2 - a^2\mathbb{E}[X]^2 - 2ab\mathbb{E}[X] - b^2 \\[2.6em]
&= a^2\mathbb{E}[X^2] - a^2\mathbb{E}[X]^2 \\[2.6em]
&= a^2 \text{Var}[X]
\end{aligned}
</script>

Sometimes we write $$\sigma^2_x$$ for variance. The square root is called standard deviation
<script type="math/tex; mode=display">
\sigma_x = \sqrt{\text{Var}[X]}
</script>



<br>
## Covariance

Covariance is similar to variance. It's the weighted average product of the distance of $$X$$ from its mean, with the distance of $$Y$$ from its mean.

<script type="math/tex; mode=display">
\begin{aligned}
\text{Cov}(X, Y)
&= \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])] \\[2.6em]
&= \mathbb{E}[(XY - \mathbb{E}[X]Y - X\mathbb{E}[Y] + \mathbb{E}[X]\mathbb{E}[Y])] \\[2.6em]
&= \mathbb{E}[XY] - 2\mathbb{E}[X]\mathbb{E}[Y] + \mathbb{E}[X]\mathbb{E}[Y] \\[2.6em]
&= \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]
\end{aligned}
</script>

Covariance is symmetric.

<script type="math/tex; mode=display">
\begin{aligned}
\text{Cov}(X, Y)
&= \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y] \\[2.6em]
&= \mathbb{E}[YX] - \mathbb{E}[Y]\mathbb{E}[X] \\[2.6em]
&= \text{Cov}(Y, X)
\end{aligned}
</script>

The covariance of $$X$$ and a constant $$c$$ is zero.

<script type="math/tex; mode=display">
\begin{aligned}
\text{Cov}(X, c)
&= \mathbb{E}[Xc] - \mathbb{E}[X]\mathbb{E}[c] \\[2.6em]
&= c\mathbb{E}[X] - c\mathbb{E}[X] = 0
\end{aligned}
</script>

The covariance of $$X$$ and a linear function of $$X$$ is slope times variance.

<script type="math/tex; mode=display">
\begin{aligned}
\text{Cov}(X, aX + b)
&= \mathbb{E}[X(aX + b)] - \mathbb{E}[X]\mathbb{E}[aX + b] \\[2.6em]
&= \mathbb{E}[aX^2 + bX] - \mathbb{E}[X](a\mathbb{E}[X] + b) \\[2.6em]
&= a\mathbb{E}[X^2] + b\mathbb{E}[X] - a\mathbb{E}[X]^2 - b\mathbb{E}[X] \\[2.6em]
&= a\mathbb{E}[X^2] - a\mathbb{E}[X]^2 \\[2.6em]
&= a \text{Var}(X)
\end{aligned}
</script>



<br>
## Correlation

Correlation is covariance normalized to lie between zero and one.

<script type="math/tex; mode=display">
\rho = \frac{\text{Cov}(X, Y)}{\sigma_x \sigma_y}
</script>

The correlation of $$X$$ and a linear function of $$X$$ is one.

<script type="math/tex; mode=display">
\begin{aligned}
\rho
&= \frac{\text{Cov}(X, aX + b)}{\sigma_x \sigma_{aX + b}} \\[2.6em]
&= \frac{a\sigma^2_x}{\sigma_x \sqrt{a^2\sigma^2_x}} \\[2.6em]
&= \frac{a\sigma^2_x}{a \sigma^2_x} = 1
\end{aligned}
</script>

Correlation is a measure of **linear** relationship. It does not capture non-linear effects. As such, independence is a stricter condition than zero correlation. Independence implies zero covariance, but zero covariance does **not** imply independence. However, non-zero covariance implies dependence, specifically, at least some linear dependence.



<br>
## Estimation

The sample mean is unbiased.

<script type="math/tex; mode=display">
\begin{aligned}
\mathbb{E} \left[ \frac{1}{n} \sum_{x \in X} x \right]
&= \frac{1}{n} \sum_{x \in X} \mathbb{E}[x] \\[2.6em]
&= \frac{1}{n} \cdot n \mathbb{E}[X] \\[2.6em]
&= \mathbb{E}[X]
\end{aligned}
</script>