---
layout: post
title: "Deriving Newton's Method"
date: 2017-07-06
header: true
footer: true
comments: true
tags: optimization, Newton's method, Taylor series, quadratic approximation
---

This post and others are heavily inspired by Goodfellow's Deep Learning [book](http://www.deeplearningbook.org). Newton's method is just a smarter version of gradient descent. Let $$J(\theta)$$ be our cost function. Gradient descent follows the gradient directly downhill from the previous point:

<script type="math/tex; mode=display">
\theta_k = \theta_{k-1} - \nabla J(\theta_{k-1}) \quad \text{SGD}
</script>

There's one problem. The direction pointing directly downhill changes as we move.

<center><img src='http://trond.hjorteland.com/thesis/img208.gif' style='width:60%;object-fit: contain'/></center>

Newton's avoids this wasteful naivety by considering how the gradient will change as we move. Since the Hessian is the gradient of the gradient, we just left-multiply our step by the inverse Hessian:

<script type="math/tex; mode=display">
\theta_k = \theta_{k-1} - H^{-1} \nabla J(\theta_{k-1}) \quad \text{Newton's}
</script>

So Newton's just rescales each element of the gradient by the weighted sum of its second derivatives. We can derive Newton's method with a few lines of calculus.

<br>
## Taylor approximation

Remember that the Taylor polynomial approximates a function from the perspective of a given point. Here's a GIF showing how the Taylor polynomial matches more and more of $$\sin(x)$$ as we add terms, from the perspective of $$x = 0$$.

<center><img src='http://mathforum.org/mathimages/imgUpload/thumb/Taylor_Main.gif/400px-Taylor_Main.gif' style='width:40%;object-fit: contain'/></center>

Well, it takes a _lot_ of terms. You'll be surprised to hear that in a lot of optimization we only use two terms. Yeah, this is a bad approximation, but it works as long as we don't move too far. Besides, after we move, we'll make a new Taylor approximation from our new point, which will suit us for another small move from there.


<br>
## Deriving Newton's from the Taylor expansion

If we look at the second-order Taylor polynomial from the perspective of $$\theta_0$$, we have

<script type="math/tex; mode=display">
J(\theta_0) + (\theta - \theta_0) \nabla J(\theta_0) + \frac{1}{2} (\theta - \theta_0) H (\theta - \theta_0)^\top
</script>

This approximates our cost function $$J(\theta)$$, as long as we don't move too far from our perspective point $$\theta_0$$. Since we're trying to minimize our cost, we'll set its gradient equal to zero. That means, in one step, we'll move in the direction and distance that gets us to a flat gradient, assuming our quadratic approximation is accurate up until that point. This assumption can be good or bad. In deep learning it can end up jumping us to saddle points. A quick note: we're using column gradients below, so $$(\theta - \theta_0)^\top$$ is a row vector which forms a dot product with $$J(\theta_0)$$.

<br>
### Finding the gradient

The gradient of the second-order Taylor approximation of the cost function is
<script type="math/tex; mode=display">
\nabla \left[ J(\theta_0) + (\theta - \theta_0)^\top \nabla J(\theta_0) + \frac{1}{2} (\theta - \theta_0)^\top H (\theta - \theta_0) \right]
</script>
and by linearity of the gradient operator
<script type="math/tex; mode=display">
\nabla J(\theta_0) + \nabla \left[ (\theta - \theta_0)^\top \nabla J(\theta_0) \right] + \frac{1}{2} \nabla \left[ (\theta - \theta_0)^\top H (\theta - \theta_0) \right]
</script>
The first term is zero since $$\theta_0$$ is fixed and we're taking the gradient with respect to $$\theta$$.

By linearity of transpose
<script type="math/tex; mode=display">
\nabla \left[ (\theta^\top - \theta_0^\top) \nabla J(\theta_0) \right] + \frac{1}{2} \nabla \left[ (\theta^\top - \theta_0^\top) H (\theta - \theta_0) \right]
</script>
and linearity of matrix product
<script type="math/tex; mode=display">
\nabla \left[ \theta^\top \nabla J(\theta_0) - \theta_0^\top \nabla J(\theta_0)  \right] + \frac{1}{2} \nabla \left[ \theta^\top H (\theta - \theta_0) - \theta_0^\top H (\theta - \theta_0) \right]
</script>
and again upon the right side
<script type="math/tex; mode=display">
\nabla \left[ \theta^\top \nabla J(\theta_0) - \theta_0^\top \nabla J(\theta_0)  \right] + \frac{1}{2} \nabla \left[\theta^\top H \theta - \theta^\top H \theta_0 - \theta_0^\top H \theta + \theta_0^\top H \theta_0 \right]
</script>
Distributing and applying the gradient operators gives
<script type="math/tex; mode=display">
\nabla J(\theta_0) + \frac{1}{2} \left[2 H \theta - H \theta_0 - (\theta_0^\top H)^\top + 0 \right]
</script>
and distributing the transpose across the product
<script type="math/tex; mode=display">
\nabla J(\theta_0) + \frac{1}{2} \left[2 H \theta - H \theta_0 - H^
\top \theta_0 \right]
</script>
and exploiting the fact that the Hessian is symmetric such that $$H^\top = H$$
<script type="math/tex; mode=display">
\begin{aligned}
\nabla J(\theta_0) + \frac{1}{2} \left[2 H \theta - 2 H \theta_0 \right] \\[1.6em]
\nabla J(\theta_0) + H(\theta - \theta_0)
\end{aligned}
</script>

<br>
### Setting the gradient to zero

So this is the gradient of our quadratic approximation of our cost function $$J(\theta)$$ when evaluated at our current parameter setting $$\theta_0$$. If we set this equation equal to zero and solve for $$\theta$$ we get the new parameter setting that sets our gradient equal to zero when exploiting 
<script type="math/tex; mode=display">
\begin{aligned}
\nabla J(\theta_0) + H(\theta - \theta_0) := 0 \\[1.6em]
\nabla J(\theta_0) + H \theta - H \theta_0 = 0 \\[1.6em]
H \theta  = H \theta_0 - \nabla J(\theta_0) \\[1.6em]
\theta = \theta_0 - H^{-1} \nabla J(\theta_0)
\end{aligned}
</script>

So taking one Newton step puts us at a point of zero gradient, to the extent that our actual cost function reflects a quadratic function such that our second-order approximation is a good one. 

<br>
## Conclusion

Newton's method is gradient descent where we scale the gradient by the inverse Hessian. This gives Newton's foresight of how the gradient will change as we move, which lets us jump straight to the point where the gradient is zero if we so choose. A learning rate of $$1$$ will make this happen. A learning rate less than $$1$$ is more conservative.

Newton's has weakness. It will actually move uphill to a point of zero gradient, and that point could also be a saddle point rather than a local minimum. Additionally you have to use $$O(n^2)$$ operations to compute the Hessian. There are ways to fix these weaknesses, known as _quasi-Newton methods_. We'll explore these in future posts. $$\quad \square$$

<br>