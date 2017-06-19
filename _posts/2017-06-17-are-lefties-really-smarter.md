---
layout: post
title: "Are Lefties Really Smarter?"
date: 2017-06-17
header: true
footer: true
comments: true
tags: statistics, data science, linear regression, false positives, p hacking, left-handed, right-handed, intelligence, IQ, overfitting, underfitting
---

There's [a viral article](https://www.indy100.com/article/left-handed-people-smarter-science-ifl-science-maths-7797656) today that the science is finally settled and left-handed people are smarter than right-handed people. There are good reasons to be skeptical. For those curious, the source paper can be found [here](http://journal.frontiersin.org/article/10.3389/fpsyg.2017.00948/full). I'm not a neuroscientist so if I'm mistaken anywhere I would love to hear from you.

### 1 - Virality

Any story that's perfectly poised to make the internet do backflips deserves an extra ounce of skepticism. False positives tend to be [over-represented in viral studies](https://www.scientificamerican.com/article/an-epidemic-of-false-claims/).

### 2 - Vague assertions 

The justification for the finding was that hand function is
> "a manifestation of brain function and is therefore related to cognition."

This is so broad a statement as to be meaningless. Every action is a manifestation of brain function. Is a manifestation of brain function necessarily related to cognition? Is cognition necessarily related to intelligence? We have a lot of bridges to build here.

### 3 - False assumptions

While it is true that [lefties have more connections between brain hemispheres](http://science.sciencemag.org/content/229/4714/665.long), a property known as _lateralization_, there is actually [no significant link between lateralization and brain function](http://www.pnas.org/content/110/36/E3435.full.pdf). This may seem counterintuitive, because [labotomized patients lose 9.2 to 17 points of IQ](https://en.wikipedia.org/wiki/Talk%3ALobotomy#IQ_drop). These facts together point to diminishing returns of lateralization beyond a certain threshold. Our default assumption should be that there's no IQ benefit from the increased lateralization found in lefties, since we're comparing against regular people not lobotomized patients.

### 4 - Mixed effect direction

Lefty children perform [worse in math](http://onlinelibrary.wiley.com/doi/10.1111/j.1467-985X.2012.01074.x/abstract) and ["significantly worse in nearly all measures of development"](https://link.springer.com/article/10.1353/dem.0.0053) when compared to righty children. Why should the effect direction reverse upon adulthood? The simplest answer is that one or both effects are spurious.

### 5 - Overparameterized model

The authors fit a quartic polynomial linear regression with the only justification being its increased expressive power. But with great expressive power comes great responsibility. Try enough models, with enough complexity, and you'll eventually find one that fits. That means, as we increase the complexity of our model, we're increasing our chance of false positives.

I think economists do the best job here. Economists first build a mathematically intuitive, theoretical explanation of phenomena. That mathematical explanation hints at the proper form of the empirical model with which to test the theory. This is a robust scientific method where we first hypothesize, and then test once, as opposed to testing many times and forming a hypothesis from what works. The latter method is prone to overfitting and false positives. The former method is prone to underfitting. In my opinion a good scientist should heir on the side of underfitting, although Erika Salomon makes a great case that it's overfitting is worth the cost of false positives, because we [also get more true positives](http://www.erikasalomon.com/2015/06/p-hacking-true-effects/).

### Conclusion

I just wanted to point out some red statistical flags. I don't mean to slander the authors of the original study, and I've not said anything to _disprove_ their statements, but only to emphasize that we should be careful. If you're a neuroscientists or someone with in-depth knowledge of the field, I'd be grateful for your setting me straight on anything above. Cheers guys :)
