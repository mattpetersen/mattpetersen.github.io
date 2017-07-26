---
layout: post
title: "Feedforward"
date: 2017-06-21
header: true
footer: true
comments: true
tags: neural networks, deep learning, mlp, feedforward
---

In this post we'll feed a dataset forward through a neural network to get an output. For an output, let's predict whether or not it's going to rain on a given day. What factors could predict rain? Some good guesses would be temperature, humidity, cloud cover, and wind speed. Let's gather some data about these factors over a few days. 

![Data raw](images/rain/data_raw_350_325.png)


Consider a network with three layers.

![Neural Network](http://www.texample.net/media/tikz/examples/PNG/neural-network.png)

This network will predict rain. The red node will output zero for sunny, and one for rainy. Each green node represents one input feature. Since there are four green nodes, we'll have four input features. The four input features we use are

* Temperature
* Humidity
* Cloud cover
* Wind speed

These input features are defined by our dataset. Typically our data are stored in a table, where

* Each row is an observation
* Each column is a feature

# Feedforward

__Input layer:__

First, we input four feature values into the green nodes.

* Temperature = 22 Celsius
* Humidity = 80 percent
* Cloud cover = 70 percent
* Wind speed = 10 km/h

This represents one observation (one row) of our data matrix. Each value is that of a feature (a column) of our data matrix.

__Note (normalize the data):__ We won't feed in these exact values. Always scale the feature values to be centered at zero with unit variance. This is called normalizing.

__Example (normalize the data):__ We normalize the temperature column by first subtracting the mean temperature from every temperature, and then dividing each temperature by the sample standard deviation of temperature. We can write this as

<dtex>\text{Temperature}_{\text{normalized}} = \frac{t - \mu_t}{\sigma_t}</dtex>

Here, temperature is a column vector (a feature column of our data matrix), and

* $$t$$ = column vector of temperature values (our data)
* $$\mu_t$$ = column vector where every element is the same, the average temperature
* $$\sigma_t$$ = sample standard deviation of temperature

__Hidden Layer:__

To compute the values of the blue nodes we do a row-vector times matrix multiplication. The row vector holds our four normalized feature values (temperature, humidity, cloud cover, and wind speed). The matrix holds the weights connecting the input layer to the hidden layer. The weight matrix will have

- $$4$$ rows (one for each input feature)
- $$5$$ columns (one for each hidden feature)

Multiplying a $$1 \times 4$$ row vector with a $$4 \times 5$$ weight matrix yields a $$1 \times 5$$ hidden vector. We see that our four normalized input features (temperature, humidity, cloud cover, and wind speed) have been converted into five hidden features.

__Note:__ We don't actually know what these five hidden features represent. _The entire goal of training a neural network is to __learn__ what features the hidden nodes should represent._ Neural networks should be thought of as automatic feature generators. Before neural networks, experts would spend years studying the nuances of datasets and trying to manually engineer features that would allow simple classifiers to perform well. Later, we'll see that the output layer of a neural network is in fact just a simple classifier, but the key is that _it's applied to learned features._

