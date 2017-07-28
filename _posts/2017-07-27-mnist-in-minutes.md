---
layout: post
title: "Load MNIST in 26 lines"
date: 2017-07-26
header: true
footer: true
comments: true
tags: python, mnist, urllib, gzip, numpy, frombuffer
---

Load MNIST into Python directly from the web. This is slower than loading from a local file (2.5 seconds vs. 0.5 seconds), but it's short and pretty.

<br>
### Just the code
---

<pre class='prettyprint'>
from urllib.request import urlopen
import gzip
import numpy as np

train_images_url = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
train_labels_url = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
test_images_url = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
test_labels_url = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'

# Train labels
with gzip.open(urlopen(train_labels_url)) as f:
    train_labels = np.frombuffer(f.read(), '>B', offset=8)

# Test labels
with gzip.open(urlopen(test_labels_url)) as f:
    test_labels = np.frombuffer(f.read(), '>B', offset=8)

# Train images
with gzip.open(urlopen(train_images_url)) as f:
    train_images = np.frombuffer(f.read(), '>B', offset=16)
    train_images = train_images.astype('float16').reshape(-1, 784) / 255

# Test images
with gzip.open(urlopen(train_images_url)) as f:
    test_images = np.frombuffer(f.read(), '>B', offset=16)
    test_images = test_images.astype('float16').reshape(-1, 784) / 255
</pre>

---

<br>
### Packages

* `urlopen` - Use an online file like it's local
* `gzip` - Open that online file
* `numpy` - Read data from that online file

---

<pre class="prettyprint">
from urllib.request import urlopen
import gzip
import numpy as np
</pre>

---

<br>
### URL's

---

<pre class="prettyprint">
train_images_url = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
train_labels_url = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
test_images_url = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
test_labels_url = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
</pre>

---

<br>
### Labels

Use `gzip.open()` to read the tar file fetched from online by `urlopen()`, and then use `np.frombuffer()` to store the bytes returned by `f.read()`. We use `offset=8` because the first eight bytes of the file are descriptive: the first four are the magic number, which specifies the file format (i.e. whether the first byte is the biggest or smallest power of two), and the next four tell us how many labels are in the file. We don't need either, so we just start reading after eight bytes. The data type `>B` means big-endian bytes, so each label is a chunk of 8 bits, where the first bit is the largest power of two and the last bit is the smallest.

---

<pre class='prettyprint'>
# Train labels
with gzip.open(urlopen(train_labels_url)) as f:
    train_labels = np.frombuffer(f.read(), '>B', offset=8)

# Test labels
with gzip.open(urlopen(test_labels_url)) as f:
    test_labels = np.frombuffer(f.read(), '>B', offset=8)
</pre>

---

<br>
### Images

Same as labels, but now we start reading after 16 bytes of descriptors (magic number, number of images, image width, image height). We cast to 16-bit floats for memory efficiency (neural nets don't need precision). We then reshape the pixels into flattened image vectors of size 28 * 28 = 784, where -1 specifies that the number of images will be inferred. Lastly we scale the pixel values to be within zero and one by dividing by the maximum pixel value of 255.

---

<pre class='prettyprint'>
# Train images
with gzip.open(urlopen(train_images_url)) as f:
    train_images = np.frombuffer(f.read(), '>B', offset=16)
    train_images = train_images.astype('float16').reshape(-1, 784) / 255

# Test images
with gzip.open(urlopen(train_images_url)) as f:
    test_images = np.frombuffer(f.read(), '>B', offset=16)
    test_images = test_images.astype('float16').reshape(-1, 784) / 255
</pre>

---

<br>