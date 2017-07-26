---
layout: post
title: "Another MNIST Loader"
date: 2017-07-26
header: true
footer: true
comments: true
tags: python, mnist, urllib, numpy, frombuffer
---

There are lots of MNIST loaders out there. This one...

* Loads straight into Numpy (faster)
* Downloads automatically if missing (easier)

**Install:**

* `pip install mnist_web`

**Usage:**

* `
train_images, train_labels, test_images, test_labels = mnist(path=None, onehot=False)
`

**Options:**

* If you leave `path` as `None`, it defaults to `/home/USER/data/mnist/` or the Windows equivalent, which I believe is `C:\Users\USER\data\mnist\`.

* Any of the four MNIST files missing from `path` will be downloaded to `path`, and it will tell you that's happening.

* If you set `onehot` to `True` then `train_labels` and `test_labels` will each be a matrix whose rows are onehot labels.

**Speed:**

* Loads in 0.5 seconds. Downloads and loads in 2.5 seconds.

<br>
## Source code

---

<pre class="prettyprint">
"""Load from /home/USER/data/mnist or elsewhere; download if missing."""

import gzip
import os
from urllib.request import urlretrieve

import numpy as np


def mnist(path=None, onehot=False):
    """Return train_images, train_labels, test_images, test_labels.

    :type path: str
    :param path: path to folder containing mnist, default value
                 is /home/USER/data/mnist or Windows equivalent

    :type onehot: bool
    :param onehot: return labels as matrices with onehot rows

    :returns: train_images, train_labels, test_images, test_labels

    Note:
        Downloads to path any mnist files missing from path
    """
    url = 'http://yann.lecun.com/exdb/mnist/'
    files = {'train_images': 'train-images-idx3-ubyte.gz',
             'train_labels': 'train-labels-idx1-ubyte.gz',
             'test_images': 't10k-images-idx3-ubyte.gz',
             'test_labels': 't10k-labels-idx1-ubyte.gz'}

    if path is None:
        # Set path to /home/USER/data/mnist or Windows equivalent
        path = os.path.join(os.path.expanduser('~'), 'data', 'mnist')

    # Create path if it doesn't exist
    os.makedirs(path, exist_ok=True)

    # Download missing files
    for file in files.values():
        if file not in os.listdir(path):
            urlretrieve(''.join((url, file)), os.path.join(path, file))
            print("Downloaded %s to %s" % (file, path))

    def _images(filepath):
        """Return images loaded from path."""
        with gzip.open(filepath) as f:
            # First 16 bytes are magic_number, n_imgs, n_rows, n_cols
            pixels = np.frombuffer(f.read(), 'B', offset=16)
        return pixels.reshape(-1, 784).astype('float32') / 255

    def _labels(filepath, onehot):
        """Return labels loaded from path."""
        def _onehot(integers):
            """Return matrix whose rows are onehot encodings of integers."""
            n_rows = integers.size
            n_cols = integers.max() + 1
            onehot = np.zeros((n_rows, n_cols), dtype='uint8')
            onehot[np.arange(n_rows), integers] = 1
            return onehot

        with gzip.open(filepath) as f:
            # First 8 bytes are magic_number, n_labels
            integer_labels = np.frombuffer(f.read(), 'B', offset=8)

        if onehot:
            return _onehot(integer_labels)
        else:
            return integer_labels

    train_images = _images(os.path.join(path, files['train_images']))
    train_labels = _labels(os.path.join(path, files['train_labels']), onehot)
    test_images = _images(os.path.join(path, files['test_images']))
    test_labels = _labels(os.path.join(path, files['test_labels']), onehot)

    return train_images, train_labels, test_images, test_labels
</pre>

---


<br>