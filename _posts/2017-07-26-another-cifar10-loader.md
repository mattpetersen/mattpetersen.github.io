---
layout: post
title: "Another CIFAR-10 Loader"
date: 2017-07-26
header: true
footer: true
comments: true
tags: python, cifar-10, urllib, tarfile
---

There are lots of CIFAR-10 loaders out there. This one...

* Does not unzip the CIFAR-10 tar file (leaner)
* Loads straight into Numpy (faster)
* Downloads the tar file automatically if missing (easier)

**Install:**

* `pip install cifar10_web`

**Usage:**

* `
train_images, train_labels, test_images, test_labels = cifar10(path=None, onehot=False)
`

**Options:**

* If you leave `path` as `None`, it defaults to `/home/USER/data/cifar10/` or the Windows equivalent, which I believe is `C:\Users\USER\data\cifar10\`.

* If the CIFAR-10 tar file is missing from `path`, it will be downloaded to `path`, and you'll be told that's happening.

* If you set `onehot` to `True` then `train_labels` and `test_labels` will each be a matrix whose rows are onehot labels.

**Speed:**

* Loads in 3.5 seconds. Downloads and loads in 9.5 seconds.

<br>
## Source code

---

<pre class="prettyprint">
"""Load from /home/USER/data/cifar10 or elsewhere; download if missing."""

import tarfile
import os
from urllib.request import urlretrieve

import numpy as np


def cifar10(path=None, onehot=False):
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
    def _onehot(integers):
        """Return matrix whose rows are onehot encodings of integers."""
        n_rows = integers.size
        n_cols = integers.max() + 1
        onehot = np.zeros((n_rows, n_cols), dtype='uint8')
        onehot[np.arange(n_rows), integers] = 1
        return onehot

    url = 'https://www.cs.toronto.edu/~kriz/'
    tar = 'cifar-10-binary.tar.gz'
    files = ['cifar-10-batches-bin/data_batch_1.bin',
             'cifar-10-batches-bin/data_batch_2.bin',
             'cifar-10-batches-bin/data_batch_3.bin',
             'cifar-10-batches-bin/data_batch_4.bin',
             'cifar-10-batches-bin/data_batch_5.bin',
             'cifar-10-batches-bin/test_batch.bin']

    if path is None:
        # Set path to home/USER/data/cifar10 or Windows equivalent
        path = os.path.join(os.path.expanduser('~'), 'data', 'cifar10')

    # Create path if it doesn't exist
    os.makedirs(path, exist_ok=True)

    # Download tarfile if missing
    if tar not in os.listdir(path):
        urlretrieve(''.join((url, tar)), os.path.join(path, tar))
        print("Downloaded %s to %s" % (tar, path))

    # Load data from tarfile
    with tarfile.open(os.path.join(path, tar)) as tar_object:
        # Each file contains 10,000 color images and 10,000 labels
        fsize = 10000 * (32 * 32 * 3) + 10000

        # There are 6 files (5 train and 1 test)
        buffr = np.zeros(fsize * 6, dtype='uint8')

        # Get members of tar corresponding to data files
        # -- The tar contains README's and other extraneous stuff
        members = [file for file in tar_object if file.name in files]

        # Sort those members by name
        # -- Ensures we load train data in the proper order
        # -- Ensures that test data is the last file in the list
        members.sort(key=lambda member: member.name)

        # Extract data from members
        for i, member in enumerate(members):
            # Get member as a file object
            f = tar_object.extractfile(member)
            # Read bytes from that file object into buffr
            buffr[i * fsize:(i + 1) * fsize] = np.frombuffer(f.read(), 'B')

    # Parse data from buffer
    # -- Examples are in chunks of 3,073 bytes
    # -- First byte is label
    # -- Next 32 * 32 * 3 = 3,072 bytes are its corresponding image
    labels = buffr[::3073]
    # Pixels are everything remaining after we delete the labels
    pixels = np.delete(buffr, np.arange(0, buffr.size, 3073))
    images = pixels.reshape(-1, 3072).astype('float32') / 255

    train_images, test_images = images[:50000], images[50000:]
    train_labels, test_labels = labels[:50000], labels[50000:]

    if onehot:
        train_labels = _onehot(train_labels)
        test_labels = _onehot(test_labels)

    return train_images, train_labels, test_images, test_labels
</pre>

---


<br>