from parse_mnist import parse_mnist
from softmax_loss import softmax_loss
import numpy as np

def test_parse_mnist():
    """
    This function tests the parse_mnist function by loading the MNIST 
    dataset and checking that it is correctly formatted.

    The MNIST dataset is a collection of 60,000 images of handwritten 
    digits. Each image is 28x28 pixels, and the dataset contains 10 classes
    (0-9). The images are stored in a gzipped file, and the labels are stored 
    in a separate gzipped file.

    The parse_mnist function is supposed to load the dataset and return two 
    numpy arrays: X, which contains the images, and y, which contains the labels.
    X should be a float32 array with shape (60000,784), and y should be a uint8
    array with shape (60000,).

    The first test checks that the data is correctly normalized. The norm of the
    first 10 images should be 27.892084, and the norm of the first 1000 images
    should be 293.0717. This is checked using np.testing.assert_allclose, which
    checks that two arrays are equal within a certain tolerance.

    The second test checks that the labels are correct. The first 10 labels should
    be [5, 0, 4, 1, 9, 2, 1, 3, 1, 4]. This is checked using np.testing.assert_equal.
    """
    X,y = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                      "data/t10k-labels-idx1-ubyte.gz")

    # Check that the data is correctly normalized
    assert X.dtype == np.float32
    assert y.dtype == np.uint8
    assert X.shape == (60000,784)
    assert y.shape == (60000,)
    np.testing.assert_allclose(np.linalg.norm(X[:10]), 27.892084)
    np.testing.assert_allclose(np.linalg.norm(X[:1000]), 293.0717,
        err_msg="""If you failed this test but not the previous one,
        you are probably normalizing incorrectly. You should normalize
        w.r.t. the whole dataset, _not_ individual images.""", rtol=1e-6)

    # Check that the labels are correct
    np.testing.assert_equal(y[:10], [5, 0, 4, 1, 9, 2, 1, 3, 1, 4])

def test_softmax_loss():
    X,y = parse_mnist("data/train-images-idx3-ubyte.gz",
                      "data/train-labels-idx1-ubyte.gz")
    np.random.seed(0)

    Z = np.zeros((y.shape[0], 10))
    np.testing.assert_allclose(softmax_loss(Z,y), 2.3025850)
    Z = np.random.randn(y.shape[0], 10)
    np.testing.assert_allclose(softmax_loss(Z,y), 2.7291998)
