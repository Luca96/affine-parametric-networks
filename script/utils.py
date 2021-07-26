import gc
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.decomposition import PCA


SEED = None


def set_random_seed(seed: int) -> int:
    """Sets the random seed for TensorFlow, numpy, python's random"""
    global SEED
    
    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        SEED = seed
        print(f'Random seed {SEED} set.')


def tf_global_norm(tensors: list, **kwargs):
    norms = [tf.norm(x, **kwargs) for x in tensors]
    return tf.sqrt(tf.reduce_sum([norm * norm for norm in norms]))


def tf_jensen_shannon_divergence(p, q):
    """Jensen-Shannon Divergence: JS(p || q) = 1/2 KL(p || m) + 1/2 KL(q || m)
        - Source: https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
    """
    m = (p + q) * 0.5
    
    return 0.5 * tf.keras.losses.kld(p, m) + \
           0.5 * tf.keras.losses.kld(q, m)


def tf_jensen_shannon_distance(p, q):
    jsd = tf_jensen_shannon_divergence(p, q)
    return tf.sqrt(tf.math.abs(jsd))


def free_mem():
    return gc.collect()


def dataset_from_tensors(tensors, batch_size: int, split=0.25, seed=SEED):
    total_size = tensors[-1].shape[0]
    val_size = int(total_size * split)
    train_size = total_size - val_size
    
    dataset = tf.data.Dataset.from_tensor_slices(tensors)
    dataset = dataset.shuffle(buffer_size=1024)

    training_set = dataset.take(train_size)
    training_set = training_set.batch(batch_size)
    
    validation_set = dataset.skip(train_size).take(val_size)
    validation_set = validation_set.batch(batch_size)
    
    free_mem()
    return training_set, validation_set


def pca_plot(pca: PCA, title='PCA', **kwargs):
    components_range = np.arange(pca.n_components_)
    explained_variance = pca.explained_variance_ratio_
    
    sns.barplot(x=components_range, y=explained_variance, **kwargs)
    
    plt.xlabel('num_components')
    plt.ylabel('Explained variance ratio')
    plt.title(title)
    plt.show()


def assert_2d_array(x, last_dim=2):
    if x is None:
        return
    
    x = np.asarray(x)
    
    assert len(x.shape) == 2
    assert x.shape[-1] == last_dim
