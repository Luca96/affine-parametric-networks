import gc
import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap

from tensorflow.keras.callbacks import ModelCheckpoint


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


def get_random_generator(seed=SEED) -> np.random.Generator:
    """Returns a numpy's random generator instance"""
    if seed is not None:
        seed = int(seed)
        assert 0 <= seed < 2 ** 32

    return np.random.default_rng(np.random.MT19937(seed=seed))


def makedir(*args: str) -> str:
    """Creates a directory"""
    path = os.path.join(*args)
    os.makedirs(path, exist_ok=True)
    return path


def tf_global_norm(tensors: list, **kwargs):
    norms = [tf.norm(x, **kwargs) for x in tensors]
    return tf.sqrt(tf.reduce_sum([norm * norm for norm in norms]))


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
    return training_set.prefetch(2), validation_set.prefetch(2)


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


def plot_history(h, keys: list, rows=2, cols=2, size=8):
    """Plots the training history of a keras model"""
    fig, axes = plt.subplots(rows, cols, figsize=(cols * size, rows * size))
    axes = np.reshape(axes, newshape=[-1])

    for ax, k in zip(axes, keys):
        ax.plot(h.epoch, h.history[k], marker='o', markersize=10, label='train')

        if f'val_{k}' in h.history:
            ax.plot(h.epoch, h.history[f'val_{k}'], marker='o', markersize=10, label='valid')

        ax.set_xlabel('# Epoch', fontsize=20)
        ax.set_ylabel(k.upper(), rotation="vertical", fontsize=20)

        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)

        ax.grid(alpha=0.5, linestyle='dashed')
        ax.legend(loc='best')

    fig.tight_layout()
    plt.show()


def compare_plot(mass, size=(12, 10), title='Comparison', x_label='x', y_label='y', legend='best', 
                 path='plot', save=None, **kwargs):
    plt.figure(figsize=(12, 10))

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    for k, v in kwargs.items():
        plt.plot(mass, v, marker='o', label=f'{k}: {round(np.mean(v).item(), 2)}')
    
    plt.legend(loc=legend)

    if isinstance(save, str):
        path = makedir(path)
        plt.savefig(os.path.join(path, f'{save}.png'), bbox_inches='tight')

    plt.show()


def plot_model(model: tf.keras.Model, show_shapes=False, layer_names=True, rankdir='TB', dpi=96, 
               expand_nested=False, **kwargs):
    return tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=show_shapes, 
                                     show_layer_names=layer_names, rankdir=rankdir, 
                                     expand_nested=expand_nested, dpi=dpi, **kwargs)


def dataset_from_sequence(sequence: tf.keras.utils.Sequence, sample_weights=False, prefetch=2):
    def gen():    
        for i in range(len(sequence)):
            yield sequence[i]
    
    if sample_weights:
        # {features, mass}, label, sample-weights
        out_types = ({'x': tf.float32, 'm': tf.float32}, tf.float32, tf.float32)
    else:
        # {features, mass}, label
        out_types = ({'x': tf.float32, 'm': tf.float32}, tf.float32)

    tf_data = tf.data.Dataset.from_generator(gen, output_types=out_types)
    
    return tf_data.prefetch(prefetch)


def load_from_checkpoint(model: tf.keras.Model, path: str, base_dir='weights', mode='max'):
    """Load the weights of a pre-built model"""
    path = os.path.join(base_dir, path)
    
    # list all files in directory
    files = os.listdir(path)

    # split into (path, ext) tuples
    files = [os.path.splitext(os.path.join(path, fname)) for fname in files]

    # keep only weights files
    files = filter(lambda x: 'data-' in x[1], files)  
    
    # from tuples get only path; remove ext
    files = map(lambda x: x[0], files)  
    
    # zip files with metric value
    files_and_metric = map(lambda x: (x, x.split('-')[-1]), files)
    
    # sort by metric value
    files = sorted(files_and_metric, key=lambda x: x[-1], reverse=mode.lower() == 'min')
    files = map(lambda x: x[0], files)
    files = list(files)
        
    # load the best weights
    print(f'Loaded from "{files[-1]}"')
    model.load_weights(files[-1])


def delete_checkpoints(path: str, base='weights', mode='max'):
    """Keeps only the best checkpoint while deleting the others"""
    path = os.path.join(base, path)

    # list all files in directory
    files = os.listdir(path)

    # split into (path, ext) tuples
    files = [os.path.splitext(os.path.join(path, fname)) for fname in files]

    # keep only weights files
    files = filter(lambda x: 'data-' in x[1], files)

    # from tuples get only path; remove ext
    files = map(lambda x: x[0], files)

    # zip files with metric value
    files_and_metric = map(lambda x: (x, float(x.split('-')[-1])), files)

    # sort by metric value
    files = sorted(files_and_metric, key=lambda x: x[-1], reverse=mode.lower() == 'min')
    files = map(lambda x: x[0], files)
    files = list(files)

    # load the best weights
    print(f'Keep "{files[-1]}"')

    for f in files[:-1]:
        os.remove(f + '.index')
        os.remove(f + '.data-00000-of-00001')
        print(f'Deleted {f}.')


def get_checkpoint(path: str, monitor='val_auc'):
    path = os.path.join('weights', path, 'weights-{epoch:02d}-' + f'\u007b{monitor}:.3f\u007d')

    return ModelCheckpoint(path,
                           save_weights_only=True, monitor=monitor,
                           mode='max', save_best_only=True)


def get_compiled_model(cls, data_or_num_features, units: list = None, save: str = None, curve='ROC', **kwargs):
    from tensorflow.keras.metrics import AUC, Precision, Recall
    from script.datasets import Hepmass

    opt = kwargs.pop('optimizer', {})
    lr = kwargs.pop('lr', 1e-3)
    compile_args = kwargs.pop('compile', {})
    monitor = kwargs.pop('monitor', 'val_auc')

    units = [300, 150, 100, 50] if units is None else units
    
    if isinstance(data_or_num_features, (int, float)):
        features_shape = (int(data_or_num_features),)
    else:    
        # use number of feature column in `data`
        features_shape = (len(data_or_num_features.columns['feature']),)

    model = cls(input_shapes=dict(m=(1,), x=features_shape), 
                units=units, **kwargs)

    model.compile(lr=lr, **opt, **compile_args,
                  metrics=['binary_accuracy', AUC(name='auc', curve=str(curve).upper()), 
                           Precision(name='precision'), Recall(name='recall')])

    if isinstance(save, str):
        model.save_path = save
        return model, get_checkpoint(path=save, monitor=monitor)
    
    return model


def get_compiled_non_parametric(data, units=None, save=None, **kwargs):
    from script.models import NN
    return get_compiled_model(cls=NN, data_or_num_features=data, units=units, save=save, **kwargs)


def get_compiled_pnn(data, units=None, save=None, **kwargs):
    from script.models import PNN
    return get_compiled_model(cls=PNN, data_or_num_features=data, units=units, save=save, **kwargs)


def get_compiled_affine(data, units=None, save=None, **kwargs):
    from script.models import AffinePNN
    return get_compiled_model(cls=AffinePNN, data_or_num_features=data, units=units, save=save, **kwargs)


def get_plot_axes(rows: int, cols: int, size=(12, 10), **kwargs):
    rows = int(rows)
    cols = int(cols)

    assert rows >= 1
    assert cols >= 1

    fig, axes = plt.subplots(nrows=rows, ncols=cols, **kwargs)

    fig.set_figwidth(size[0] * cols)
    fig.set_figheight(size[1] * rows)

    return axes


def project_manifold(model, x, y, amount=100_000, size=(12, 10), name='pNN', palette=None, 
                     projection=TSNE, **kwargs):
    free_mem()
    
    # predict
    z = model.predict(x, batch_size=1024, verbose=1)
    
    # select only true positives
    mask = np.round(z['y']).reshape((-1, 1)) == np.reshape(y, (-1, 1))
    mask = np.squeeze(mask)

    # take a random subset
    r = z['r'][mask]

    if isinstance(amount, int):
        idx = np.random.choice(np.arange(r.shape[0]), size=min(len(r), int(amount)), replace=False)
    else:
        idx = np.arange(r.shape[0])
    
    del z
    free_mem()
    
    # manifold learning method
    n_jobs = kwargs.pop('n_jobs', -1)

    if projection != Isomap:
        method = projection(random_state=kwargs.pop('random_state', SEED), n_jobs=n_jobs, **kwargs)
    else:
        method = Isomap(n_jobs=n_jobs, **kwargs)
        
    emb = method.fit_transform(r[idx])
    
    # make dataframe
    m = np.squeeze(x['m'][mask][idx])
    m = np.round(m).astype(np.int32)
    
    df = pd.DataFrame({'x': emb[:, 0], 'y': emb[:, 1], 'Mass': m, 
                       'Label': np.reshape(y[mask][idx], newshape=m.shape)})
    
    # plot
    ax0, ax1 = get_plot_axes(rows=1, cols=2, size=(12, 10))

    ax0 = sns.scatterplot(data=df, x='x', y='y', hue='Mass', legend='full', palette=palette, ax=ax0)
    ax0.set_title(f'Intermediate Representation Manifold ({name})')

    ax1 = sns.scatterplot(data=df, x='x', y='y', hue='Label', legend='full', ax=ax1)
    
    plt.tight_layout()
    plt.show()
    
    free_mem()


class SignificanceRatio(tf.keras.metrics.Metric):
    def __init__(self, bins=50, **kwargs):
        super().__init__(**kwargs)

        self.bins = int(bins)
        self.value_range = (0.0, 1.0)

        # init signal and background accumulators
        self.s = tf.Variable(initial_value=tf.zeros(shape=(self.bins,), dtype=tf.int32), trainable=False)
        self.b = tf.Variable(initial_value=tf.zeros(shape=(self.bins,), dtype=tf.int32), trainable=False)

    # @tf.function
    def update_state(self, true, pred, *args, **kwargs):
        sig_mask = tf.reshape(true, shape=[-1]) == 1.0
        bkg_mask = tf.logical_not(sig_mask)

        y = tf.squeeze(pred)
        y_sig = tf.boolean_mask(y, mask=sig_mask)
        y_bkg = tf.boolean_mask(y, mask=bkg_mask)

        s_hist = tf.histogram_fixed_width(y_sig, self.value_range, nbins=self.bins)
        b_hist = tf.histogram_fixed_width(y_bkg, self.value_range, nbins=self.bins)

        self.s.assign_add(s_hist)
        self.b.assign_add(b_hist)

    def result(self):
        ams = []

        for i in range(self.bins):
            s_i = tf.cast(tf.reduce_sum(self.s[i:]), dtype=tf.float32)
            b_i = tf.cast(tf.reduce_sum(self.b[i:]), dtype=tf.float32)

            ams.append(s_i / tf.math.sqrt(s_i + b_i))

        s = tf.cast(tf.reduce_sum(self.s), dtype=tf.float32)
        max_ams = s / tf.math.sqrt(s)

        # return ratio
        return tf.reduce_max(ams) / max_ams

    def reset_state(self):
        self.s.assign(tf.zeros(shape=(self.bins,), dtype=tf.int32))
        self.b.assign(tf.zeros(shape=(self.bins,), dtype=tf.int32))
