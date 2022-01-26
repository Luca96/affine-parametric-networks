import gc
import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, SpectralEmbedding, LocallyLinearEmbedding

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


def integral_area(x: list, y: list, degree=None, scale_x=False):
    """First fits the given coordinates (`x`, `y`) with np.Polynomial, then integrates 
       it to compute the area"""
    assert len(x) == len(y)
    
    if scale_x:
        x = x / np.max(x)
    
    domain = (np.min(x), np.max(x))
    degree = len(y) - 1 if degree is None else int(degree)
    
    # fit curve to given (x, y) points
    poly = np.polynomial.Polynomial.fit(x, y, deg=degree, domain=domain)
    
    # compute area by numerical integration
    area, err = scipy.integrate.quad(poly, a=domain[0], b=domain[1])
    return area, err, poly
    

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


def plot_history(history, cols: int, rows: int, title: str, figsize=(30, 20), **kwargs):
    fig, axes = plt.subplots(cols, rows, figsize=figsize)
    fig.suptitle(title)
    
    df = pd.DataFrame(history.history)

    val_columns = list(filter(lambda x: 'val_' in x, df.columns))
    plt_columns = list(map(lambda x: x[x.find('_') + 1:], val_columns))
    
    for i, axis in enumerate(axes.flat):
        cols = [plt_columns[i], val_columns[i]]
        
        axis.plot(df[cols])
        axis.set_title(cols[0])
        axis.legend(cols, loc='best')
    
        if i + 1 >= len(val_columns):
            break


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


def get_checkpoint(path: str, monitor='val_auc'):
    path = os.path.join('weights', path, 'weights-{epoch:02d}-' + f'\u007b{monitor}:.3f\u007d')

    return ModelCheckpoint(path,
                           save_weights_only=True, monitor=monitor,
                           mode='max', save_best_only=True)


def get_compiled_model(cls, data_or_num_features, units: list = None, save: str = None, curve='ROC', **kwargs):
    from tensorflow.keras.metrics import AUC, Precision, Recall
    from script.datasets import Hepmass, Dataset

    opt = kwargs.pop('optimizer', {})
    lr = kwargs.pop('lr', 1e-3)
    compile_args = kwargs.pop('compile', {})

    monitor = kwargs.pop('monitor', 'val_auc')
    ams_args = kwargs.pop('ams', {})

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
                           Precision(name='precision'), Recall(name='recall'),
                           SignificanceRatio(name='ams', **ams_args)])

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


def plot_mass_reliance(result: dict, data, size=(12, 9), loc='best'):
    from script.datasets import Hepmass
    
    none_x = np.array(result['none'])
    all_x = np.array(result['all'])
    
    metric = 200 * np.abs(none_x / all_x - 0.5)
    plt.figure(figsize=size)
    
    plt.title('Mass Reliance')
    plt.xlabel('mass')
    plt.ylabel('%')
    
    if isinstance(data, Hepmass):
        mass = data.unique_mass
    else:
        mass = data.unique_signal_mass
    
    ax = plt.plot(mass, metric, label='$m_{r}$: ' + f'{round(np.mean(metric), 2)}%')
    plt.scatter(mass, metric, color=ax[-1].get_color())
    
    plt.legend(loc=loc)
    plt.show()


def plot_model_distribution(model, dataset, bins=20, name='Model', sample_frac=None, size=24):
    from script.datasets import Hepmass, Dataset

    if isinstance(dataset, Hepmass):
        fig, axes = plt.subplots(ncols=3, nrows=2)
        
        masses = dataset.unique_mass + [None]
        ds_name = 'HEPMASS'
    
        fig.set_figheight(size // len(masses))
        fig.set_figwidth(size)
    else:
        ds_name = 'Dataset'
    
    plt.suptitle(f'[{ds_name}] {name}\'s output distribution', verticalalignment='center')
    
    for i, mass in enumerate(dataset.unique_mass):
        ax = axes[i]
        x, y = dataset.get_by_mass(mass, sample=sample_frac)
        
        out = model.predict(x, batch_size=128, verbose=0)
        out = np.asarray(out)
        
        bkg = out[y == 0.0]
        sig = out[y == 1.0]
        
        ax.set_title(f'{int(round(mass))} GeV')
        ax.set_xlabel('Probability')
        ax.set_ylabel('Num. Events')
        
        ax.hist(sig, bins=bins, alpha=0.55, label='sig', color='blue', edgecolor='blue')
        ax.hist(bkg, bins=bins, alpha=0.7, label='bkg', color='red', histtype='step', 
                hatch='//', linewidth=2, edgecolor='red')
        ax.legend(loc='best')

    fig.tight_layout()


def project_manifold(model, x, y, amount=100_000, size=(12, 10), name='pNN', palette=None, mass=1.0, 
                     scaler=None, projection=TSNE, **kwargs):
    free_mem()
    
    # predict
    z = model.predict(x, batch_size=1024, verbose=1)

    # select only true positives
    mask = np.round(z['y']) == y
    mask = np.squeeze(mask)
    
    # take a random subset
    r = z['r'][mask]
    idx = np.random.choice(np.arange(r.shape[0]), size=int(amount), replace=False)
    
    del z
    free_mem()
    
    # manifold learning method
    if projection != Isomap:
        method = projection(random_state=utils.SEED, n_jobs=-1, **kwargs)
    else:
        method = Isomap(n_jobs=-1, **kwargs)
        
    emb = method.fit_transform(r[idx])
    
    # make dataframe
    m = np.squeeze(x['m'][mask][idx])
    
    if (scaler is not None) and callable(getattr(scaler, 'inverse_transform', None)):
        m = scaler.inverse_transform(np.reshape(m, newshape=(-1, 1)))
        m = np.squeeze(m)
    else:
        m = mass * m
    
    df = pd.DataFrame({'x': emb[:, 0], 'y': emb[:, 1], 'm': np.round(m)})
    
    # plot
    plt.figure(figsize=(12, 10))

    ax = sns.scatterplot(data=df, x='x', y='y', hue='m', legend='full', palette=palette)
    ax.set_title(f'Intermediate Representation Manifold ({name})')
    
    plt.show()
    free_mem()
    return df


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

    def reset_states(self):
        self.s.assign(tf.zeros(shape=(self.bins,), dtype=tf.int32))
        self.b.assign(tf.zeros(shape=(self.bins,), dtype=tf.int32))
