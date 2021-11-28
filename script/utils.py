import gc
import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

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


def get_dataset(dataset, split: str, batch_size: int, sample_weights=False, **kwargs):
    from script.datasets import FairDataset, DataSequence
    assert isinstance(dataset, FairDataset)
    
    sequence = DataSequence(dataset, batch_size=batch_size, split=split.lower(), **kwargs)
    
    def gen():    
        for i in range(len(sequence)):
            yield sequence[0]
    
    if sample_weights:
        # {features, mass}, label, sample-weights
        out_types = ({'x': tf.float32, 'm': tf.float32}, tf.float32, tf.float32)
    else:
        # {features, mass}, label
        out_types = ({'x': tf.float32, 'm': tf.float32}, tf.float32)

    tf_data = tf.data.Dataset.from_generator(gen, output_types=out_types)
    
    return tf_data.prefetch(3)


def dataset_from_sequence(sequence: tf.keras.utils.Sequence, sample_weights=False, prefetch=2):
    def gen():    
        for i in range(len(sequence)):
            yield sequence[0]
    
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


def get_compiled_model(cls, data, units: list = None, save: str = None, curve='ROC', **kwargs):
    from tensorflow.keras.metrics import AUC, Precision, Recall
    from script.datasets import Hepmass, Dataset

    opt = kwargs.pop('optimizer', {})
    lr = kwargs.pop('lr', 1e-3)
    compile_args = kwargs.pop('compile', {})

    monitor = kwargs.pop('monitor', 'val_auc')

    units = [300, 150, 100, 50] if units is None else units
    
    if isinstance(data, Hepmass):
        features_shape = (data.features.shape[-1],)
    else:
        features_shape = (data.train_features.shape[-1],)
    
    model = cls(input_shapes=dict(m=(1,), x=features_shape), 
                units=units, **kwargs)

    model.compile(lr=lr, **opt, **compile_args,
                  metrics=['binary_accuracy', AUC(name='auc', curve=str(curve).upper()), 
                           Precision(name='precision'), Recall(name='recall'),
                           # Significance2(name='sig2'), 
                           # Significance3(name='sig3')
                           ])

    if isinstance(save, str):
        return model, get_checkpoint(path=save, monitor=monitor)
    
    return model


def get_compiled_non_parametric(data, units=None, save=None, **kwargs):
    from script.models import NN
    return get_compiled_model(cls=NN, data=data, units=units, save=save, **kwargs)


def get_compiled_pnn(data, units=None, save=None, **kwargs):
    from script.models import PNN
    return get_compiled_model(cls=PNN, data=data, units=units, save=save, **kwargs)


def get_compiled_affine(data, units=None, save=None, **kwargs):
    from script.models import AffinePNN
    return get_compiled_model(cls=AffinePNN, data=data, units=units, save=save, **kwargs)


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


# class Significance(tf.keras.metrics.Metric):
#     def __init__(self, bins=20, eps=1e-7, **kwargs):
#         super().__init__(**kwargs)
        
#         self.cuts = tf.linspace(0.0, 1.0 + eps, num=bins)
#         self.ams = self.add_weight(name='ams', initializer='zeros')
    
#     def update_state(self, true, pred):
#         sig_mask = true == 1.0
#         bkg_mask = true == 0.0
        
#         ams = []
        
#         for i in range(self.cuts.shape[0] - 1):
#             cut_mask = (pred >= self.cuts[i]) & (pred < self.cuts[i + 1])
            
#             # select signals and bkg (as true positives of both classes)
#             s = tf.shape(pred[sig_mask & cut_mask])[0]
#             s = tf.cast(s, dtype=tf.float32)

#             b = tf.shape(pred[bkg_mask & cut_mask])[0]
#             b = tf.cast(b, dtype=tf.float32)
            
#             # compute approximate median significance (AMS)
#             ams.append(s / tf.sqrt(s + b))
        
#         self.ams.assign(tf.reduce_max(ams))
        
#     def result(self):
#         return self.ams.value()
    
#     def reset_states(self):
#         self.ams.assign(0.0)


# class Significance2(tf.keras.metrics.Metric):
#     def __init__(self, bins=20, eps=1e-7, **kwargs):
#         super().__init__(**kwargs)
    
#         self.cuts = tf.linspace(0.0, 1.0 + eps, num=bins)
#         self.init_shape = (tf.shape(self.cuts)[0] - 1,)

#         self.sig_count = self.add_weight(name='s', shape=self.init_shape, initializer='zeros',
#                                          aggregation=tf.VariableAggregation.NONE)
#         self.bkg_count = self.add_weight(name='b', shape=self.init_shape, initializer='zeros',
#                                          aggregation=tf.VariableAggregation.NONE)
    
#     def update_state(self, true: tf.Tensor, pred: tf.Tensor):
#         sig_mask = tf.reshape(true == 1.0, shape=[-1])
#         bkg_mask = tf.logical_not(sig_mask)
        
#         sig = []
#         bkg = []
        
#         for i in range(self.cuts.shape[0] - 1):
#             cut_mask = (pred >= self.cuts[i]) & (pred < self.cuts[i + 1])
#             cut_mask = tf.reshape(cut_mask, shape=[-1])

#             # select signals and bkg (as true positives of both classes)
#             s = tf.shape(pred[sig_mask & cut_mask])[0]
#             b = tf.shape(pred[bkg_mask & cut_mask])[0]
            
#             sig.append(s)
#             bkg.append(b)

#         self.sig_count.assign_add(tf.cast(sig, dtype=tf.float32))
#         self.bkg_count.assign_add(tf.cast(bkg, dtype=tf.float32))
    
#     # @tf.function
#     def result(self):
#         s = self.sig_count.value()
#         b = self.bkg_count.value()

#         # compute approximate median significance (AMS)
#         ams = s / tf.square(s + b) 
#         return tf.reduce_max(ams)
    
#     def reset_states(self):
#         self.sig_count.assign(tf.zeros(shape=self.init_shape))
#         self.bkg_count.assign(tf.zeros(shape=self.init_shape))


# class Significance3(tf.keras.metrics.Metric):
#     def __init__(self, bins=20, eps=1e-7, **kwargs):
#         super().__init__(**kwargs)
        
#         self.cuts = tf.linspace(0.0, 1.0 + eps, num=bins)
#         self.init_shape = (tf.shape(self.cuts)[0] - 1,)

#         self.sig_count = self.add_weight(name='s', shape=self.init_shape, initializer='zeros',
#                                          aggregation=tf.VariableAggregation.NONE)
#         self.bkg_count = self.add_weight(name='b', shape=self.init_shape, initializer='zeros',
#                                          aggregation=tf.VariableAggregation.NONE)
    
#     def update_state(self, true: tf.Tensor, pred: tf.Tensor):
#         sig_mask = tf.reshape(true == 1.0, shape=[-1])
#         bkg_mask = tf.logical_not(sig_mask)
        
#         sig = []
#         bkg = []
        
#         for i in range(self.cuts.shape[0] - 1):
#             cut_mask = (pred >= self.cuts[i]) & (pred < self.cuts[i + 1])
#             cut_mask = tf.reshape(cut_mask, shape=[-1])

#             # select signals and bkg (as true positives of both classes)
#             s = tf.shape(pred[sig_mask & cut_mask])[0]
#             b = tf.shape(pred[bkg_mask & cut_mask])[0]
            
#             sig.append(s)
#             bkg.append(b)

#         self.sig_count.assign_add(tf.cast(sig, dtype=tf.float32))
#         self.bkg_count.assign_add(tf.cast(bkg, dtype=tf.float32))
    
#     # @tf.function
#     def result(self):
#         s = self.sig_count.value()
#         b = self.bkg_count.value()

#         # compute approximate median significance (AMS)
#         return s + b
    
#     def reset_states(self):
#         self.sig_count.assign(tf.zeros(shape=self.init_shape))
#         self.bkg_count.assign(tf.zeros(shape=self.init_shape))
