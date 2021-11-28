import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from script import free_mem, assert_2d_array
from script import utils
from script.datasets.dataset import Dataset

from typing import Union, List


class DataContainer:
    """Wraps a pd.DataFrame, provides test-train split, and balancing of samples"""
    
    def __init__(self, df: pd.DataFrame, columns: dict, balance=None, x_scaler=None,
                 splits=(0.25, 0.2), replace_mass=None, seed=utils.SEED, verbose=False):
        self.df = df
        self.seed = seed
        
        if isinstance(splits, tuple):
            # train-test split
            train_df, test_df = train_test_split(self.df, test_size=splits[-1], shuffle=True, 
                                                 random_state=self.seed)
            # train-validation split
            train_df, valid_df = train_test_split(train_df, test_size=splits[0], shuffle=True, 
                                                  random_state=self.seed)
            
            self.train = DataContainer(train_df, columns, balance=balance, splits=None, replace_mass=replace_mass, seed=seed)
            self.valid = DataContainer(valid_df, columns, splits=None, replace_mass=replace_mass, seed=seed)
            self.test = DataContainer(test_df, columns, splits=None, replace_mass=replace_mass, seed=seed)
            self.all = DataContainer(df, columns, splits=None, replace_mass=replace_mass, seed=seed)

            self.is_split = True
        else:
            self.is_split = False
            
            self.features = self.df[columns['feature']]
            self.labels = self.df[columns['label']]
            self.masses = self.df[columns['mass']]
        
            if isinstance(replace_mass, (float, int)):
                if verbose:
                    print(f'[Container] replacing mass {self.masses.values[0]} with {replace_mass}...')
            
                self.masses = pd.Series(np.ones_like(self.masses) * replace_mass, 
                                        name=self.masses.name, index=self.masses.index)

            if isinstance(balance, (int, float)):
                # sample "balance - size" amont of data, then concat
                size = self.features.shape[0]
                
                if size < balance:
                    amount = int(balance - size)
                    x = self.features.sample(n=amount, replace=amount > len(self), random_state=self.seed)
                    
                    self.features = pd.concat([self.features, x], ignore_index=True)
                    self.labels = pd.concat([self.labels, self.labels.loc[x.index]], ignore_index=True)
                    self.masses = pd.concat([self.masses, self.masses.loc[x.index]], ignore_index=True)
                    
        self.columns = columns
    
    @property
    def shape(self):
        return self.df.shape
    
    @property
    def index(self):
        return self.df.index
    
    def __len__(self):
        return self.df.shape[0]
    
    def __call__(self):
        return self.df
    
    def get(self, split='test') -> dict:
        df = self._get_df(split)
            
        return self._copy_and_reshape(x=df.features.values, 
                                      y=df.labels.values, m=df.masses.values)
    
    def sample(self, split='test', **kwargs) -> dict:
        df = self._get_df(split)
            
        if kwargs.get('n', 0) > df.features.shape[0]:
            kwargs['replace'] = True
            
        x = df.features.sample(**kwargs, random_state=self.seed)
        y = df.labels.loc[x.index]
        m = df.masses.loc[x.index]
        
        return self._copy_and_reshape(x=x.values, y=y.values, m=m.values)
    
    def transform(self, scaler):
        if self.is_split:
            self.train.transform(scaler)
            self.valid.transform(scaler)
            self.test.transform(scaler)
        else:
            self.features = pd.DataFrame(scaler.transform(self.features), index=self.features.index,
                                         columns=self.features.columns)
    
    def describe(self):
        return self.df.describe()
    
    def as_numpy(self, split='train') -> np.ndarray:
        df = self._get_df(split)
        
        x = df.features.values
        y = np.reshape(df.labels.values, newshape=(-1, 1))
        m = np.reshape(df.masses.values, newshape=(-1, 1))
        
        return np.concatenate([x, m, y], axis=-1)
    
    def _get_df(self, split: str):
        assert split is not None
        split = split.lower()
        
        if split == 'test':
            return self.test
        
        if split == 'train':
            return self.train

        if split == 'all':
            return self.all
        
        return self.valid
    
    @staticmethod
    def _copy_and_reshape(x, y, m) -> dict:
        return dict(x=np.copy(x),
                    y=np.reshape(np.copy(y), newshape=(-1, 1)),
                    m=np.reshape(np.copy(m), newshape=(-1, 1)))
    
    def __str__(self):
        return str(self.df)


class FairDataset:
    """Class that wraps Monte-Carlo samples"""
    
    SIGNAL_PATH = os.path.join('data', 'signal.csv')
    BACKGROUND_PATH = os.path.join('data', 'background.csv')
    
    FEATURE_COLUMNS = Dataset.FEATURE_COLUMNS
    MASS_INTERVALS = Dataset.MASS_INTERVALS
    
    def __init__(self, scaler=None, seed=utils.SEED):
        self.seed = seed
        self.df = None
        
        self.sig = dict()
        self.bkg = None
        
        self.columns = []
        self.unique_signal_mass = None
        self.scaled_mass = []
        
        self.scaler = scaler
        
    def load(self, signal: Union[str, pd.DataFrame] = None, bkg: Union[str, pd.DataFrame] = None, 
             select_mass: list = None, balance=True, validation_size=0.25, test_size=0.2, verbose=False,
             name_col='bkg_name', aggregate_names=True):
        '''load'''
        if isinstance(self.df, pd.DataFrame):
            print('[Dataset] already loaded.')
            return
        
        # load or provide signal
        if signal is None or isinstance(signal, str):
            print('[signal] loading...')
            signal = pd.read_csv(signal or self.SIGNAL_PATH, dtype=np.float32, na_filter=False)
        
        elif isinstance(signal, pd.DataFrame):
            signal = signal
        else:
            raise ValueError

        # print('[signal] loading...')
        # signal = pd.read_csv(self.SIGNAL_PATH, dtype=np.float32, na_filter=False)
        
        # load or provide background
        if bkg is None or isinstance(bkg, str):
            print('[background] loading...')
            background = pd.read_csv(bkg or self.BACKGROUND_PATH, na_filter=False)

        elif isinstance(bkg, pd.DataFrame):
            background = bkg
        else:
            raise ValueError

        # print('[background] loading...')
        # background = pd.read_csv(self.BACKGROUND_PATH, dtype=np.float32, na_filter=False)
        
        # keep track of names if available:
        if isinstance(name_col, str) and name_col in background.columns:
            names = background[name_col]

            if aggregate_names:
                def convert_names(x):
                    if 'ST_' in x:
                        return 'ST'
                    
                    if 'diboson_' in x:
                        return 'diboson'
                    
                    return x

                new_names = names.apply(convert_names)

                self.names_df = pd.DataFrame({'name': new_names})
                self.original_names = pd.DataFrame({'name': names})
            else:
                self.names_df = pd.DataFrame({'name': names})
                self.original_names = self.names_df.copy()

            background.drop(columns=[name_col], inplace=True)

        # select masses
        self.all_mass = sorted(signal['mA'].unique())
        self.max_mass = np.max(self.all_mass)
        
        if isinstance(select_mass, (list, tuple, np.ndarray)):
            print('[signal] selecting mass...')
            mask = np.full(shape=[len(signal)], fill_value=False)
            
            for mass in select_mass:
                mask |= signal['mA'] == mass
            
            signal = signal[mask]
            utils.free_mem()
        
        self.df = pd.concat([signal, background], ignore_index=True)
        # self.df.reset_index(inplace=True)
        
        # select from `df` (all data), signal and background
        self.signal_df = self.df[self.df['type'] == 1.0]
        self.backgr_df = self.df[self.df['type'] == 0.0]
        
        # mass
        self.unique_signal_mass = sorted(signal['mA'].unique())
        self.unique_mass = self.unique_signal_mass
        
        # columns
        self.columns = dict(feature=self.FEATURE_COLUMNS,
                            label=signal.columns[-1], 
                            mass=signal.columns[0])

        # "containerize"
        self.bkg = DataContainer(df=self.backgr_df, columns=self.columns, balance=False, 
                                 splits=(validation_size, test_size), seed=self.seed, verbose=verbose)
        
        # slice `signal` by mass
        mass_slices = []
        
        for mass in self.unique_signal_mass:
            mass_df = self.signal_df[self.signal_df[self.columns['mass']] == mass]
            mass_slices.append(mass_df)
        
        if balance is True:
            balance = 1.0
        
        if isinstance(balance, float):
            assert 0.0 < balance <= 1.0
            oversampling = balance
            
            # compute balance size
            print('[Dataset] balancing signal samples per-mass...')
            balance = mass_slices[0].shape[0] * oversampling
            
            for df in mass_slices:
                balance = max(balance, df.shape[0] * oversampling)
        else:
            balance = None
        
        # k = 0
        for i, mass in enumerate(self.unique_signal_mass):
            # for j in range(k, len(self.all_mass)):
            #     if mass == self.all_mass[j]:
            #         break
            #     else:
            #         k += 1
                
            # new_mass = k + 1
            new_mass = float(mass / self.max_mass)
            self.scaled_mass.append(new_mass)
            
            self.sig[mass] = DataContainer(df=mass_slices[i], columns=self.columns, balance=balance, 
                                           splits=(validation_size, test_size), 
                                           replace_mass=new_mass, seed=self.seed, verbose=verbose)

        # feature scaling
        if self.scaler is not None:
            print('[Dataset] fitting and scaling features...')
            self.fit_and_scale()
        
        print('[Dataset] loaded.')
        self.train_features = self.bkg.train.features  # backward compatibility
    
    def scale_mass(self, mass: list, shape=None):
        x = np.array(mass) / self.max_mass
        return np.reshape(x, newshape=shape or [-1])

    def fit_and_scale(self):
        indices = self.bkg.train.index
        
        # 1. gather indices of training set, 
        for container in self.sig.values():
            indices.union(container.train.index)
        
        # 2. then loc
        train_df = self.df.loc[indices, self.columns['feature']]
        
        # 3. fit
        self.scaler.fit(train_df)
        
        # 4. transform each
        self.bkg.transform(scaler=self.scaler)
        
        for container in self.sig.values():
            container.transform(scaler=self.scaler)
    
    def get(self, split='test', sample_mass=True) -> dict:
        bkg = self.bkg.get(split=split)
        data = {}

        for k, v in bkg.items():
            data[k] = [v]

        for sig in self.sig.values():
            for k, v in sig.get(split=split).items():
                data[k].append(v)

        if sample_mass:
             # sample mass for bkg samples
            mass = np.random.choice(self.scaled_mass, size=bkg['m'].shape[0])
            mass = np.reshape(mass, newshape=(-1, 1))
            
            data['m'][0] = mass  # replace first item in the list at key "m"

        data = {k: np.concatenate(v, axis=0) for k, v in data.items()}
        y = data.pop('y')
        
        return data, y


    def get_by_mass(self, mass, sample_mass=False, sample=None, balance=True, bkg=None, 
                    split='test') -> dict:
        assert mass in self.sig
        
        # sample or get signal
        if isinstance(sample, int):
            sig = self.sig[mass].sample(n=sample, split=split)
        
        elif isinstance(sample, float):
            sig = self.sig[mass].sample(frac=sample, split=split)
        else:
            sig = self.sig[mass].get(split=split)
            
        # sample or get background
        if balance:
            bkg = self.bkg.sample(n=sig['x'].shape[0], split=split)
        
        elif isinstance(bkg, int):
            bkg = self.bkg.sample(n=bkg, split=split)
        
        elif isinstance(bkg, float):
            bkg = self.bkg.sample(frac=bkg, split=split)
        else:
            bkg = self.bkg.get(split=split)
        
        if sample_mass:
            # sample mass for bkg samples
            mass = np.random.choice(self.scaled_mass, size=bkg['m'].shape[0])
            mass = np.reshape(mass, newshape=(-1, 1))
            
            bkg['m'] = mass
        
        # merge "sig" and "bkg" into a dict
        data = {k: np.concatenate([v, bkg[k]], axis=0) for k, v in sig.items()}
        y = data.pop('y')
        
        return data, y
    
    # TODO: error bars
    def plot_auc(self, model, legend='best', name='pNN', split='test', batch_size=1024, 
                 size=(12, 10), ax=None, show=True, **kwargs):
        auc = []
        mass = self.unique_signal_mass
        
        if not isinstance(model, list):
            model = [model for _ in range(len(mass))]

        if ax is None:
            ax = plt.gca()

            fig = ax.figure
            fig.set_figwidth(size[0])
            fig.set_figheight(size[1])
            
        for i, m in enumerate(mass):
            x, y = self.get_by_mass(m, split=split, **kwargs)
            
            score = model[i].evaluate(x=x, y=y, batch_size=batch_size, verbose=0)
            auc.append(round(score[2], 4))
        
        ax.set_title(f'AUC vs Mass ({name})')
        ax.set_ylabel('AUC')
        ax.set_xlabel('Mass (GeV)')
    
        ax.plot(mass, auc, marker='o', label=f'avg: {round(np.mean(auc), 2)}')
        ax.legend(loc=legend)
        
        if show:
            plt.show()
        
        return np.array(auc)
        
    def plot_significance(self, model, bins=20, name='pNN', size=4, batch_size=1024, split='test', **kwargs):
        def safe_div(a, b):
            if b == 0.0:
                return 0.0
            
            return a / b

        fig, axes = plt.subplots(ncols=4, nrows=6)
        axes = np.reshape(axes, newshape=[-1])
        
        fig.set_figwidth(int(size * 5))
        fig.set_figheight(int(size * 5))
        
        plt.suptitle(f'[MCS] {name}\'s Output Distribution & Significance', 
                     y=1.02, verticalalignment='top')
        
        for i, mass in enumerate(self.unique_signal_mass + [None]):
            ax = axes[i]
            
            if mass is None:
                x, y = self.get(split=split, **kwargs)
                title = 'Total'
            else:
                x, y = self.get_by_mass(mass, split=split, **kwargs)
                title = f'{int(mass)} GeV'
                
            out = model.predict(x, batch_size=batch_size, verbose=0)
            out = np.asarray(out)
            
            sig_mask = y == 1.0
            bkg_mask = y == 0.0
            
            cuts = np.linspace(0.0, 1.0, num=bins)
            ams = []
            
            bx = ax.twinx()
            
            ax.hist(out[sig_mask], bins=bins, alpha=0.55, label='sig', color='blue', edgecolor='blue')
            ax.hist(out[bkg_mask], bins=bins, alpha=0.7, label='bkg', color='red', histtype='step', 
                    hatch='//', linewidth=2, edgecolor='red')
            
            for i in range(len(cuts) - 1):
                lo, up = cuts[i], cuts[i + 1]
                cut_mask = out >= lo
                
                # select signals and bkg
                s = out[sig_mask & cut_mask].shape[0]
                b = out[bkg_mask & cut_mask].shape[0]
            
                # compute approximate median significance (AMS)
                ams.append(safe_div(s, np.sqrt(s + b)))

            k = np.argmax(ams)
            
            bx.grid(False)
            bx.plot(cuts, [0.0] + ams, color='g', label='Significance')
            
            ax.axvline(x=cuts[k + 1], linestyle='--', linewidth=2, color='g',
                       label=f'{round(cuts[k + 1], 1)}: {round(ams[k], 1)}')
            
            bx.set_ylabel('Significance')
            ax.set_title(title)
            ax.set_xlabel('Probability')
            ax.set_ylabel('Num. Events')
            
            ax.legend(loc='best')
        
        fig.tight_layout()


class DataSequence(tf.keras.utils.Sequence):
    def __init__(self, dataset: FairDataset, batch_size: int, split='train', augment: Union[bool, int, float] = False, 
                 seed=utils.SEED, prob='balance'):
        assert prob in ['uniform', 'balance', 'inverse']
        self.rnd = utils.get_random_generator(seed=seed)
        
        self.bkg = dataset.bkg.as_numpy(split)
        self.sig = np.concatenate([v.as_numpy(split) for v in dataset.sig.values()], axis=0)
        
        self.mass = dataset.scaled_mass
        self.size = dataset.df.shape[0]
        
        self.batch_size = batch_size
        self.half_batch = batch_size // 2

        # data augmentation for signal
        if isinstance(augment, (bool, int, float)):
            self.should_augment = (augment is True) or (augment > 0)

            if isinstance(augment, float):
                self.augment_size = int(self.batch_size * augment)
            else:
                self.augment_size = int(augment)

            # take indices
            start = 0
            self.indices = []

            for v in dataset.sig.values():
                size = len(v)

                self.indices.append(np.arange(start, start + size))
                start += size

            # determine sampling prob for each mass
            if prob == 'uniform':
                self.probs = np.ones(shape=(len(self.mass), )) / len(self.mass)
            else:
                max_size = start
                self.probs = [len(v) / max_size for v in dataset.sig.values()]
                self.probs[-1] += 1.0 - sum(self.probs)  # probs must sum to one

                if prob == 'balance':
                    self.probs.reverse()
        else:
            self.should_augment = False
        
    def __len__(self):
        return int(self.size / self.batch_size)
    
    def __getitem__(self, idx):
        sig = self.sample(self.sig, size=self.half_batch)
        bkg = self.sample(self.bkg, size=self.half_batch)
        
        # sample mass for bkg
        bkg_m = self.rnd.choice(self.mass, size=self.half_batch)
        bkg_m = np.reshape(bkg_m, newshape=(-1, 1))
        
        # data-aug
        if self.should_augment:
            aug_x, aug_m, aug_y = self.sample_aug()

            features = [sig[0], bkg[0], aug_x]
            masses = [sig[1], bkg_m, aug_m]
            labels = [sig[2], bkg[2], aug_y]
        else:
            features = [sig[0], bkg[0]]
            masses = [sig[1], bkg_m]
            labels = [sig[2], bkg[2]]

        # merge features, mass, and labels
        x = np.concatenate(features, axis=0)
        m = np.concatenate(masses, axis=0)
        y = np.concatenate(labels, axis=0)
        
        return dict(x=x, m=m), y
    
    def sample(self, x: np.array, size: int) -> tuple:
        x = x[self.rnd.choice(x.shape[0], size=size, replace=False)]
        return self._unpack(x)
    
    def sample_aug(self):
        size = int(self.augment_size / len(self.mass))
        indices = self.rnd.choice(np.arange(len(self.indices)), p=self.probs, size=len(self.mass))

        aug = dict(x=[], m=[], y=[])

        for i in indices:
            # get random indices for mass[i]
            idx = self.rnd.choice(self.indices[i], size=size)

            aug_x = self.sig[0][idx]  # features
            aug_m = self.rnd.choice(self.mass, size=size)  # sample mass
            aug_y = np.array(sig[1][idx] == aug_m, dtype=np.int32)  # determine aug-labels

            aug['x'].append(aug_x)
            aug['m'].append(aug_m)
            aug['y'].append(aug_y)

        # return (features, mass, labels)
        return np.concatenate(aug['x'], axis=0), np.concatenate(aug['m'], axis=0), \
               np.concatenate(aug['y'], axis=0)

    @staticmethod
    def _unpack(x: np.ndarray) -> tuple:
        # features, mass, labels
        return x[:, :-2], x[:, -2, None], x[:, -1, None]  # "None" to expand last dim
            