import math
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from script import utils
from script import evaluation as eval_utils

from script.datasets import Hepmass
from script.models import PNN

from typing import Union, List


def metric_with_error(model: PNN, dataset: Hepmass, metric: str, index: int, figsize=(12, 9), num_folds=10, 
                      verbose=0, silent=False, style='bar', return_average=True, show=True, **kwargs):
    """Computes given metric on disjoint folds of the given data, quantifying how much uncertain the predictions are"""
    plt.figure(figsize=figsize)

    style = style.lower()
    assert style in ['bar', 'dot', 'fill']
    
    metric_name = metric
    mass = dataset.unique_mass
    metric = {fold: [] for fold in range(num_folds)}
    
    for m in mass:
        x, y = dataset.get_by_mass(m, **kwargs)
        
        folds = eval_utils.split_data(data=(x, y), num_folds=num_folds)
        
        for i, fold in enumerate(folds):
            score = model.evaluate(x=fold[0], y=fold[1], batch_size=128, verbose=verbose)

            metric_score = round(score[index], 4)
            metric[i].append(eval_utils.check_underflow(metric_score, score[1]))
        
        if not silent:
            print(f'Mass: {math.ceil(m)} done.')
    
    # compute average metric (over folds)
    avg_metric = []
    
    for i, _ in enumerate(mass):
        score = 0.0
        
        for fold in range(num_folds):
            score += metric[fold][i]
        
        avg_metric.append(round(score / num_folds, 4))
    
    plt.title(f'{metric_name} vs Mass')
    plt.ylabel(metric_name)
    plt.xlabel('Mass')
    
    label = f'avg: {round(np.mean(avg_metric), 2)}'

    if style == 'dot': 
        plt.plot(mass, avg_metric, marker='o', label=label)
        
        for i in range(num_folds):
            plt.scatter(mass, metric[i], s=30, color='r')
    else:
        values = np.array(list(metric.values()))
        avg_auc = np.array(avg_metric)

        if style == 'bar':
            min_err = avg_auc - np.min(values, axis=0)
            max_err = np.max(values, axis=0) - avg_auc

            plt.errorbar(mass, avg_auc, yerr=np.stack([min_err, max_err]), fmt='ob', 
                         capsize=5.0, elinewidth=1, capthick=1)
            plt.plot(mass, avg_auc, marker='o', label=label)
        else:
            min_err = np.min(values, axis=0)
            max_err = np.max(values, axis=0)

            plt.fill_between(mass, min_err, max_err, color='gray', alpha=0.2)

            plt.plot(mass, avg_auc, marker='o', label=label)
    
    if show:
        plt.legend(loc='best')
        plt.show()

    if return_average:
        return np.mean(list(metric.values()), axis=0)
    
    return metric


def auc_with_error(model: PNN, dataset: Hepmass, index=2, figsize=(12, 9), num_folds=10, verbose=0, 
                   silent=False, style='bar', return_average=True, show=True, **kwargs):
    """Computes AUC on disjoint folds of the given data, quantifying how much uncertain the predictions are"""

    return metric_with_error(model, dataset, metric='AUC', index=index, figsize=figsize, num_folds=num_folds, show=show,
                             verbose=verbose, silent=silent, style=style, return_average=return_average, **kwargs)


def auc_vs_no_mass(model: PNN, dataset: Hepmass, auc_index=2, fake_mass=[], figsize=(26, 20), 
                   sample_frac=None, verbose=0, silent=False):
    """Computes AUC by faking the true mass"""
    plt.figure(figsize=figsize)
    
    # scale mass
    if dataset.m_scaler is not None:
        scaled_fake_mass = dataset.m_scaler.transform(np.reshape(fake_mass, newshape=(-1, 1)))
        scaled_fake_mass = np.squeeze(scaled_fake_mass)
    else:
        scaled_fake_mass = fake_mass
    
    mass = dataset.unique_mass
    auc = {m: [] for m in scaled_fake_mass}

    for m in mass:
        if not silent:
            print(f'Mass {math.ceil(m)}')

        x, y = dataset.get_by_mass(m, sample=sample_frac)
        
        for fake_m in scaled_fake_mass:
            x['m'] = np.ones_like(x['m']) * fake_m
        
            score = model.evaluate(x=x, y=y, batch_size=128, verbose=verbose)
            
            auc_score = round(score[auc_index], 4)
            auc[fake_m].append(eval_utils.check_underflow(auc_score, score[1]))
    
    plt.title('AUC vs Mass')
    
    for i, fake_m in enumerate(fake_mass):
        k = scaled_fake_mass[i]
        
        plt.plot(mass, auc[k], marker='o', label=f'm-{round(fake_m, 2)}')
        # plt.scatter(mass, auc[k], s=50)
        
    plt.legend(loc='best')
    plt.ylabel('AUC')
    plt.xlabel('Mass')
    
    plt.show()
    return auc


def auc_vs_mass_no_features(model: PNN, dataset: Hepmass, auc_index=2, figsize=(26, 20), 
                            sample_frac=None, features={}, verbose=1, silent=False):
    """Computes AUC by dropping one or more features"""
    plt.figure(figsize=figsize)

    mass = dataset.unique_mass
    auc = {k: [] for k in features.keys()}
    
    for label, indexes in features.items():
        if not silent:
            print(f'Features: {label}, {indexes}')
        
        for m in mass:
            x, y = dataset.get_by_mass(m, sample=sample_frac)
        
            # mask features
            for i in indexes:
                zero_feature = np.zeros_like(x['x'][:, i])
                x['x'][:, i] = zero_feature
            
            if not silent:
                print(f'Mass: {math.ceil(m)} done.')
    
            score = model.evaluate(x=x, y=y, batch_size=128, verbose=verbose)
            
            auc_score = round(score[auc_index], 4)
            auc[label].append(eval_utils.check_underflow(auc_score, score[1]))
            
    plt.title(f'AUC vs Mass')
    
    for label in features.keys():
        plt.plot(mass, auc[label], marker='o')
        # plt.scatter(mass, auc[label], s=50, label=label)
    
    plt.ylabel('AUC')
    plt.xlabel('Mass')
    
    plt.legend()
    plt.show()
    return auc


def plot_significance(model, dataset: Hepmass, bins=20, name='Model', sample_frac=None, 
                      size=14, ams_eq=2):
    def safe_div(a, b):
        if b == 0.0:
            return 0.0
        
        return a / b

    fig, axes = plt.subplots(ncols=3, nrows=2)
    axes = np.reshape(axes, newshape=[-1])
    
    fig.set_figwidth(size)
    fig.set_figheight(size // 2)
    
    plt.suptitle(f'[HEPMASS] {name}\'s Output Distribution + Significance', 
                 y=1.02, verticalalignment='top')
    
    for i, mass in enumerate(dataset.unique_mass + [None]):
        ax = axes[i]
        
        if mass is None:
            x, y = dataset.get(sample=sample_frac)
            title = 'Total'
        else:
            x, y = dataset.get_by_mass(mass, sample=sample_frac)
            title = f'{int(round(mass))} GeV'
            
        out = model.predict(x, batch_size=128, verbose=0)
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
            
            cut_mask = (out > lo) & (out <= up)
            
            # select signals and bkg (as true positives of both classes)
            s = out[sig_mask & cut_mask].shape[0]
            b = out[bkg_mask & cut_mask].shape[0]
            
            # compute approximate median significance (AMS)
            if ams_eq == 1:
                val = np.sqrt(2 * ((s + b) * np.log(1 + safe_div(s, b)) - s))
            elif ams_eq == 2:
                val = safe_div(s, np.sqrt(s + b))
            else:
                val = safe_div(s, np.sqrt(b))
            
            ams.append(val)
        
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


def plot_predicted_mass(inverse: tf.keras.Model, dataset: Hepmass, bins=10, size=14, **kwargs):
    # build data-frame, first
    df = pd.DataFrame({'True mass': [], 'Pred. mass': [], 'label': []})
    
    for i, mass in enumerate(dataset.unique_mass):
        x, y = dataset.get_by_mass(mass, **kwargs)
        
        prob = inverse.predict([x['x'], y], batch_size=128, verbose=0)
        pred = tf.reduce_sum(prob * dataset.unique_mass, axis=-1)
        
        a = dataset.m_scaler.inverse_transform(x['m'])
        a = np.round(np.reshape(a, newshape=[-1]))
        b = np.reshape(pred, newshape=[-1]) 
        
        df = pd.concat([df, pd.DataFrame({'True mass': a, 'Pred. mass': b,
                                          'label': np.reshape(y, newshape=[-1])})])
    
    df.replace(to_replace={0.0: 'bkg', 1.0: 'sig'}, inplace=True)
    
    # then plot
    fig, axes = plt.subplots(ncols=3, nrows=2)
    axes = np.reshape(axes, newshape=[-1])
    
    fig.set_figwidth(size)
    fig.set_figheight(size // 2)
    
    plt.suptitle(f'[HEPMASS] Predicted Mass Distribution', y=1.02, verticalalignment='top')
    
    for i, mass in enumerate(dataset.unique_mass + [None]):
        ax = axes[i]
        
        if mass is None:
            title = 'Total'
            dd = df 
        else:
            dd = df.loc[df['True mass'] == round(mass)]
            title = f'{int(round(mass))} GeV'
        
        # signal histogram
        sns.histplot(data=dd[dd.label == 'sig'], x='Pred. mass', label='sig', bins=bins,
                     stat='probability', color='#1616F28C', ax=ax)
        
        # bkg histogram
        sns.histplot(data=dd[dd.label == 'bkg'], x='Pred. mass', label='bkg', bins=bins,
                     stat='probability', color='#CD1313B2', ax=ax)
        
        # hatch the bkg's bars
        for bar in ax.containers[-1]:
            bar.set_hatch('//')

        ax.set_ylabel('Frac. events')
        ax.set_title(title)

        sig_bars, bkg_bars = ax.containers
    
        # stack bars
        for i, bar in enumerate(bkg_bars):
            sbar = sig_bars[i]
            sbar.zorder = 100
            
            bar.set_width(sbar.get_width())
            bar.set_xy(sbar.get_xy())

        ax.legend(loc='best')
    
    fig.tight_layout()
    plt.show()
