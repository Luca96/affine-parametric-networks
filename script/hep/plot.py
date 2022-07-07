"""HEPMASS/plot"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, roc_auc_score

from script import utils, cms
from script.datasets import Hepmass
from script.cms.data import sample

from typing import Union


PALETTE = {'signal': 'blue', 'bkg': 'red'}


def get_ams_and_cut(model, dataset, bins=50, weight=True, all_bkg=True, ratio=False, features=None):
    """Computes the significance and its best-cut for each mass"""
    ams = []
    cuts = np.linspace(0.0, 1.0, num=bins)
    
    features, mass, label = _get_columns(dataset, features)
    
    sig = dataset.signal
    bkg = dataset.background
    
    if not all_bkg:
        # should not weight
        weight = False  
        
    for m in dataset.unique_signal_mass:
        # select both signal and background in interval (m-d, m+d)
        s = sig[sig[mass] == m]
        
        if not all_bkg:
            b = bkg[bkg[mass] == m]
        else:
            b = bkg
        
        # prepare data
        x = pd.concat([s[features], b[features]], axis=0).values
        y = pd.concat([s[label], b[label]], axis=0).values.reshape([-1])
        x = dict(x=x, m=np.ones_like(y[:, np.newaxis]) * mass)
    
        # predict data
        out = model.predict(x=x, batch_size=1024, verbose=0)
        out = np.asarray(out)

        y_s = np.squeeze(out[y == 1.0])
        y_b = np.squeeze(out[y == 0.0])
        
        ams_ = []
        w_b = np.ones_like(y_b)
        
        if weight:
            w_b /= len(dataset.unique_signal_mass)
        
        s, _ = np.histogram(y_s, bins=bins, range=(0, 1))
        b, _ = np.histogram(y_b, bins=bins, range=(0, 1), weights=w_b)

        for i in range(s.shape[0]):
            s_i = np.sum(s[i:])
            b_i = np.sum(b[i:])

            ams_.append(s_i / np.sqrt(s_i + b_i))

        if ratio:
            max_ams = len(y_s) / np.sqrt(len(y_s))
            ams_ = np.array(ams_) / max_ams

        ams.append(ams_)

    ams = np.array(ams)
    ams[np.isnan(ams) | np.isinf(ams)] = 0

    return np.max(ams, axis=-1), cuts[np.argmax(ams, axis=-1)]


def get_curve_auc(dataset: Hepmass, models: dict, bins=50, features=None, weight=True, 
                  which='ROC', **kwargs):
    """Plots the AUC of the ROC curve at each signal's mass""" 
    assert which.upper() in ['ROC', 'PR']
    
    features, mass, label = _get_columns(dataset, features)
    
    sig = dataset.signal
    bkg = dataset.background

    if which.upper() == 'ROC':
        curve_fn = cms.plot.roc_auc
    else:
        curve_fn = cms.plot.pr_auc
    
    mass = dataset.unique_signal_mass
    auc = {k: [] for k in models.keys()}

    wb = np.ones((bkg.shape[0],))  # weight for background

    if weight:
        wb /= len(mass)

    b_values = bkg[features].values
    b_labels = bkg[label]
    
    for name, model in models.items():
        for m in mass:
            # select data
            s = sig[sig[mass] == m]
                
            # prepare data
            x = np.concatenate([s[features].values, b_values], axis=0)
            y = pd.concat([s[label], b_labels], axis=0).values.reshape([-1])
            x = dict(x=x, m=np.ones_like(y[:, np.newaxis]) * m)
    
            # predict data
            out, _ = _get_predictions(model, x, y)
            
            ws = np.ones((s.shape[0],))
            w = np.concatenate([ws, wb], axis=0)

            # compute curve AUC
            _, _, m_auc, _, _ = curve_fn(true=y, pred=out, weights=w, cut=0.5, **kwargs)
            auc[name].append(m_auc)
    
    return auc


def significance(model, dataset: Hepmass, mass: int, digits=4, bins=50, size=(12, 10), ax=None,
                 legend='best', palette=PALETTE, path='plot', save=None, show=True, ratio=True,
                 weight=True, features=None, name='Model', all_bkg=True):
    """Plots the output distribution of the model, along it's weighted significance"""
    if ax is None:
        fig = plt.figure(figsize=size)
        ax = fig.gca()
    
    features, mass_col, label = _get_columns(dataset, features)
    
    s = dataset.signal[dataset.signal[mass] == mass]
    b = dataset.background
    num_sig = len(s)
    
    if not all_bkg:
        # should not weight
        weight = False  
        
        # select bkg by mass
        b = b[b[mass_col] == mass]
    
    # prepare data
    x = np.concatenate([s[features].values, b[features].values], axis=0)
    y = pd.concat([s[label], b[label]]).values.reshape([-1])
    x = dict(x=x, m=np.ones_like(y[:, np.newaxis]) * mass) 
    
    # predict
    out, (y_s, y_b) = _get_predictions(model, x, y)
    w_b = np.ones_like(y_b)
    
    if weight:
        w_b *= len(y_s) / len(y_b)
    
    # histograms
    ax.hist(y_s, bins=bins, alpha=0.55, label='signal', color=palette['signal'],
            edgecolor=palette['signal'], histtype='step', linewidth=2, hatch='//')
    
    ax.hist(y_b, bins=bins, alpha=0.7, label='bkg', color=palette['bkg'],
            histtype='step', edgecolor=palette['bkg'],  weights=w_b)
    
    # compute significance
    cuts = np.linspace(0.0, 1.0, num=bins)
    ams = []
    
    s, _ = np.histogram(y_s, bins=bins, range=(0, 1))
    b, _ = np.histogram(y_b, bins=bins, range=(0, 1), weights=w_b)
    
    for i in range(bins):
        s_i = np.sum(s[i:])
        b_i = np.sum(b[i:])
        
        ams.append(s_i / np.sqrt(s_i + b_i))
    
    if ratio:
        ams_max = num_sig / np.sqrt(num_sig)
        ams = np.array(ams) / ams_max

    ams = np.array(ams)
    ams[np.isnan(ams) | np.isinf(ams)] = 0

    k = np.argmax(ams)
    
    # plot significance and best-cut
    bx = ax.twinx()
    bx.grid(False)
    bx.plot(cuts, ams, color='g', label='Significance')

    ax.axvline(x=cuts[k], linestyle='--', linewidth=2, color='g',
               label=f'{round(cuts[k], digits)}: {round(ams[k].item(), digits)}')
    
    if ratio:
        bx.set_ylabel(r'Significance Ratio: $(s\cdot\sqrt{s_\max}) /(s_\max\cdot\sqrt{s+b})$')
    else:
        bx.set_ylabel(r'Significance: $s / \sqrt{s+b}$')
    
    ax.legend(loc=legend)

    # fix legend z-order issue due to `bx` axis
    bx.set_zorder(1)
    ax.set_zorder(2)
    ax.set_facecolor((0, 0, 0, 0))
    
    ax.set_xlabel('Class Label Probability')
    ax.set_ylabel('Weighted Count')
    
    # title
    str1 = f'{name} Output @ {int(mass)} GeV'
    str2 = f'# signal = {int(np.sum(s))}, # bkg = {len(dataset.background)}'

    ax.set_title(f'{str1}\n{str2}')
    
    if isinstance(save, str):
        path = utils.makedir(path)
        plt.savefig(os.path.join(path, f'{save}.png'), bbox_inches='tight')
    
    if show:
        plt.show()
    
    return np.max(ams), cuts[k]


def significance_vs_mass(models: dict, dataset: Hepmass, bins=50, size=(12, 10), path='plot', 
                         features=None, save=None, weight=True, ratio=True, all_bkg=True, legend='best'):
    fig, axes = plt.subplots(nrows=1, ncols=2)
    
    fig.set_figwidth(size[0] * 2)
    fig.set_figheight(size[1])
    
    features, _, label = _get_columns(dataset, features)
    
    ams = {}
    cut = {}
    
    sig = dataset.signal
    bkg = dataset.background
    
    if not all_bkg:
        weight = False
    else:
        b_values = bkg[features].values
        b_labels = bkg[label]
    
    for name, model in models.items():
        ams[name] = []
        cut[name] = []
        
        for mass in dataset.unique_signal_mass:
            # select data
            s = sig[sig['mass'] == mass]
                
            if not all_bkg:
                b = bkg[bkg['mass'] == mass]
                
                b_values = b[features].values
                b_labels = b[label]
                
            # prepare data
            x = np.concatenate([s[features].values, b_values], axis=0)
            y = pd.concat([s[label], b_labels], axis=0).values.reshape([-1])
            x = dict(x=x, m=np.ones_like(y[:, np.newaxis]) * mass)
    
            # predict data
            out, (y_s, y_b) = _get_predictions(model, x, y)
            w_b = np.ones_like(y_b)
            
            if weight:
                w_b *= len(y_s) / len(y_b)
            
            # compute significance
            best_ams, best_cut = _get_best_ams_cut(y_s, y_b, w_b, bins=bins)
            
            if ratio:
                max_ams = len(s) / np.sqrt(s.shape[0])
                ams[name].append(best_ams / max_ams)
            else:
                ams[name].append(best_ams)

            cut[name].append(best_cut)
    
    # plot AMS and CUT
    mass = dataset.unique_signal_mass
    
    for key, ams_ in ams.items():
        cuts = cut[key]
        
        axes[0].plot(mass, ams_, marker='o', label=f'{key}: {round(np.mean(ams_).item(), 3)}')
        axes[1].plot(mass, cuts, marker='o', label=f'{key}: {round(np.mean(cuts).item(), 3)}')
        
    axes[0].set_xlabel('Mass (GeV)')
    axes[0].set_ylabel('Significance / Max Significance')
    axes[0].set_title(f'Comparison Significance vs Mass [#bins = {bins}; weighted = {weight}]')
    axes[0].set_xticks(mass)
    axes[0].legend(loc=legend)
    
    axes[1].set_xlabel('Mass (GeV)')
    axes[1].set_ylabel('Best Cut')
    axes[1].set_title(f'Comparison Best-Cut vs Mass [#bins = {bins}; weighted = {weight}]')
    axes[1].set_xticks(mass)
    axes[1].legend(loc=legend)
    
    fig.tight_layout()
    
    if isinstance(save, str):
        path = utils.makedir(path)
        plt.savefig(os.path.join(path, f'{save}.png'), bbox_inches='tight')
    
    plt.show()


def auc_vs_mass(dataset: Hepmass, models: dict, bins=50, size=(12, 10), path='plot', legend='best',
                features=None, save=None, ax=None, weight=True, which='ROC', digits=3, **kwargs):
    """Plots the AUC of the ROC curve at each signal's mass""" 
    assert which.upper() in ['ROC', 'PR']
    
    features, _, label = _get_columns(dataset, features)
    
    sig = dataset.signal
    bkg = dataset.background

    if which.upper() == 'ROC':
        curve_fn = cms.plot.roc_auc
        title = 'ROC-AUC'
    else:
        curve_fn = cms.plot.pr_auc
        title = 'PR-AUC'
    
    mass = dataset.unique_signal_mass
    auc = {k: [] for k in models.keys()}

    wb = np.ones((bkg.shape[0],))  # weight for background

    if weight:
        wb /= len(mass)

    b_values = bkg[features].values
    b_labels = bkg[label]
    
    for name, model in models.items():
        for m in mass:
            # select data
            s = sig[sig['mass'] == m]
                
            # prepare data
            x = np.concatenate([s[features].values, b_values], axis=0)
            y = pd.concat([s[label], b_labels], axis=0).values.reshape([-1])
            x = dict(x=x, m=np.ones_like(y[:, np.newaxis]) * m)
    
            # predict data
            out, _ = _get_predictions(model, x, y)
            
            ws = np.ones((s.shape[0],))
            w = np.concatenate([ws, wb], axis=0)

            # compute curve AUC
            _, _, m_auc, _, _ = curve_fn(true=y, pred=out, weights=w, cut=0.5, **kwargs)
            auc[name].append(m_auc)
    
    # plot
    if ax is None:
        plt.figure(figsize=size)
        ax = plt.gca()

        should_show = True
    else:
        should_show = False
    
    ax.set_title(f'{title}')
    
    for k, v in auc.items():
        ax.plot(mass, v, marker='o', label=f'{k}: {round(np.mean(v), digits)}')
    
    ax.set_xlabel('Mass (GeV)')
    ax.set_ylabel('AUC')
        
    ax.legend(loc=legend)
    
    if isinstance(save, str):
        path = utils.makedir(path)
        plt.savefig(os.path.join(path, f'{save}_{int(mass)}m.png'), bbox_inches='tight')
    
    if should_show:
        plt.tight_layout()
        plt.show()
    
    return auc
    

def compare_roc(dataset, models_and_cuts: dict, mass: float, size=(12, 10), digits=3,
                path='plot', save=None, ax=None, legend='lower right', features=None, 
                weight=True, all_bkg=True, **kwargs):
    """Compares ROC curves for different models"""
    features, _, label = _get_columns(dataset, features)
    
    s = dataset.signal[dataset.signal['mass'] == mass]
    b = dataset.background
    
    if not all_bkg:
        weight = False
        b = b[b['mass'] == mass]
    
    # prepare data
    x = pd.concat([s[features], b[features]], axis=0).values
    y = pd.concat([s[label], b[label]], axis=0).values.reshape([-1])
    x = dict(x=x, m=np.ones_like(y[:, np.newaxis]) * mass)
    
    # predict data
    if ax is None:
        plt.figure(figsize=size)
        ax = plt.gca()

        should_show = True
    else:
        should_show = False
    
    ax.set_title(f'ROC @ {int(mass)} GeV')

    for k, (model, cut) in models_and_cuts.items():
        out, _ = _get_predictions(model, x, y)
        w = np.ones_like(out)
        
        if weight:
            w[y == 0.0] = 1.0 / len(dataset.unique_signal_mass)

        fpr, tpr, auc, cut_fpr, cut_tpr = cms.plot.roc_auc(true=y, pred=out, weights=w, 
                                                           cut=cut, **kwargs)
    
        ax.plot(fpr, tpr, label=f'{k}: {np.round(auc, digits)} (AUC)')
        ax.scatter(cut_fpr, cut_tpr, label=f'Significance @ {round(cut, digits)}')
        
    ax.set_xlabel('Background Efficiency (False Positive Rate)')
    ax.set_ylabel('Signal Efficienty (True Positive Rate)')
    ax.legend(loc=legend)
    
    if isinstance(save, str):
        path = utils.makedir(path)
        plt.savefig(os.path.join(path, f'{save}_{int(mass)}m.png'), bbox_inches='tight')
    
    if should_show:    
        plt.show()

        
def compare_pr(dataset, models_and_cuts: dict, mass: float, size=(12, 10), digits=3,
               path='plot', save=None, ax=None, legend='lower left', features=None, 
               weight=True, all_bkg=True, **kwargs):
    """Comparison of Precision-Recall Curves"""
    features, _, label = _get_columns(dataset, features)
    
    s = dataset.signal[dataset.signal['mass'] == mass]
    b = dataset.background

    if not all_bkg:
        weight = False
        b = b[b['mass'] == mass]
    
    # prepare data
    x = pd.concat([s[features], b[features]], axis=0).values
    y = pd.concat([s[label], b[label]], axis=0).values.reshape([-1])
    x = dict(x=x, m=np.ones_like(y[:, np.newaxis]) * mass)
    
    # predict data
    if ax is None:
        plt.figure(figsize=size)
        ax = plt.gca()

        should_show = True
    else:
        should_show = False
    
    ax.set_title(f'Precision-Recall Curve @ {int(mass)} GeV')

    for k, (model, cut) in models_and_cuts.items():
        out, _ = _get_predictions(model, x, y)
        w = np.ones_like(out)
        
        if weight:
            w[y == 0.0] = 1.0 / len(dataset.unique_signal_mass)

        precision, recall, auc, cut_prec, cut_rec = cms.plot.pr_auc(true=y, pred=out, 
                                                                    weights=w, cut=cut, 
                                                                    **kwargs)
        
        ax.plot(recall, precision, label=f'{k}: {np.round(auc, digits)} (AUC)')
        ax.scatter(cut_rec, cut_prec, label=f'Significance @ {round(cut, digits)}')
    
    ax.set_xlabel('Signal Efficiency (Recall)')
    ax.set_ylabel('Purity (Precision)')
    ax.legend(loc=legend)
    
    if isinstance(save, str):
        path = utils.makedir(path)
        plt.savefig(os.path.join(path, f'{save}_{int(mass)}m.png'), bbox_inches='tight')
    
    if should_show:    
        plt.show()


def _get_predictions(model, x, y):
    out = model.predict(x=x, batch_size=1024, verbose=0)
    out = np.asarray(out)

    y_sig = np.squeeze(out[y == 1.0])
    y_bkg = np.squeeze(out[y == 0.0])
    
    return out, (y_sig, y_bkg)


def _get_best_ams_cut(y_sig, y_bkg, w_bkg, bins: int):
    cuts = np.linspace(0.0, 1.0, num=int(bins))
    ams = []
    
    s, _ = np.histogram(y_sig, bins=bins, range=(0, 1))
    b, _ = np.histogram(y_bkg, bins=bins, range=(0,1), weights=w_bkg)

    for i in range(s.shape[0]):
        s_i = np.sum(s[i:])
        b_i = np.sum(b[i:])

        ams.append(s_i / np.sqrt(s_i + b_i))

    ams = np.array(ams)
    ams[np.isnan(ams) | np.isinf(ams)] = 0

    return np.max(ams), cuts[np.argmax(ams)]


def _get_columns(dataset, features):
    if features is None:
        features = dataset.columns['feature']
    
    mass = dataset.columns['mass']
    label = dataset.columns['label']
    
    return features, mass, label
