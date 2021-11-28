"""CMS/plot"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, roc_auc_score
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

from script import utils
from script.datasets import Dataset
from script.cms.data import sample

from typing import Union


def significance(model, dataset: Dataset, mass: int, category: int, signal: str, delta=50, 
                 bins=20, size=(12, 10), legend='best', name='Model',
                 path='plot', save=None, show=True, ax=None, ratio=False):
    """Plots the output distribution of the model, along it's weighted significance"""
    if ax is None:
        fig = plt.figure(figsize=size)
        ax = fig.gca()
    
    # select both signal and background in interval (m-d, m+d)
    sig = dataset.signal[dataset.signal['mA'] == mass]
    sig = sig[(sig['dimuon_M'] >= mass - delta) & (sig['dimuon_M'] < mass + delta)]

    bkg = dataset.background
    bkg = bkg[(bkg['dimuon_M'] >= mass - delta) & (bkg['dimuon_M'] < mass + delta)]
    
    num_sig = sig.shape[0]
    
    # prepare data
    x = pd.concat([sig[dataset.columns['feature']],
                   bkg[dataset.columns['feature']]], axis=0).values

    y = np.reshape(pd.concat([sig['type'], bkg['type']], axis=0).values, newshape=[-1])
    x = dict(x=x, m=np.ones_like(y[:, np.newaxis]) * mass)
    
    # predict data
    out = model.predict(x=x, batch_size=1024, verbose=0)
    out = np.asarray(out)

    y_sig = np.squeeze(out[y == 1.0])
    y_bkg = np.squeeze(out[y == 0.0])

    w_bkg = bkg['weight'].values
    w_sig = np.ones_like(y_sig)
    
    # plot
    names = dataset.names_df.loc[bkg.index.values]
    df = pd.DataFrame({'Output': y_bkg, 'Bkg': np.squeeze(names), 'weight': w_bkg})
    
    # plot histograms
    sns.histplot(data=df, x='Output', hue='Bkg', multiple='stack', edgecolor='.3', linewidth=0.5, bins=bins,
                 weights='weight', ax=ax, binrange=(0.0, 1.0),
                 palette={'DY': 'green', 'TTbar': 'red', 'ST': 'blue', 'diboson': 'yellow'})
    
    h_bkg, _ = np.histogram(y_bkg, bins=bins, weights=w_bkg)
    h_bkg = np.sum(h_bkg)
    
    h_sig, _ = np.histogram(y_sig, bins=bins)
    h_sig = np.sum(h_sig)
    
    w_sig = np.ones_like(y_sig) * (h_bkg / h_sig)
    
    ax.hist(y_sig, bins=bins, alpha=0.5, label='signal', color='purple', edgecolor='purple', 
            linewidth=2, hatch='//', histtype='step', range=(0.0, 1.0),
            weights=w_sig)
    
    # compute significance
    sig_mask = np.squeeze(y == 1.0)
    bkg_mask = np.squeeze(y == 0.0)

    cuts = np.linspace(0.0, 1.0, num=bins)
    ams = []
    ams_max = (np.sum(sig_mask) * w_sig[0]) / np.sqrt(np.sum(sig_mask) * w_sig[0])
    
    bx = ax.twinx()
    
    s, _ = np.histogram(y_sig, bins=bins, weights=w_sig)
    b, _ = np.histogram(y_bkg, bins=bins, weights=w_bkg)
    
    for i in range(s.shape[0]):
        s_i = np.sum(s[i:])
        b_i = np.sum(b[i:])
        
        ams.append(s_i / np.sqrt(s_i + b_i))
    
    if ratio:
        ams = np.array(ams) / ams_max
    
    k = np.argmax(ams)
    
    # add stuff to plot
    bx.grid(False)
    bx.plot(cuts, ams, color='g', label='Significance')

    ax.axvline(x=cuts[k], linestyle='--', linewidth=2, color='g',
               label=f'{round(cuts[k], 3)}: {round(ams[k], 3)}')
    
    if ratio:
        bx.set_ylabel(r'Significance Ratio: $(s\cdot\sqrt{s_\max}) /(s_\max\cdot\sqrt{s+b})$')
    else:
        bx.set_ylabel(r'Significance: $s / \sqrt{s+b}$')
    
    leg = ax.get_legend()
    ax.legend(loc='upper left')
    ax.add_artist(leg)
    
    ax.set_xlabel('Class Label Probability')
    ax.set_ylabel('Weighted Num. Events')
    
    # title
    str0 = f'Category-{category} (#bins = {bins})'
    str1 = f'@{(int(mass - delta), int(mass + delta))} dimuon_M (bkg)'
    str2 = f'{name} Output Distribution @ {int(mass)}mA (signal {signal}), {str1}'
    str3 = f'# signal = {sig.shape[0]}, # bkg = {bkg.shape[0]}'
    
    ax.set_title(f'{str0}\n{str2}\n{str3}')
    
    if isinstance(save, str):
        path = utils.makedir(path)
        plt.savefig(os.path.join(path, f'{save}.png'), bbox_inches='tight')
    
    if show:
        plt.show()
    
    return np.max(ams), cuts[k]


def _compute_significance(model, dataset: Dataset, delta=50, bins=20):
    ams = []
    cuts = np.linspace(0.0, 1.0, num=bins)
    
    for mass in dataset.unique_signal_mass:
        # select both signal and background in interval (m-d, m+d)
        sig = dataset.signal[dataset.signal['mA'] == mass]
        sig = sig[(sig['dimuon_M'] >= mass - delta) & (sig['dimuon_M'] < mass + delta)]

        bkg = dataset.background
        bkg = bkg[(bkg['dimuon_M'] >= mass - delta) & (bkg['dimuon_M'] < mass + delta)]
    
        # prepare data
        x = pd.concat([sig[dataset.columns['feature']],
                       bkg[dataset.columns['feature']]], axis=0).values

        y = np.reshape(pd.concat([sig['type'], bkg['type']], axis=0).values, newshape=[-1])
        x = dict(x=x, m=np.ones_like(y[:, np.newaxis]) * mass)
    
        # predict data
        out = model.predict(x=x, batch_size=1024, verbose=0)
        out = np.asarray(out)

        y_sig = np.squeeze(out[y == 1.0])
        y_bkg = np.squeeze(out[y == 0.0])
        
        # compute weights
        w_bkg = bkg['weight'].values
        w_sig = np.ones_like(y_sig)
    
        h_bkg, _ = np.histogram(y_bkg, bins=bins, weights=w_bkg)
        h_bkg = np.sum(h_bkg)

        h_sig, _ = np.histogram(y_sig, bins=bins)
        h_sig = np.sum(h_sig)

        w_sig = np.ones_like(y_sig) * (h_bkg / h_sig)
    
        # compute significance
        sig_mask = np.squeeze(y == 1.0)
        bkg_mask = np.squeeze(y == 0.0)

        ams_ = []
        s, _ = np.histogram(y_sig, bins=bins, weights=w_sig)
        b, _ = np.histogram(y_bkg, bins=bins, weights=w_bkg)

        for i in range(s.shape[0]):
            s_i = np.sum(s[i:])
            b_i = np.sum(b[i:])

            ams_.append(s_i / np.sqrt(s_i + b_i))

        # k = np.argmax(ams)
        ams.append(ams_)
    
    return np.max(ams, axis=-1), cuts[np.argmax(ams, axis=-1)]


def compare_significance(models: list, dataset: Dataset, mass: float, category: int, *args, path='plot', save=None, 
                         size=(12, 10), share_y=True, **kwargs):
    fig, axes = plt.subplots(nrows=1, ncols=len(models), sharey=bool(share_y))
    
    fig.set_figwidth(size[0] * len(models))
    fig.set_figheight(size[1])
    
    for i, model in enumerate(models):
        significance(model, dataset, mass, category, *args, **kwargs, save=None, show=False, ax=axes[i])
    
    if isinstance(save, str):
        path = utils.makedir(path)
        plt.savefig(os.path.join(path, f'{save}_{int(mass)}mA.png'), bbox_inches='tight')
    
    plt.show()

    
def significance_ratio_vs_mass(models_and_data: dict, category: int, signal: str, delta=50, 
                               bins=20, size=(10, 9), path='plot', save=None):
    fig, axes = plt.subplots(nrows=1, ncols=2)
    
    fig.set_figwidth(size[0] * 2)
    fig.set_figheight(size[1])
    
    if isinstance(models_and_data, tuple):
        models_and_data = {'': models_and_data}
    
    ams = {}
    cut = {}
    
    for name, (model, dataset) in models_and_data.items():
        ams[name] = []
        cut[name] = []
        
        for mass in dataset.unique_signal_mass:
            sig = dataset.signal
    
            # select data
            s = sig[sig['mA'] == mass]
            s = s[(s['dimuon_M'] >= mass - delta) & (s['dimuon_M'] < mass + delta)]

            b = dataset.background
            b = b[(b['dimuon_M'] >= mass - delta) & (b['dimuon_M'] < mass + delta)]

            # prepare data
            x = pd.concat([s[dataset.columns['feature']],
                           b[dataset.columns['feature']]], axis=0).values

            y = np.reshape(pd.concat([s['type'], b['type']], axis=0).values, newshape=[-1])
            x = dict(x=x, m=np.ones_like(y[:, np.newaxis]) * mass)
    
            # predict data
            out, y_pred, w = _get_predictions_and_weights(model, x, y, bkg=b, bins=bins)
            best_ams, best_cut = _get_best_ams_cut(*y_pred, *w, bins=bins)
            
            max_ams = (s.shape[0] * w[0][0]) / np.sqrt(s.shape[0] * w[0][0])
            
            ams[name].append(best_ams / max_ams)
            cut[name].append(best_cut)
    
    # plot AMS and CUT
    for key, ams_ in ams.items():
        cuts = cut[key]
        data = models_and_data[key][1]
        
        axes[0].plot(data.unique_signal_mass, ams_, marker='o', label=f'{key}: {round(np.mean(ams_).item(), 3)}')
        axes[1].plot(data.unique_signal_mass, cuts, marker='o', label=f'{key}: {round(np.mean(cuts).item(), 3)}')
        
    axes[0].set_xlabel('Mass (GeV)')
    axes[0].set_ylabel('Significance / Max Significance')
    axes[0].set_title(f'Category-{category}; #bins = {bins} - {signal}\nComparison Significance vs mA')
    axes[0].legend(loc='best')
    
    axes[1].set_xlabel('Mass (GeV)')
    axes[1].set_ylabel('Best Cut')
    axes[1].set_title(f'Category-{category}; #bins = {bins} - {signal}\nComparison Best-Cut vs mA')
    axes[1].legend(loc='best')
    
    fig.tight_layout()
    
    if isinstance(save, str):
        path = utils.makedir(path)
        plt.savefig(os.path.join(path, f'{save}.png'), bbox_inches='tight')
    
    plt.show()
    

def cut(models_and_data, category: int, signal: str, delta=50, bins=20, 
        size=(12, 10), legend='best', name='Model', path='plot', save=None, 
        show=True, ax=None):
    """Plots the value of the best cut as the mass (mA) varies"""
    if ax is None:
        fig = plt.figure(figsize=size)
        ax = fig.gca()
    
    if isinstance(models_and_data, tuple):
        models_and_data = {'': models_and_data}
    
    # cuts = {}
    
    for key, (model, data) in models_and_data.items():
        # cuts[key] = []
        cuts = []
        masses = np.sort(data.unique_signal_mass)
        
        # select both signal and background in interval (m-d, m+d)
        for mass in masses:
            sig = data.signal[data.signal['mA'] == mass]
            sig = sig[(sig['dimuon_M'] >= mass - delta) & (sig['dimuon_M'] < mass + delta)]

            bkg = data.background
            bkg = bkg[(bkg['dimuon_M'] >= mass - delta) & (bkg['dimuon_M'] < mass + delta)]

             # prepare data
            x = pd.concat([sig[data.columns['feature']],
                           bkg[data.columns['feature']]], axis=0).values

            y = np.reshape(pd.concat([sig['type'], bkg['type']], axis=0).values, newshape=[-1])
            x = dict(x=x, m=np.ones_like(y[:, np.newaxis]) * mass)
    
            # predict data
            out, y_pred, w = _get_predictions_and_weights(model, x, y, bkg, bins=bins)
            _, best_cut = _get_best_ams_cut(*y_pred, *w, bins=bins)
            
            # cuts[key].append(best_cut)
            cuts.append(best_cut)
        
        # plot
        ax.plot(masses, cuts, marker='o', label=f'{key}: {round(np.mean(cuts).item(), 3)}')
    
    ax.set_xlabel('Mass (GeV)')
    ax.set_ylabel('Best Cut')
    
    ax.legend(loc=str(legend))
    
    # title
    str0 = f'Category-{category} [#bins = {bins}]'
    ax.set_title(f'[{name}] Best Cut vs mA ({str0})')
    
    if isinstance(save, str):
        path = utils.makedir(path)
        plt.savefig(os.path.join(path, f'{save}.png'), bbox_inches='tight')
    
    if show:
        plt.show()


def curves(model, dataset: Dataset, mass: int, category: int, signal: str, delta=50, 
           bins=20, size=(10, 9), legend='best', seed=utils.SEED, path='plot', save=None):
    """Plots the PR and ROC curves"""
    fig, axes = plt.subplots(nrows=1, ncols=2)
    
    fig.set_figwidth(size[0] * 2)
    fig.set_figheight(size[1])
    
    sig = dataset.signal[dataset.signal['mA'] == mass]
    sig = sig[(sig['dimuon_M'] >= mass - delta) & (sig['dimuon_M'] < mass + delta)]
    
    bkg = dataset.background
    bkg = bkg[(bkg['dimuon_M'] >= mass - delta) & (bkg['dimuon_M'] < mass + delta)]
    
    num_sig = sig.shape[0]
    
    # prepare data
    x = pd.concat([sig[dataset.columns['feature']],
                   bkg[dataset.columns['feature']]], axis=0).values

    y = np.reshape(pd.concat([sig['type'], bkg['type']], axis=0).values, newshape=[-1])
    x = dict(x=x, m=np.ones_like(y[:, np.newaxis]) * mass)
    
    # predict data
    out = model.predict(x=x, batch_size=1024, verbose=0)
    out = np.asarray(out)
    
    y_sig = np.squeeze(out[y == 1.0])
    y_bkg = np.squeeze(out[y == 0.0])
    
    # compute weights
    w_bkg = bkg['weight'].values
    
    h_bkg, _ = np.histogram(y_bkg, bins=bins, weights=w_bkg)
    h_bkg = np.sum(h_bkg)
    
    h_sig, _ = np.histogram(y_sig, bins=bins)
    h_sig = np.sum(h_sig)
    
    w_sig = np.ones_like(y_sig) * (h_bkg / h_sig)
    w = np.concatenate([w_sig, w_bkg], axis=0)
    
    str1 = f'@{(int(mass - delta), int(mass + delta))} dimuon_M (bkg)'
    
    # PR-curve
    PrecisionRecallDisplay.from_predictions(y_true=y, y_pred=out, sample_weight=w, ax=axes[0],
                                            name=f'pNN @ {int(mass)}mA (signal {signal}), {str1}')
    axes[0].set_title(f'[pNN] PR Curve @ {int(mass)}mA (category {category})')
    
    # ROC curve
    RocCurveDisplay.from_predictions(y_true=y, y_pred=out, sample_weight=w, ax=axes[1],
                                     name=f'pNN @ {int(mass)}mA (signal {signal}), {str1}')
    axes[1].set_title(f'[pNN] ROC Curve @ {int(mass)}mA (category {category})')
    
    fig.tight_layout()
    
    if isinstance(save, str):
        path = utils.makedir(path)
        plt.savefig(os.path.join(path, f'{save}.png'), bbox_inches='tight')
    
    plt.show()


def roc_auc(true, pred, weights, cut: float, eps=1e-4):
    fpr, tpr, t = roc_curve(true, pred, sample_weight=weights)
    auc = roc_auc_score(true, pred, average='micro', sample_weight=weights)
    
    # find significance along the curve
    idx = (np.abs(cut - np.array(t))).argmin()
    # idx = (np.abs(cut[:, np.newaxis] - np.array(t))).argmin(axis=-1)

    return fpr, tpr, auc, fpr[idx], tpr[idx]


def pr_auc(true, pred, weights, cut: float, eps=1e-4):
    precision, recall, t = precision_recall_curve(true, pred, sample_weight=weights)
    auc = average_precision_score(true, pred, average='micro', sample_weight=weights)
    
    # find significance along the curve
    idx = (np.abs(cut - t)).argmin()
    
    return precision, recall, auc, precision[idx], recall[idx]


def compare_roc(dataset: Dataset, models_and_cuts: dict, mass: float, category: int, delta=50.0, bins=20,
                size=(12, 10), digits=3, path='plot', save=None, name='Model', **kwargs):
    
    def get_predictions_and_weights(model, x, y, bkg):
        out = model.predict(x=x, batch_size=1024, verbose=0)
        out = np.asarray(out)

        y_sig = np.squeeze(out[y == 1.0])
        y_bkg = np.squeeze(out[y == 0.0])

        # compute weights
        w_bkg = bkg['weight'].values

        h_bkg, _ = np.histogram(y_bkg, bins=bins, weights=w_bkg)
        h_bkg = np.sum(h_bkg)

        h_sig, _ = np.histogram(y_sig, bins=bins)
        h_sig = np.sum(h_sig)

        w_sig = np.ones_like(y_sig) * (h_bkg / h_sig)
        w = np.concatenate([w_sig, w_bkg], axis=0)
        
        return out, w
    
    sig = dataset.signal[dataset.signal['mA'] == mass]
    sig = sig[(sig['dimuon_M'] >= mass - delta) & (sig['dimuon_M'] < mass + delta)]
    
    bkg = dataset.background
    bkg = bkg[(bkg['dimuon_M'] >= mass - delta) & (bkg['dimuon_M'] < mass + delta)]

    # prepare data
    x = pd.concat([sig[dataset.columns['feature']],
                   bkg[dataset.columns['feature']]], axis=0).values

    y = np.reshape(pd.concat([sig['type'], bkg['type']], axis=0).values, newshape=[-1])
    x = dict(x=x, m=np.ones_like(y[:, np.newaxis]) * mass)
    
    # predict data
    plt.figure(figsize=size)
    plt.title(f'[{name}] ROC @ {int(mass)}mA [category-{int(category)}]')
    
    for k, (model, cut) in models_and_cuts.items():
        out, w = get_predictions_and_weights(model, x, y, bkg)
        
        fpr, tpr, auc, cut_fpr, cut_tpr = roc_auc(true=y, pred=out, weights=w, cut=cut, **kwargs)
    
        plt.plot(fpr, tpr, label=f'AUC ({k}) {np.round(auc, digits)}')
        plt.scatter(cut_fpr, cut_tpr, label=f'significance @ {round(cut, digits)}')
    
    plt.xlabel('Background Efficiency (False Positive Rate)')
    plt.ylabel('Signal Efficienty (True Positive Rate)')
    
    plt.legend(loc='lower right')
    
    if isinstance(save, str):
        path = utils.makedir(path)
        plt.savefig(os.path.join(path, f'{save}_{int(mass)}mA.png'), bbox_inches='tight')
    
    plt.show()

    
def compare_pr(dataset: Dataset, models_and_cuts: dict, category: int, mass: float, delta=50.0, bins=20, 
               size=(12, 10), digits=3, path='plot', save=None, name='Model', **kwargs):

    def get_predictions_and_weights(model, x, y, bkg):
        out = model.predict(x=x, batch_size=1024, verbose=0)
        out = np.asarray(out)

        y_sig = np.squeeze(out[y == 1.0])
        y_bkg = np.squeeze(out[y == 0.0])

        # compute weights
        w_bkg = bkg['weight'].values

        h_bkg, _ = np.histogram(y_bkg, bins=bins, weights=w_bkg)
        h_bkg = np.sum(h_bkg)

        h_sig, _ = np.histogram(y_sig, bins=bins)
        h_sig = np.sum(h_sig)

        w_sig = np.ones_like(y_sig) * (h_bkg / h_sig)
        w = np.concatenate([w_sig, w_bkg], axis=0)
        
        return out, w
    
    sig = dataset.signal[dataset.signal['mA'] == mass]
    sig = sig[(sig['dimuon_M'] >= mass - delta) & (sig['dimuon_M'] < mass + delta)]
    
    bkg = dataset.background
    bkg = bkg[(bkg['dimuon_M'] >= mass - delta) & (bkg['dimuon_M'] < mass + delta)]

    # prepare data
    x = pd.concat([sig[dataset.columns['feature']],
                   bkg[dataset.columns['feature']]], axis=0).values

    y = np.reshape(pd.concat([sig['type'], bkg['type']], axis=0).values, newshape=[-1])
    x = dict(x=x, m=np.ones_like(y[:, np.newaxis]) * mass)
    
    # predict data
    plt.figure(figsize=size)
    plt.title(f'[{name}] PR Curve @ {int(mass)}mA [category-{int(category)}]')
    
    for k, (model, cut) in models_and_cuts.items():
        out, w = get_predictions_and_weights(model, x, y, bkg)
        
        precision, recall, auc, cut_prec, cut_rec = pr_auc(true=y, pred=out, weights=w, cut=cut, **kwargs)
        
        plt.plot(recall, precision, label=f'AUC ({k}) {np.round(auc, digits)}')
        plt.scatter(cut_rec, cut_prec, label=f'significance @ {round(cut, digits)}')
    
    plt.xlabel('Recall (signal efficiency)')
    plt.ylabel('Precision (purity)')
    
    plt.legend(loc='lower left')
    
    if isinstance(save, str):
        path = utils.makedir(path)
        plt.savefig(os.path.join(path, f'{save}_{int(mass)}mA.png'), bbox_inches='tight')
    
    plt.show()


def var_posteriori(dataset: Dataset, models: list, variables: list, mass: float, cuts: list, category: int, case: int, 
                   delta=50.0, size=(12, 10), legend='best', bins=25, share_y=True,
                   path='plot', weight=False, seed=utils.SEED, save=None, min_limit=None, max_limit=None):
    assert len(models) == 2, 'only two models can be compared'

    def predict(model, x, cut):
        out = model.predict(x=x, batch_size=1024, verbose=0)
        out = np.asarray(out)
    
        sig_mask = np.squeeze(out >= cut) & (y == 1.0)
        bkg_mask = np.squeeze(out < cut) & (y == 0.0)

        y_sig = np.squeeze(out[sig_mask])
        y_bkg = np.squeeze(out[bkg_mask])

        w_bkg = ds[bkg_mask]['weight'].values
        w_sig = np.ones_like(y_sig)

        # compute signal weights
        h_bkg, _ = np.histogram(y_bkg, bins=bins, weights=w_bkg)
        h_bkg = np.sum(h_bkg)

        h_sig, _ = np.histogram(y_sig, bins=bins)
        h_sig = np.sum(h_sig)

        w_sig = np.ones_like(y_sig) * (h_bkg / h_sig)
        names = dataset.names_df.loc[ds[bkg_mask].index]
        
        return out, (sig_mask, bkg_mask), (y_sig, y_bkg), (w_sig, w_bkg), names 
    
    sig = dataset.signal[dataset.signal['mA'] == mass]
    bkg = dataset.background
    
    # case
    if case == 2:
        sig = sig[(sig['dimuon_M'] >= mass - delta) & (sig['dimuon_M'] < mass + delta)]
        bkg = bkg[(bkg['dimuon_M'] >= mass - delta) & (bkg['dimuon_M'] < mass + delta)]
    
    ds = pd.concat([sig, bkg], axis=0)
    num_sig = sig.shape[0]
    
    # prepare data
    x = pd.concat([sig[dataset.columns['feature']],
                   bkg[dataset.columns['feature']]], axis=0).values

    y = np.reshape(pd.concat([sig['type'], bkg['type']], axis=0).values, newshape=[-1])
    x = dict(x=x, m=np.ones_like(y[:, np.newaxis]) * mass)
    
    # predict data
    out1, masks1, y1, w1, names1 = predict(models[0], x, cuts[0])
    out2, masks2, y2, w2, names2 = predict(models[1], x, cuts[1])
    
    out = [out1, out2]
    mask = [masks1, masks2]
    labels = [y1, y2]
    w = [w1, w2]
    names = [names1, names2]
    
    for col in variables:
        fig, axes = plt.subplots(nrows=1, ncols=2, sharey=bool(share_y))
        
        fig.set_figwidth(size[0] * 2)
        fig.set_figheight(size[1])
        
        ax = fig.gca()
        
        for i, ax in enumerate(axes):
            s = ds[mask[i][0]][col]
            b = ds[mask[i][1]][col]
        
            df = pd.DataFrame({col: b, 'Bkg': np.squeeze(names[i]), 'weight': w[i][1]})

            if ('met' in col) or ('pt' in col):
                n_bins = 50
                max_limit = 500.0

            elif col == 'ljet_n':
                n_bins = 25
                max_limit = 11.0

            elif col == 'dimuon_M':
                n_bins = 25
                max_limit = 500.0
                min_limit = 300
            else:
                n_bins = 25
                max_limit = None

            range_min = min(b.min(), s.min())
            range_max = max(b.max(), s.max())

            if isinstance(min_limit, (int, float)):
                range_min = max(range_min, min_limit)

            if isinstance(max_limit, (int, float)):
                range_max = min(range_max, max_limit)

            # plot histograms
            sns.histplot(data=df, x=col, hue='Bkg', multiple='stack', edgecolor='.3', linewidth=0.5, bins=bins,
                         weights='weight', ax=ax, binrange=(range_min, range_max),
                         palette={'DY': 'green', 'TTbar': 'red', 'ST': 'blue', 'diboson': 'yellow'})

            ax.hist(s, bins=bins, alpha=0.7, label='signal', color='purple', edgecolor='purple', 
                    linewidth=2, hatch='//', histtype='step',
                    range=(range_min, range_max), weights=w[i][0])

            leg = ax.get_legend()
            ax.legend(loc='upper left')
            ax.add_artist(leg)

            ax.set_xlabel(col)
            ax.set_ylabel('Count')
            
            fig.tight_layout()
        
        if isinstance(save, str):
            path = utils.makedir(path)
            plt.savefig(os.path.join(path, f'{save}_{col}_{int(mass)}_mA.png'), bbox_inches='tight')
        
        plt.show()


def curve_vs_mass(models_and_data: dict, category: int, signal: str, delta=50, bins=20, 
                  size=(12, 10), path='plot', save=None, title='pNN', auc=False, curve='roc', 
                  legend='best', **kwargs):
    is_roc = curve.lower() == 'roc'
    
    if isinstance(models_and_data, tuple):
        models_and_data = {'': models_and_data}
    
    if auc:
        curve = {name: [] for name in models_and_data.keys()} 
    
    elif is_roc:
        curve = {name: dict(FPR=[], TPR=[]) for name in models_and_data.keys()}
    else:
        curve = {name: dict(precision=[], recall=[]) for name in models_and_data.keys()}
    
    for name, (model, dataset) in models_and_data.items():
        for mass in dataset.unique_signal_mass:
            sig = dataset.signal
    
            # select data
            s = sig[sig['mA'] == mass]
            s = s[(s['dimuon_M'] >= mass - delta) & (s['dimuon_M'] < mass + delta)]

            b = dataset.background
            b = b[(b['dimuon_M'] >= mass - delta) & (b['dimuon_M'] < mass + delta)]

            # prepare data
            x = pd.concat([s[dataset.columns['feature']],
                           b[dataset.columns['feature']]], axis=0).values

            y = np.reshape(pd.concat([s['type'], b['type']], axis=0).values, newshape=[-1])
            x = dict(x=x, m=np.ones_like(y[:, np.newaxis]) * mass)

            # compute curve @ cut
            out, y_pred, w = _get_predictions_and_weights(model, x, y, bkg=b, bins=bins)
            _, cut = _get_best_ams_cut(*y_pred, *w, bins=bins)
            w = np.concatenate(w, axis=0)

            if is_roc:
                _, _, auc_, cut_fpr, cut_tpr = cms.plot.roc_auc(true=y, pred=out, weights=w, cut=cut, **kwargs)

                if auc:
                    curve[name].append(auc_)
                else:
                    curve[name]['FPR'].append(cut_fpr)
                    curve[name]['TPR'].append(cut_tpr)
            else:
                _, _, auc_, cut_prec, cut_rec = cms.plot.pr_auc(true=y, pred=out, weights=w, cut=cut, **kwargs)

                if auc:
                    curve[name].append(auc_)
                else:
                    curve[name]['precision'].append(cut_prec)
                    curve[name]['recall'].append(cut_rec)
    # plot
    plt.figure(figsize=size)
    plt.title(f'[{title}] {"ROC" if is_roc else "PR"} Curve @ Best-Cut vs mA [category-{int(category)}]')
    
    ax = plt.gca()
    
    if not auc:
        bx = plt.twinx()
        bx.grid(False)
        
        for name, (_, dataset) in models_and_data.items():
            mass = dataset.unique_signal_mass
            
            for key, value in curve[name].items():
                plt.plot(mass, value, marker='o', label=f'{key}-{name}: {round(np.mean(value).item(), 2)}')
    else:
        for name, (_, dataset) in models_and_data.items():
            mass = dataset.unique_signal_mass
            
            plt.plot(mass, curve[name], marker='o', label=f'AUC-{name}: {round(np.mean(curve[name]).item(), 2)}')
    
    ax.set_xlabel('Mass (GeV)')
    
    if auc:
        ax.set_ylabel('AUC')
    
    elif is_roc:
        bx.set_ylabel('Signal Efficiency (TPR)')
        ax.set_ylabel('Background Efficiency (FPR)')
    else:
        bx.set_ylabel('Precision (purity)')
        ax.set_ylabel('Recall (signal efficiency)')
    
    plt.legend(loc=str(legend).lower())
    plt.show()

    
def _get_predictions_and_weights(model, x, y, bkg, bins):
        out = model.predict(x=x, batch_size=1024, verbose=0)
        out = np.asarray(out)

        y_sig = np.squeeze(out[y == 1.0])
        y_bkg = np.squeeze(out[y == 0.0])

        # compute weights
        w_bkg = bkg['weight'].values

        h_bkg, _ = np.histogram(y_bkg, bins=bins, weights=w_bkg)
        h_bkg = np.sum(h_bkg)

        h_sig, _ = np.histogram(y_sig, bins=bins)
        h_sig = np.sum(h_sig)

        w_sig = np.ones_like(y_sig) * (h_bkg / h_sig)
        
        return out, (y_sig, y_bkg), (w_sig, w_bkg)

    
def _get_best_ams_cut(y_sig, y_bkg, w_sig, w_bkg, bins):
    cuts = np.linspace(0.0, 1.0, num=bins)
    ams = []

    s, _ = np.histogram(y_sig, bins=bins, weights=w_sig)
    b, _ = np.histogram(y_bkg, bins=bins, weights=w_bkg)

    for i in range(s.shape[0]):
        s_i = np.sum(s[i:])
        b_i = np.sum(b[i:])

        ams.append(s_i / np.sqrt(s_i + b_i))

    return np.max(ams), cuts[np.argmax(ams)]
