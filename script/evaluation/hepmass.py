import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from script import utils
from script import evaluation as eval_utils

from script.datasets import Hepmass
from script.models import PNN

from typing import Union, List


def metric_with_error(model: PNN, dataset: Hepmass, metric: str, index: int, figsize=(26, 20), num_folds=10, 
                      verbose=0, silent=False, style='bar', return_average=True, **kwargs):
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
    
    if style == 'dot': 
        plt.plot(mass, avg_metric, label='avg')
        plt.scatter(mass, avg_metric, s=50, color='b')
        
        for i in range(num_folds):
            plt.scatter(mass, metric[i], s=30, color='r')
    else:
        values = np.array(list(metric.values()))
        avg_auc = np.array(avg_metric)

        if style == 'bar':
            min_err = avg_metric - np.min(values, axis=0)
            max_err = np.max(values, axis=0) - avg_metric

            plt.errorbar(mass, avg_metric, yerr=np.stack([min_err, max_err]), fmt='ob', 
                         capsize=5.0, elinewidth=1, capthick=1)
            plt.plot(mass, avg_metric, label='avg')
        else:
            min_err = np.min(values, axis=0)
            max_err = np.max(values, axis=0)

            plt.fill_between(mass, min_err, max_err, color='gray', alpha=0.2)

            plt.plot(mass, avg_metric, label='avg')
            plt.scatter(mass, avg_metric, s=50, color='b')
    
    plt.show()

    if return_average:
        return np.mean(list(metric.values()), axis=0)
    
    return metric


def auc_with_error(model: PNN, dataset: Hepmass, index=2, figsize=(26, 20), num_folds=10, verbose=0, 
                   silent=False, style='bar', return_average=True, **kwargs):
    """Computes AUC on disjoint folds of the given data, quantifying how much uncertain the predictions are"""

    return metric_with_error(model, dataset, metric='AUC', index=index, figsize=figsize, num_folds=num_folds, 
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
    
    plt.title(f'AUC vs Mass')
    
    for i, fake_m in enumerate(fake_mass):
        k = scaled_fake_mass[i]
        
        plt.plot(mass, auc[k], label=f'm-{round(fake_m, 2)}')
        plt.scatter(mass, auc[k], s=50)
        
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
        plt.plot(mass, auc[label])
        plt.scatter(mass, auc[label], s=50, label=label)
    
    plt.ylabel('AUC')
    plt.xlabel('Mass')
    
    plt.legend()
    plt.show()
    return auc
