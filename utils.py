from sklearn import metrics
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random
import string
import os
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import os
# from tikzplotlib import save as tikz_save
import math
from ipdb import set_trace
mask = ''.join(random.sample(string.ascii_letters, 8))

def split_train_test_val_unseen(x, y, test_ratio, val_ratio, random_state=None, del_features=True):
    idx_norm = y == 0
    idx_out = y !=0

    n_f = x.shape[1]

    if del_features:
        del_list = []
        for i in range(n_f):
            if np.std(x[:, i]) == 0:
                del_list.append(i)
        if len(del_list) > 0:
            print("Pre-process: Delete %d features as every instances have the same behaviour: " % len(del_list))
            x = np.delete(x, del_list, axis=1)

    x_train_norm, x_teval_norm, y_train_norm, y_teval_norm = train_test_split(x[idx_norm], y[idx_norm],
                                                                              test_size=test_ratio + val_ratio,
                                                                              random_state=random_state)
    x_test_norm, x_val_norm, y_test_norm, y_val_norm = train_test_split(x_teval_norm, y_teval_norm,
                                                                        test_size=val_ratio / (test_ratio + val_ratio),
                                                                        random_state=random_state)
    x_train,x_test,x_val=x_train_norm,x_test_norm,x_val_norm
    y_train,y_test,y_val= y_train_norm,y_test_norm,y_val_norm
    x_out,y_out=x[idx_out], y[idx_out]
    for i in np.unique(y_out):
        idx_out=y==i
        x_train_out, x_teval_out, y_train_out, y_teval_out = train_test_split(x[idx_out], y[idx_out],
                                                                            test_size=test_ratio + val_ratio,
                                                                            random_state=random_state)
        x_test_out, x_val_out, y_test_out, y_val_out = train_test_split(x_teval_out, y_teval_out,
                                                                        test_size=val_ratio / (test_ratio + val_ratio),
                                                                        random_state=random_state)
        x_train = np.concatenate((x_train, x_train_out))
        x_test = np.concatenate((x_test, x_test_out))
        x_val = np.concatenate((x_val, x_val_out))
        y_train = np.concatenate((y_train, y_train_out))
        y_test = np.concatenate((y_test, y_test_out))
        y_val = np.concatenate((y_val, y_val_out))

    from collections import Counter
    print('train counter', Counter(y_train))
    print('val counter  ', Counter(y_val))
    print('test counter ', Counter(y_test))

    return x_train, y_train, x_test, y_test, x_val, y_val

def unseen_setting(x_train, y_train, x_test, y_test, x_val, y_val, unseen_class):
    """
    unseen setting
    """
    idx_known = y_train != unseen_class
    x_train = x_train[idx_known]
    y_train = y_train[idx_known]
    y_train[y_train != 0] = 1
    idx_known=y_val!=unseen_class
    x_val=x_val[idx_known]
    y_val=y_val[idx_known]
    y_val[y_val!=0]=1
    
    idx_known = (y_test == unseen_class) | (y_test == 0)
    x_test = x_test[idx_known]
    y_test = y_test[idx_known]
    y_test[y_test != 0] = 1
    
    print("unseen setting:")
    from collections import Counter
    print('train counter', Counter(y_train))
    print('val counter  ', Counter(y_val))
    print('test counter ', Counter(y_test))
    
    # Scale to range [0,1]
    minmax_scaler = MinMaxScaler()
    minmax_scaler.fit(x_train)
    x_train = minmax_scaler.transform(x_train)
    x_test = minmax_scaler.transform(x_test)
    x_val = minmax_scaler.transform(x_val)
    return x_train, y_train, x_test, y_test, x_val, y_val

class logger:
    def __init__(self,filepath):
        self.filepath=filepath
    def info(self,s,print_text=False):
        if print_text:
            print(s)
        with open(self.filepath, 'a') as file:
            file.write(s + '\n') 
        file.close()

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=1e-4,
                 model_name="", trace_func=print, structrue='torch'):
        self.structure = structrue

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

        self.trace_func = trace_func
        if structrue == 'torch':
            # self.path = "checkpoints/" + model_name + "." + mask + '_checkpoint.pt'
            # self.path = "checkpoints/" + model_name + "_" + date_string + '_checkpoint.pt'
            self.path=model_name+'_checkpoint.pt'
        elif structrue == 'keras':
            self.path = "checkpoints/" + model_name + '.' + mask + "_checkpoint.h5"
        if not os.path.exists(os.path.split(self.path)[0]):
            os.mkdir(os.path.split(self.path)[0])
        self.current_best=False

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score <= self.best_score + self.delta:
            if self.counter==0:
                self.current_best=True
            else:
                self.current_best=False
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        if self.structure == 'torch':
            torch.save(model.state_dict(), self.path)
        elif self.structure == 'keras':
            model.save(self.path)

        self.val_loss_min = val_loss


def evaluate(y_true, y_prob):
    if np.isnan(y_prob).any():
        print('score has nan')
        auroc, aupr=0,0
    else:
        auroc = metrics.roc_auc_score(y_true, y_prob)
        precision, recall, _ = metrics.precision_recall_curve(y_true, y_prob)
        aupr = metrics.auc(recall, precision)
    return auroc, aupr

def split_train_test(x, y, test_size, random_state=None):
    idx_norm = y == 0
    idx_out = y == 1

    n_f = x.shape[1]
    del_list = []
    for i in range(n_f):
        if np.std(x[:, i]) == 0:
            del_list.append(i)
    if len(del_list) > 0:
        print("Pre-process: Delete %d features as every instances have the same behaviour: " % len(del_list))
        x = np.delete(x, del_list, axis=1)

    # keep outlier ratio, norm is normal out is outlier
    x_train_norm, x_test_norm, y_train_norm, y_test_norm = train_test_split(x[idx_norm], y[idx_norm],
                                                                            test_size=test_size,
                                                                            random_state=random_state)
    x_train_out, x_test_out, y_train_out, y_test_out = train_test_split(x[idx_out], y[idx_out],
                                                                        test_size=test_size,
                                                                        random_state=random_state)
    x_train = np.concatenate((x_train_norm, x_train_out))
    x_test = np.concatenate((x_test_norm, x_test_out))
    y_train = np.concatenate((y_train_norm, y_train_out))
    y_test = np.concatenate((y_test_norm, y_test_out))

    # # Scale to range [0,1]
    minmax_scaler = MinMaxScaler()
    minmax_scaler.fit(x_train)
    x_train = minmax_scaler.transform(x_train)
    x_test = minmax_scaler.transform(x_test)

    return x_train, y_train, x_test, y_test


def split_train_test_val(x, y, test_ratio, val_ratio, random_state=None, del_features=True):
    idx_norm = y == 0
    idx_out = y == 1

    n_f = x.shape[1]

    if del_features:
        del_list = []
        for i in range(n_f):
            if np.std(x[:, i]) == 0:
                del_list.append(i)
        if len(del_list) > 0:
            print("Pre-process: Delete %d features as every instances have the same behaviour: " % len(del_list))
            x = np.delete(x, del_list, axis=1)

    x_train_norm, x_teval_norm, y_train_norm, y_teval_norm = train_test_split(x[idx_norm], y[idx_norm],
                                                                              test_size=test_ratio + val_ratio,
                                                                              random_state=random_state)
    x_train_out, x_teval_out, y_train_out, y_teval_out = train_test_split(x[idx_out], y[idx_out],
                                                                          test_size=test_ratio + val_ratio,
                                                                          random_state=random_state)

    x_test_norm, x_val_norm, y_test_norm, y_val_norm = train_test_split(x_teval_norm, y_teval_norm,
                                                                        test_size=val_ratio / (test_ratio + val_ratio),
                                                                        random_state=random_state)
    x_test_out, x_val_out, y_test_out, y_val_out = train_test_split(x_teval_out, y_teval_out,
                                                                    test_size=val_ratio / (test_ratio + val_ratio),
                                                                    random_state=random_state)

    x_train = np.concatenate((x_train_norm, x_train_out))
    x_test = np.concatenate((x_test_norm, x_test_out))
    x_val = np.concatenate((x_val_norm, x_val_out))
    y_train = np.concatenate((y_train_norm, y_train_out))
    y_test = np.concatenate((y_test_norm, y_test_out))
    y_val = np.concatenate((y_val_norm, y_val_out))

    from collections import Counter
    print('train counter', Counter(y_train))
    print('val counter  ', Counter(y_val))
    print('test counter ', Counter(y_test))

    # # Scale to range [0,1]
    minmax_scaler = MinMaxScaler()
    minmax_scaler.fit(x_train)
    x_train = minmax_scaler.transform(x_train)
    x_test = minmax_scaler.transform(x_test)
    x_val = minmax_scaler.transform(x_val)

    return x_train, y_train, x_test, y_test, x_val, y_val


def semi_setting_ratio(y_train, labeled_ratio=0.01):
    """
    default: using ratio to get known outliers, also can using n_known_outliers to get semi-y
    use the first k outlier as known
    :param y_train:
    :param labeled_ratio:
    :return:
    """
    outlier_indices = np.where(y_train == 1)[0]
    n_outliers = len(outlier_indices)
    n_known_outliers = min(math.ceil(n_outliers*labeled_ratio), n_outliers)
    print('anomaly_known_n:',n_known_outliers)

    known_idx = outlier_indices[:n_known_outliers]

    new_y_train = np.zeros_like(y_train, dtype=int)
    new_y_train[known_idx] = 1
    return new_y_train

def get_n_prototypes(dataset_name,dataset2n_prototypes):
    path=dataset2n_prototypes+'.csv'
    if os.path.exists(path):
        df=pd.read_csv(path)
        if dataset_name in df['dataset'].values:
            n_prototypes=df.loc[df['dataset']==dataset_name,'n_prototypes'].values[0]
        else:
            n_prototypes=0
    else:
        n_prototypes=0
    return int(n_prototypes)