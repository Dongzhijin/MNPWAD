import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import glob
import argparse
import ast
import time
import utils
from datetime import datetime
import json
from pdb import set_trace
from multiprocessing import Process, Manager
from MNPWAD import MNPWAD

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default='data', help='data path')
    parser.add_argument('--datasets', type=str, default='',
                        help='dataset name of the path')
    parser.add_argument('--outputpath', type=str, default='output')
    parser.add_argument('--algo', type=str, default="MNPWAD")
    parser.add_argument('--trainflag', type=str, default="")
    parser.add_argument('--labeled_ratio', type=float, default=0.01)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--use_es', type=ast.literal_eval, default=True)
    parser.add_argument('--base_model', type=str, default='MNPWAD')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--n_emb', type=int, default=8)
    parser.add_argument('--m1', type=float, default=0.02)
    parser.add_argument('--debug', type=ast.literal_eval, default=False)
    parser.add_argument('--pretrainAE', type=ast.literal_eval, default=True)
    parser.add_argument('--n_prototypes', type=int, default=0)
    parser.add_argument('--dataset2n_prototypes', type=str, default='')
    
    
    return parser.parse_args()

def train_model(dataset_name,device,args,results,output_path):
    f = glob.glob(os.path.join(args.datapath, f'{dataset_name}.csv'))
    assert len(f) == 1
    df = pd.read_csv(f[0])
    logger=utils.logger(f'{output_path}/log/{dataset_name}.txt')
    model_name = args.algo
    logger.info("------------------ Dataset: [%s] ------------- Device: [%s]-------------" % (dataset_name,device),print_text=True)
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(method='ffill', inplace=True)
    x = df.values[:, :-1]
    y = np.array(df.values[:, -1], dtype=int)
    x_train, y_train, x_test, y_test, x_val, y_val = utils.split_train_test_val(x, y,
                                                                                test_ratio=0.35,
                                                                                val_ratio=0.05,
                                                                                random_state=2024,
                                                                              del_features=True)
    if args.labeled_ratio<1:
        semi_y = utils.semi_setting_ratio(y_train, labeled_ratio=args.labeled_ratio)
    else:
        semi_y=y_train
    runs=args.runs
    rauc, raucpr, rtime = np.zeros(runs), np.zeros(runs), np.zeros(runs)
    params={'device':device,'use_es':args.use_es,'logger':logger,'base_model':args.base_model,'output_path':output_path,'dataset_name':dataset_name
            ,'batch_size':args.batch_size,'n_emb':args.n_emb,'lr':args.lr,'m1':args.m1,'lambda_kl':args.lambda_kl,'n_prototypes':args.n_prototypes}
    if args.dataset2n_prototypes!='':
        params['n_prototypes']=utils.get_n_prototypes(dataset_name,args.dataset2n_prototypes)
    for i in range(runs):
        st = time.time()
        params['seed'] = 42+i
        model = eval(model_name)(**params)
        model.fit(x_train, semi_y, val_x=x_val, val_y=y_val,pretrainAE=args.pretrainAE)
        score = model.predict(x_test)
        auroc, aupr = utils.evaluate(y_test, score)
        rtime[i] = time.time() - st
        rauc[i] = auroc
        raucpr[i] = aupr

        txt = f'{dataset_name}, AUC-ROC: {auroc:.3f}, AUC-PR: {aupr:.3f}, ' \
            f'time: {rtime[i]:.1f}, runs: [{i+1}/{runs}]'
        logger.info(txt,print_text=False)
        doc = open(output_path+'/middle_result/' + f'{dataset_name}.csv', 'a')
        print(txt, file=doc)
        doc.close()
    print_text = f"{dataset_name}, AUC-ROC, {np.average(rauc):.3f}±{np.std(rauc):.3f}," \
                f" AUC-PR, {np.average(raucpr):.3f}±{np.std(raucpr):.3f}, {np.average(rtime):.1f}," \
                f" {runs}runs, {model.n_prototypes}_prototypes," \
                f" {model_name}, {str(model.param_lst)}"
    logger.info(print_text,print_text=True)
    doc = open(output_path+'/middle_result/' + f'{dataset_name}.csv', 'a')
    print(print_text, file=doc)
    doc.close()
    results[dataset_name] = (print_text,np.average(rauc),np.average(raucpr))
    
if __name__ == '__main__':
    args=parse_args()
    current_date = datetime.now()
    date_string = current_date.strftime('%Y%m%d%H%M')
    output_path=os.path.join(args.outputpath, args.trainflag+'-'+date_string)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        os.makedirs(os.path.join(output_path,'log'), exist_ok=True)
        os.makedirs(os.path.join(output_path,'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(output_path,'middle_result'), exist_ok=True)
    torch.multiprocessing.set_start_method('spawn')
    datasets = args.datasets.split(',')
    if torch.cuda.is_available():
        gpu_list=args.gpu.split(',') if ',' in args.gpu else [args.gpu]
    manager = Manager()
    results = manager.dict()
    processes = []
    for i, dataset in enumerate(datasets):
        device= 'cuda:'+gpu_list[i%len(gpu_list)] if torch.cuda.is_available() else 'cpu'
        p = Process(target=train_model, args=(dataset, device,args, results,output_path))
        processes.append(p)
        p.start()
        if args.debug:
            break
    for p in processes:
        p.join()

    results = dict(results)
    summary = []
    doc = open(output_path+'/all_result.csv', 'a')
    logger=utils.logger(f'{output_path}/all_result.txt')
    for dataset in datasets:
        print(results[dataset][0], file=doc)
        summary.append(results[dataset][1:])
        logger.info(results[dataset][0],print_text=True)
    text = f"avg, AUC-ROC, {np.average([x[0] for x in summary]):.3f}, AUC-PR, {np.average([x[1] for x in summary]):.3f}"
    print(text, file=doc)
    doc.close()
    print(text)
    logger.info(text)