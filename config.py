# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 09:51:11 2022 (3.10.6 ? )

@author: ManouvanWijlen

DGCNN server configuration file
"""
import os
import argparse
import shutil

####################_______________MOD_B_____________############################
#pulizia delle directory
def clean_directories():
    # Elimina le dir specificate
    to_delete = ['./Input/csv/', './Input/g', './Input/testg', './Output/', './dataset']
    for dir in to_delete:
        if os.path.exists(dir):
            shutil.rmtree(dir)
    # Crea le nuove dir se non esistono
    to_create = ['./Input/csv', './Input/xes', './Input/g', './Input/testg', './Output/petri_nets', './Output/Pre-cage', 
                 './Output/register','./dataset/processed', './Output/checkpoints/best_test', './Output/checkpoints/best_train', 
                 './Output/checkpoints/immagini/best_test', './Output/checkpoints/immagini/best_train', './Output/checkpoints/immagini/cm_epoch']
    for dir in to_create:
        if not os.path.exists(dir):
            os.makedirs(dir)


def load():
    parser=argparse.ArgumentParser() 
    
    parser.add_argument('--attr_list', default=['bu', 'nrchanges', 'ttmotif', 'plant', 'matgroup', 'polines', 'vendor', 'item'])
    #data
    parser.add_argument('--data_dir', default="./dataset")
    #parser.add_argument('--csv_name', default="PreprocessedLog_total_240822_10attr_nonegdpodd.csv")
    parser.add_argument('--csv_name', default="testCsv.csv")
    #parser.add_argument('--xes_name', default="PreprocessedLog_total_240822_10attr_nonegdpodd_startend.xes")
    parser.add_argument('--xes_name', default="testXes.xes")
    parser.add_argument('--net_name', default="./Output/petri_nets") #__MOD_B_#
    parser.add_argument('--checkpoint_dir', default="./Output/checkpoints")
    parser.add_argument('--dataset', default='p2p')
    parser.add_argument('--nrcases',default=4000,type=int)
    
    #training
    parser.add_argument('--patience', default=10,type=int, help="nr of epochs with no improvement (early stopping)")
    parser.add_argument('--per', default=67, type=int, help='percentage training data for train-test split')
    #model
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--num_runs', default=1, type=int)
    parser.add_argument('--num_neurons', default=100,type=int)
    parser.add_argument('--lr_run', default=1, type=int, help="0 lr=1e-03, 1 lr=1e-04, 2 lr=1e-05")
    parser.add_argument('--learning_rate', default=0.0001,type=float)
    parser.add_argument('--batch_size', default=256,type=int)
    parser.add_argument('--batch_size_valid', default=64,type=int)
    parser.add_argument('--k',default=15,type=int)
    parser.add_argument('--dropout',default=0.1, type=float)
    parser.add_argument('--num_layers', default=5, type=int)
    
    args = parser.parse_args()
    return args
