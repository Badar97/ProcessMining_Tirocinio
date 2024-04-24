#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 16:39:51 2024

@author: lgenga
"""

import sys
import pandas as pd
import networkx as nx
import ast
import json
import re

from shutil import rmtree
from os import makedirs
from os.path import join, exists
from config import load, INPUT_G_PATH, INPUT_XES_PATH, CSV_PATH, STATE_PATH
args = load()


pd.set_option('display.max_columns', None) 
pd.set_option('display.width', None) 
pd.set_option('display.max_colwidth', None)



def graphs(lines):
    vertex_re = re.compile('^v *(\d+) *(.*)')
    edge_re = re.compile('^e *(\d+) *(\d*) *(.*)')
    for line in lines:
        if line.strip() == 'XP':
            graph = nx.DiGraph()
        if match := vertex_re.match(line):
            graph.add_node(int(match.group(1)), label=match.group(2))
        elif match := edge_re.match(line):
            graph.add_edge(int(match.group(1)), int(match.group(2)), label=match.group(3))
        elif not line.strip():
            yield graph

def get_subgraph(graph, pos):
    result=""
    #nodes = [node for node in graph.nodes() if int(node) <= pos]
    edges = [(source, target) for source, target in graph.edges() if int(target) <= pos]
    for node, data in graph.nodes(data=True):
        if int(node) <= pos:
            result=result+str(int(node))+" "+data.get('label', 'N/A')+"\n"
    for source, target in edges:
        result=str(result)+str(source)+" "+str(target)+"\n"
    return result

def get_igs_string(row, col_id_name, mapping, graphs):
    #this function takes the status column and for each case retrieves the ig corresponding to all the nodes/edges up to the event
    case_id_event=row[col_id_name]
    global current_case
    if mapping[case_id_event] != current_case:
        current_case=case_id_event
        print("Doing case "+str(mapping[current_case])+" out of "+str(len(graphs)))
    pos_event=row["pos"]
    state=row["state"]
    #aggiunto io
    state_json = json.dumps(state)
    list_of_dicts = ast.literal_eval(state_json)
    graph_string=""
    if len(list_of_dicts)==1:
        for key, value in list_of_dicts.items():
            g=graphs[mapping[key]]
            print("working with IG "+str(mapping[key]))
            subgraph_string=get_subgraph(g, value)
            graph_string=graph_string+subgraph_string     
    else:
            graph_string=graph_string+"XP\n"
            for key, value in list_of_dicts.items():
                if key in mapping:
                    g=graphs[mapping[key]]
                    subgraph_string=get_subgraph(g, value)
                    graph_string=graph_string+subgraph_string
                    graph_string=graph_string+"XP\n"
                else:
                    print('Key '+key+" not found in the mapping")
    
    c_id_ev = case_id_event.replace(' ', '')
    ig_name = c_id_ev + '_' + str(pos_event) + '.g'
    #ig_file = f'{save_path}/stategraphs/' + c_id_ev + '_' + str(pos_event) + '.g'
    ig_file = join(STATE_PATH, 'stategraphs', ig_name)
    f = open(ig_file, "w")
    f.write(graph_string)
    f.close()
    return ig_file

def find_state(row, col_id_name, log):
    #for every event, we want to check for every concurrent case what is the event most close in time which
    #the event we are giving it
    print("doing "+row[col_id_name])
    event_ci=row[col_id_name]
    conc_cases=row['concurrent_cases']
    most_recent_event=dict()
    for c in conc_cases:
        mask=(log[col_id_name]==c) & (log["Timestamp"]<=row["Timestamp"])
        df_filt=log[mask]
        nevent=len(df_filt)
        most_recent_event[c]=nevent
    return most_recent_event

def find_concurrent_cases(row,casespan,col_id_name, mapping):
    event_ci=row[col_id_name]
    print("doing "+str(mapping[event_ci])+" out of "+str(len(casespan)))
    mask=(casespan['min_ts']<=row['Timestamp']) & (casespan['max_ts']>=row['Timestamp']) & (casespan[col_id_name]!=event_ci)
    filtered_cases=casespan[mask]
    return filtered_cases[col_id_name].tolist()

def log2casespan(log, col_id_name):
    result_df = log.groupby(col_id_name)['Timestamp'].agg(['min', 'max']).reset_index()
    result_df.columns = [col_id_name, 'min_ts', 'max_ts']
    return result_df

if __name__ == '__main__':
    #modifica il nome name_file    
    #name_file = 'BPI13O' #****1.33MB****____ok____
    #name_file = 'andreaHelpdesk' #****5.59MB****
    #name_file = 'PrepaidTravelCost' #****8.39MB****____ok____
    #name_file = 'RequestForPayment' #****19.8****____ok____
    #name_file = 'PermitLog_SE_noSpace' #****22.1MB****
    #name_file = 'andrea_bpi12w' #****24.8MB****
    #name_file = 'InternationalDeclarations' #****31.6MB****
    #name_file = 'BPI2012_SE' #****54.1MB****
    #name_file = 'road-start-event' #****248MB****

    if exists(STATE_PATH):
        rmtree(STATE_PATH)

    try:
        log_name = sys.argv[1]
    except (Exception, ):
        log_name = None
    if log_name is None:
        log_name = args.xes_name
    filename = join(INPUT_XES_PATH, log_name)
    if not exists(filename):
        raise FileNotFoundError(f'File: {filename} not found')

    #name_file = filename.split('/')[-1].split('.')[0]
    name_file = filename.split('xes\\')[-1].split('.')[0]

    #xes_file = f'./Input/xes/{name_file}.xes'
    #g_file = f'./Input/g/{name_file}_instance_graphs.g'
    g_file = join(INPUT_G_PATH, f'{name_file}_instance_graphs.g')
    #log = pm4py.read_xes(xes_file)
    #dataframe = pm4py.convert_to_dataframe(log)
    col_id_name = 'Case ID'

    log = pd.read_csv(join(CSV_PATH, f'{name_file}.csv'), sep=',')
    log['pos'] = log.groupby(col_id_name).cumcount() + 1
    
    caseids = log[col_id_name].unique()
    mapping = dict()
    i = 0
    for c in caseids:
        mapping[c] = i
        i = i + 1
    
    if not exists(join(STATE_PATH, 'stategraphs')):
        makedirs(join(STATE_PATH, 'stategraphs'))
    
    casespan = log2casespan(log, col_id_name)
    print('casespan done')
    #casespan.to_csv(f'{STATE_PATH}/casespan.csv', index = False)
    casespan.to_csv(join(STATE_PATH, 'casespan.csv'), index = False)
    log['concurrent_cases'] = log.apply(find_concurrent_cases, args = (casespan, col_id_name, mapping), axis = 1)
    log['state'] = log.apply(find_state, args=(col_id_name, log), axis = 1)
    log.to_csv(join(STATE_PATH, f'{name_file}_status.csv'),sep=';')
    
    with open(g_file) as file:
        graphs = list(graphs(file))
    
    current_case = 0
    print('Doing case ' + str(current_case) + ' out of ' + str(len(caseids)))
    
    log['igs'] = log.apply(get_igs_string, args = (col_id_name, mapping, graphs), axis = 1 )
    log.to_csv(join(STATE_PATH, f'{name_file}_with_graphs.csv'), index = False, sep = ';')