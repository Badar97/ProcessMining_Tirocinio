from pm4py import convert_to_dataframe, write_pnml, read_xes, write_xes
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.visualization.petri_net.common import visualize as pn_visualize
from pm4py.objects.petri_net.obj import PetriNet
from pm4py.objects.log.util import artificial
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.streaming.importer.xes import importer as xes_importer
from pm4py.analysis import check_soundness
from IPython import display
from graphviz import Digraph
from pm4py.read import read_pnml

import sys
from time import time
import csv
from os.path import join, exists, isfile
from os import listdir
from config import INPUT_PATH, load, clean_directories, \
INPUT_XES_PATH, LOGS_WITH_NO_START_END, INPUT_G_PATH, OUTPUT_PN_PATH, CSV_PATH
args = load()
csv_path = args.csv_name


def check_start_end_traces(log_path):
    log = read_xes(log_path)
    convert_xes_to_csv(log, join(CSV_PATH, csv_path))
    log_name = log_path.split('/')[-1]
    log_name_no_ext = log_name.split('.')[0]
    files_in_path = []
    files_in_path += [item for item in listdir(INPUT_XES_PATH) if isfile(join(INPUT_XES_PATH, item))]
    # if log is not in list, it has START and END activities
    if log_name_no_ext in LOGS_WITH_NO_START_END:
        new_log_name = f'{log_name_no_ext}_correct.xes'
        # if not exists, generating it with start and end activities
        print('Adding START and END activities to all traces..')
        new_log = artificial.insert_artificial_start_end(log, parameters={
            artificial.Parameters.PARAM_ARTIFICIAL_START_ACTIVITY: 'START',
            artificial.Parameters.PARAM_ARTIFICIAL_END_ACTIVITY: 'END'
        })
        print(f'Saving new log: {new_log_name}..\n->\tUse it next time to avoid this step ;)')
        write_xes(new_log, join(INPUT_XES_PATH, new_log_name))
        return new_log, new_log_name
    else:
        return log, log_name


def extract_process_model(log_name, log, net_path=None, threshold=0.4):
    if net_path is not None:
        print('Loading process model..')
        net, initial_marking, final_marking = read_pnml(net_path)
        return net, initial_marking, final_marking
    else:
        print('Extracting process model..')
        net, initial_marking, final_marking = (inductive_miner.apply(
            log, variant=inductive_miner.Variants.IMf, parameters=
            {inductive_miner.Variants.IMf.value.Parameters.NOISE_THRESHOLD: threshold}))
        print('Saving process model..')
        write_pnml(net, initial_marking, final_marking,
                   join(OUTPUT_PN_PATH, f'{log_name.split(".")[0]}_petri_net.pnml'))

        # gviz = pn_visualize.apply(net, initial_marking, final_marking)
        # gviz.render(filename="petri")

        print('Process model infos..')
        check_soundness(net, initial_marking, final_marking)
        return net, initial_marking, final_marking


# This function convert the .xes file to .csv file
def convert_xes_to_csv(log_path, csv_path):
    df_xes = convert_to_dataframe(log_path)  # convert il file .xes in un dataframe
    df_xes = df_xes.rename(columns={'case:concept:name': 'Case ID', 'time:timestamp': 'Timestamp'})
    df_xes.drop(['Activity'], errors='ignore', axis=1, inplace=True)
    df_xes = df_xes.rename(columns={'concept:name': 'Activity'})
    # df_xes = df_xes.rename(columns=lambda x: x.replace('case:', ''))
    column_sort = ['Case ID', 'Activity', 'Timestamp'] + [col for col in df_xes.columns if
                                                          col not in ['Case ID', 'Activity', 'Timestamp']]
    df_xes = df_xes[column_sort]
    df_xes.to_csv(csv_path, index=False)
    return df_xes


def find_causal_relationships(net):
    dict_succ = find_successors(net)
    result = []
    for key, item in dict_succ.items():
        for s in item:
            result.append((key.label, s.label))
    return result


def find_successors_of_transition(net, transition):
    sources = {transition}
    targets = set()
    visited = set()
    while sources:
        source = sources.pop()
        if not (type(source) is PetriNet.Transition and source.label is not None):
            visited.add(source)
        for arc in source.out_arcs:
            if arc.target in visited:
                continue
            if type(arc.target) is PetriNet.Transition and arc.target.label is not None:
                targets.add(arc.target)
            else:
                sources.add(arc.target)
    return targets


def find_successors(net):
    return {transition: find_successors_of_transition(net, transition) for transition in net.transitions if
            transition.label is not None}


# Questa funzione riceve in input la traccia, la rete di petri, il marking iniziale e finale.
# restituisce in output due liste, una contenente le tracce allineate al modello,
# l'altra con le tracce allineata rispetto al log.
def pick_aligned_trace(trace, net, initial_marking, final_marking):
    aligned_traces = alignments.apply_trace(trace, net, initial_marking, final_marking)
    temp = []
    id = 0
    al = []
    temp1 = []
    id1 = 0
    fin = []

    for edge in aligned_traces['alignment']:
        id += 1
        temp.append((id, edge[1]))
    al.append(temp)
    for edge in aligned_traces['alignment']:
        id1 += 1
        temp1.append((id1, edge[0]))
    fin.append(temp1)

    return al, fin


# Check Trace Conformance
# This function checks whether a given trace is conform to a petri-net through alignment based conformance checking.
# It takes in input the trace, the petri-net, the initial marking, and the final marking.
# It returns two lists: a list of sequences of deleted activities, and a list of sequences of inserted activities
# in the trace. Each element of the two lists is a list itself, this sublist contains pairs where the first element
# is the position of the deleted/inserted activity and the second is the activity label.
# In sequences of deleted activities the position value is always the same,
# while for inserted activities it's incremental.
def check_trace_conformance(trace, net, initial_marking, final_marking):
    aligned_traces = alignments.apply_trace(trace, net, initial_marking, final_marking)
    D = []
    I = []
    id = 0
    temp_d = []
    temp_i = []
    prev_d = False
    curr_d = False
    prev_i = False
    curr_i = False
    del_count = 1
    for edge in aligned_traces['alignment']:
        id += 1
        if edge[1] is None:
            id -= 1
            continue
        if edge[0] == '>>':
            temp_d.append((id, edge[1]))
            curr_d = True
            id -= 1
        if edge[1] == '>>':
            temp_i.append((id, edge[0]))
            curr_i = True

        if prev_i and not curr_i:
            if len(temp_i) > 0:
                I.append(temp_i)
            temp_i = []
        prev_i = curr_i
        curr_i = False
        if prev_d and not curr_d:
            if len(temp_d) > 0:
                D.append(temp_d)
            temp_d = []

        prev_d = curr_d
        curr_d = False
    if len(temp_i) > 0:
        I.append(temp_i)
    if len(temp_d) > 0:
        D.append(temp_d)
    return D, I


# MAPPING
# La funzione mapping prende in input le due liste contenente le due tracce allineate.
# Crea quindi una lista di tuple di 3 elementi dove:
# - il primo elemento della tupla corrisponde al nome dell'attività
# - il secondo elemento della tupla corrisponde alla posizione dell'attività nella traccia presente nel log
# - il terzo elemento della tupla corrisponde alla posizione dell'attività nella traccia allineata rispetto al modello;
#   alle attività soggette a insertion (e quindi non presenti nel modello) viene assegnata la posizione a partire
#   dalla lunghezza della traccia (quindi se ho una traccia di 10 elementi, la prima insertion avrà posizione 10+1,
#   la seconda 10+2 ecc)
# Il mapping permette di capire quali sono le attività inserite nel grafo,
# quindi la funzione restituisce anche una lista ins contenente le insertion.
# crea il mapping, ritorna il map e la lista ins (lista degli inserimenti)
def mapping(L1, L2):
    map = [0] * len(L1)
    id1 = 0
    id2 = 0
    ins = []
    for i in range(len(L1)):
        e1 = L1[i]
        e2 = L2[i]
        if e1[1] == e2[1]:
            id1 += 1
            id2 += 1
            map[i] = (e1[1], id1, id2)
        elif e1[1] == '>>':  # insertion
            id1 += 1
            map[i] = (e2[1], id1, 0)
        elif e2[1] == '>>':  # deletion
            id2 += 1
            map[i] = (e1[1], 0, id2)

    for j in range(len(L1)):
        e1 = L1[j]
        e3 = map[j]
        if e1[1] == '>>':
            id2 += 1
            map[j] = (e3[0], e3[1], id2)
            ins.append((e3[0], e3[1], id2))

    return map, ins


# Extract Instance Graph
# This function extracts the instance graph of a trace.
# Every event of the trace is saved in a list V which represents the set of the nodes of the graph.
# An event is a pair of an ID (generated incrementally) and the activity label.
# The edges instead are saved as a pair of events in a list W.
# The algorithm is based on the definition 18 of the original paper.
def extract_instance_graph(trace, cr):
    V = []
    W = []
    id = 1
    for event in trace:
        # V.append((id, event.get("concept:name")))
        V.append(event)
        id += 1
    # print("IG")
    for i in range(len(V)):
        for k in range(i + 1, len(V)):
            e1 = V[i]
            e2 = V[k]
            if (e1[1], e2[1]) in cr:
                flag_e1 = True
                for s in range(i + 1, k):
                    e3 = V[s]
                    if (e1[1], e3[1]) in cr:
                        flag_e1 = False
                        break
                flag_e2 = True
                for s in range(i + 1, k):
                    e3 = V[s]
                    if (e3[1], e2[1]) in cr:
                        flag_e2 = False
                        break

                if flag_e1 or flag_e2:
                    W.append((e1, e2))
    return V, W


# View Instance Graph
# This function takes in input the list of nodes (events) and the list of edges and returns a GraphViz object.
# The other two inputs are view and the title of the graph.
# By default, view is true so the function will show the graph in the output window.
def view_instance_graph(V, W, view=True, title="Instance Graph"):
    # Conversion to string indexes
    V2 = []
    W2 = []
    for node in V:
        V2.append((str(node[0]), "{0} = {1}".format(node[0], node[1])))
    for edge in W:
        W2.append(((str(edge[0][0]), "{0} = {1}".format(edge[0][0], edge[0][1])),
                   (str(edge[1][0]), "{0} = {1}".format(edge[1][0], edge[1][1]))))

    # print('V2: ', V2)
    # print('W2: ', W2)

    dot = Digraph(comment=title, node_attr={'shape': 'circle'})
    for e in V2:
        dot.node(e[0], e[1])
    for w in W2:
        dot.edge(w[0][0], w[1][0])
    if view:
        display.display(dot)

    return dot


# Compliant trace
# Prende in input la traccia ed elimina dalla stessa i move '>>'
def compliant_trace(trace):
    t = []
    id = 0
    for event in trace:
        if event[1] == '>>':
            continue
        else:
            id += 1
            t.append((id, event[1]))

    return t


# Deletion Repair
# Questa funzione descrive il funzionamento della deletion repair. 
# Prende in input la lista dei nodi, la lista degli archi, il mapping e la lista delle deletion.
# restituisce in output i nuovi nodi e archi ottenuti a seguito della deletion repair.
def del_repair(V, W, map, deletion):
    Eremp = []
    Erems = []
    Pred = []
    Succ = []
    W1 = []
    V1 = []
    d = []
    W2 = []

    to_del = (deletion[2], deletion[0])
    # print('Da cancellare = ', to_del)
    for i in range(len(W)):
        e1 = W[i]
        e2 = e1[1]
        e3 = e1[0]
        if e2 == to_del:
            Eremp.append((e3, to_del))
        if e3 == to_del:
            Erems.append((to_del, e2))

    for a in Eremp:  # crea liste Pred e Succ
        Pred.append(a[0])
    for b in Erems:
        Succ.append(b[1])

    for ep in Eremp:
        W.remove(ep)
    for es in Erems:
        W.remove(es)

    V.remove(to_del)

    for p in Pred:
        for s in Succ:
            W.append((p, s))

    return V, W


# Insertion Repair
# Questa funzione descrive il procedimento dell'insertion repair.
def ins_repair(V, W, map, insertion, V_n, ins_list, Vpos):
    Eremp = []
    Pred = []
    Succ = []
    pos_t = []
    W_num = []
    V1 = []
    # ins_num = []

    V.insert(insertion[1] - 1, (insertion[2], insertion[0]))
    Vpos.insert(insertion[1] - 1, (insertion[2], insertion[0]))
    pos_t.append(insertion[1])

    W_num = edge_number(W)
    V_num = node_number(V)
    # ins_num = ins_list_num(ins_list)

    # print('ins num: ',ins_num)

    # print('Vpos agg: ', Vpos)

    for p in pos_t:  # numero dell'insertion
        # print('P=',p)
        # print('Len Vpos: ', len(Vpos))
        if p < len(Vpos):
            position = Vpos[p]  # posizione in cui va inserito il nodo
            # print('Position = ', position)
            pos = position[0]
        else:
            # Inserimento a ultimo posto. La posizione di inserimento è maggiore o uguale della lunghezza
            # di Vpos (che non considera nodi da cancellare)
            position = V[-1]
            pos = position[0]
            # print('ULtimo elemento ', Vpos[-2])
            # print(pos)

        # print('pos: ', pos)
        # in Vpos il nodo da inserire viene messo in posizione p-1 perchè il vettore parte da 0,
        # quindi il precedente lo trovo come p-2
        p_pred = Vpos[p - 2]
        pos_pred = p_pred[0]
        # print('P pred: ', p_pred)
        # linee 6-12 pseudocodice
        if is_path(pos_pred, pos, W_num, V):  # se c'è un cammino tra p-1 e p
            # print(is_path(pos_pred,pos,W_num,V))
            for i in range(len(W)):
                arc = W[i]
                a0 = arc[0]
                a1 = arc[1]
                if pos == a1[0] and (a0, a1) not in Eremp:
                    # se esiste un arco nel grafo che entra in posizione p e non è ancora stato inserito in Eremp si
                    # trovano gli archi entranti (e quindi i nodi Pred) nel nodo in posizione in cui va
                    # fatto l'inserimento
                    Eremp.append((a0, a1))
                    Pred.append(a0)
            for n in Pred:
                # linee 9-10 pseudocodice, si controllano eventuali parallelismi non considerati nel ciclo precedente
                for k in range(len(W)):
                    e = W[k]
                    e0 = e[0]
                    e1 = e[1]
                    if e0 == n and (e0, e1) not in Eremp:
                        Eremp.append((e0, e1))
        else:
            # linee 14-15 pseudocodice, l'insertion avviene all'interno di un parallelismo
            for m in range(len(W)):
                edge = W[m]
                edge0 = edge[0]
                edge1 = edge[1]
                if pos_pred == edge0[0] and (edge0, edge1) not in Eremp:
                    Eremp.append((edge0, edge1))
                    Pred.append(edge0)
                elif pos_pred == edge1[0] and pos_pred == V_n[-1]:
                    # insertion all'ultimo nodo del grafo
                    Pred.append(edge1)

    # linea 17 pseudocodice
    for erem in range(len(Eremp)):
        suc = Eremp[erem]
        suc1 = suc[1]
        if suc1 not in Succ:
            Succ.append(suc1)

    # print('Pred = ', Pred)
    # print('Succ = ', Succ)
    # print('Eremp = ', Eremp)

    # linea 18 pseudocodice
    for el in Eremp:
        if el in W:
            W.remove(el)

    for i in Pred:
        if (i, (insertion[2], insertion[0])) not in W:
            W.append((i, (insertion[2], insertion[0])))

    for s in Succ:
        if ((insertion[2], insertion[0]), s) not in W:
            W.append(((insertion[2], insertion[0]), s))

    W_num = edge_number(W)
    V_num = node_number(V)

    # print('V: ', V)
    # print('VPOS Finale: ', Vpos)

    # print('++++++++++++++')

    return V, W


# Edge number
# Questa funzione prende in input la lista degli archi, dove ogni arco è rappresentato come coppia ((1,A),(2,B)) e
# restituisce la lista di archi rappresentati solo come coppie numeriche (1,2).
# Questa funzione è utilizzata all'interno dell'insertion repair.
# crea una lista dove gli archi sono riportati solo come coppia di numeri
def edge_number(W):
    W_number = []

    for i in range(len(W)):
        arc = W[i]
        a0 = arc[0]
        a1 = arc[1]
        W_number.append((a0[0], a1[0]))

    return W_number


# Node number
# Questa funzione prende in input una lista di coppie (id, label) corrispondenti ad ogni nodo del grafo.
# Restituisce una lista contenente solo il numero di ogni nodo.

# funzione che, data in input una lista di coppie (numero, label) corrispendenti ad ogni nodo del grafo,
# restituisce una lista con solo i numeri di ogni nodo
def node_number(V):
    V_number = []
    for i in range(len(V)):
        nod = V[i]
        V_number.append(nod[0])

    return V_number


# Is Path
# Funzione booleana per controllare se, dati due nodi a e b in input, esista un cammino che li collega.
# Viene utilizzata durante l'insertion repair per gestire i parallelismi.
# funzione booleana per verificare se tra due nodi dati in input esiste un cammino che li collega
def is_path(a, b, W, V):
    flag = False
    if (a, b) in W:
        flag = True
        return flag
    else:
        for c in range(len(V)):
            e = V[c]
            if (a, e[0]) in W:
                flag = is_path(e[0], b, W, V)
            else:
                continue

    return flag


# Aggiorna label
# Questa funzione, invocata al termine del repairing, aggiorna le label dei nodi in base a quanto contenuto nel mapping.
# aggiorna le label dei nodi in base al mapping
def update_label(W, map, V):
    W1 = []
    V1 = []

    for i in range(len(W)):
        arc = W[i]
        a0 = arc[0]
        a1 = arc[1]
        for j in range(len(map)):
            e = map[j]
            if a0 == (e[2], e[0]):
                for k in range(len(map)):
                    f = map[k]
                    if a1 == (f[2], f[0]):
                        W1.append(((e[1], e[0]), (f[1], f[0])))

    for i1 in range(len(V)):
        node = V[i1]
        for j1 in range(len(map)):
            e = map[j1]
            if node == (e[2], e[0]):
                V1.append((e[1], e[0]))

    return W1, V1


# SAVE FILE
"""
def save_g_file(V, W, path, time, sort_labels):
    with open(path, 'w') as f:
        f.write("# Execution Time: {0:.3f} s\n".format(time))
        # f.write("# Deleted Activities: {0}\n".format(D))
        # f.write("# Inserted Activities: {0}\n".format(I))
        for n in V:
            f.write("v {0} {1}\n".format(n[0], n[1]))
        f.write("\n")
        if sort_labels:
            W.sort()
        for e in W:
            f.write("e {0} {1} {2}__{3}\n".format(e[0][0], e[1][0], e[0][1], e[1][1]))
"""


def save_g_final(V, W, path, sort_labels):
    with open(path, 'a') as f:
        f.write("XP \n")
        for n in V:
            f.write("v {0} {1}\n".format(n[0], n[1]))
        if sort_labels:
            W.sort()
        for e in W:
            f.write("e {0} {1} {2}__{3}\n".format(e[0][0], e[1][0], e[0][1], e[1][1]))
        f.write("\n")
        f.close()


def save_csv(path, aligned, model_moves, num):
    with open(path, 'a') as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow([num, aligned, model_moves])
        f.close()


def BIG(filename=None, sort_labels=False, initial_marking=None, final_marking=None, net=None):
    init_time = time()
    #g_name = filename.split('/')[-1].split('.')[0]
    g_name = filename.split('xes\\')[-1].split('.')[0]
    print("BIG algorithm started...")
    log_path = join(INPUT_XES_PATH, filename)
    streaming_ev_object = xes_importer.apply(log_path, variant=xes_importer.Variants.XES_TRACE_STREAM)

    print('Finding causal relationships..')
    cr = find_causal_relationships(net)

    n = 0

    Aligned = []
    L = []
    Align = []
    L1 = []
    A = []
    A1 = []
    map = []
    compliant = []
    ins = []

    print(f'Processing traces...')
    for index, trace in enumerate(streaming_ev_object):
        trace_start_time = time()
        n += 1
        # pos trace
        Aligned, A = pick_aligned_trace(trace, net, initial_marking, final_marking)
        Align = Aligned[0]
        A1 = A[0]
        # print('Aligned to model')
        # print(Align)
        # print('with invisible moves')
        # print(A1)
        map, ins = mapping(Align, A1)

        # compliant mi serve per generare l'ig in base al modello (rimuove dalla traccia allineata i move)
        compliant = compliant_trace(Align)
        effettiva = compliant_trace(A1)

        # print(compliant)
        # print('Effettiva: ', effettiva)

        # print("map: ", map)
        # print("ins: ", ins)

        d = []

        num = trace.attributes.get('concept:name')
        id = trace.attributes.get('variant-index')

        # estrazione dell' IG su cui poi devo fare riparazione
        V, W = extract_instance_graph(compliant, cr)
        # print('V')
        # print(V)
        # print('W')
        # print(W)

        # ottengo liste di nodi e archi contenenti solo i valori numerici dei nodi
        V_n = node_number(V)
        W_n = edge_number(W)

        for element in map:  # crea lista dei nodi da cancellare
            if element[1] == 0:
                d.append(element)

        # Vpos lista dei nodi utilizzata per repair. inizialmente si rimuovono da essa i nodi relativi a deletion.
        # In seguito viene passata in input alla funzione di insertion repair e ogni attività viene inserita
        # all'interno di Vpos
        # nella posizione di inserimento. così facendo vpos sarà sempre aggiornata a ogni inserimento.
        # Al termine delle insertion avrò la mia lista dei nodi aggiornata.

        Vpos = []
        for node in V:
            Vpos.append(node)

        for el in map:
            if el[1] == 0:
                Vpos.remove((el[2], el[0]))

        # print('Vpos = ', Vpos)

        # print('INSERTION REPAIR')

        for insertion in ins:
            V, W = ins_repair(V, W, map, insertion, V_n, ins, Vpos)

        # print('W repaired: ', W)

        # print('DELETION REPAIR')
        # print(d)

        for deletion in d:
            V, W = del_repair(V, W, map, deletion)

        # aggiorna le label dei nodi in base a quanto contenuto nel mapping

        W_new, V_new = update_label(W, map, V)

        # riordina le liste di nodi e archi in base agli id
        V_new.sort()
        W_new.sort()

        # print('V finale: ',V_new)
        # print('W finale: ',W_new)

        # dot = viewInstanceGraph(V,W)
        # dot.save()
        save_g_final(V_new, W_new,
                     join(INPUT_G_PATH, f'{g_name}_instance_graphs.g'), sort_labels)
        print(f'\tTrace {index} processed in  {time() - trace_start_time} seconds..')

    print(f'BIG algorithm completed in {time() - init_time} seconds..')


def process(log_name, net_path):
    clean_directories()
    if log_name is None:
        log_name = args.xes_name

    log_filename = join(INPUT_PATH, 'xes', log_name)
    if not exists(log_filename):
        raise FileNotFoundError(f'File: {log_filename} not found')

    print(f'Processing log: {log_filename}..')
    log, log_filename = check_start_end_traces(log_filename)
    net, initial_marking, final_marking = extract_process_model(log_filename, log, net_path=net_path, threshold=0.4)
    BIG(filename=log_filename, net=net, initial_marking=initial_marking, final_marking=final_marking)


if __name__ == '__main__':
    try:
        log_name = sys.argv[1]
    except (Exception, ):
        log_name = None
    try:
        net_path = sys.argv[2]
    except (Exception,):
        net_path = None
    process(log_name, net_path)
