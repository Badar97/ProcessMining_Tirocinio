import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch_geometric.data import InMemoryDataset, Data

from config import PREFIX_PATH, OUTPUT_DS_PATH, STATE_PATH
from dotG import get_g_dataframe

from os.path import join, exists, isdir
import string

file_name = None
file_id = None
g_dataframe = None
att_numerici = None
att_categorici = None
ids = None
prefix = None

"""classe Pytorch per la creazione o lettura del dataset per allenare la rete.

il file che contiene il dataset sarà creato all'interno della cartella process, questa certela
viene creata all'interno della cartella dataset precedentemente creata nel drive.
Il nome del file creato sarà quello passato dal metodo processed_file_names della classe.
Quando la classe viene richiamata prima controlla se all'interno della cartella process è presente il file con il nome
passato nel metodo processed_file_names, se presente andrà a leggere il file, se non presente andrà ad attivare il
metodo process per la creazione di un nuovo dataset e gli darà il nome passato tramite il metodo processed_file_names.
In questo script viene utilizzato per la creazione del dataset, si consiglia di controllare che non siano presenti
all'interno della cartella process dataset con il nome inserito all'interno del metodo processed_file_names. 
Per creare il dataset completo nel metodo process mettere listGraph = create_sub_graph()
Per creare il dataset con undersamplig nel metodo process mettere listGraph=undersamplig(create_sub_graph(), n) ->
 sostituire n con il numero di elementi da prendere per ogni tipologia di activity da predirre
"""


class TraceDataset(InMemoryDataset):

    def __init__(self, transform=None, pre_transform=None):
        super(TraceDataset, self).__init__(OUTPUT_DS_PATH, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        # return ['andreaa_bpi12w_par.pt']
        return [file_name + '_par.pt']

    def process(self):
        print('--------------------------------------------------------')
        print(f"OUTPUT_DS_PATH è impostato a: {OUTPUT_DS_PATH}")
        print(f"Il dataset verrà salvato in: {self.processed_paths[0]}")
        # a subgraph for each trace?
        # vedi commento 366
        # ritorna lista dei sottografi
        listGraph = create_sub_graph()

        def one_hot_encode(value, one_hot_dict):
            # La lunghezza del vettore One-Hot è il numero massimo nel dizionario
            one_hot_vector = [0] * max(one_hot_dict.values())
            if value in one_hot_dict:
                # Sottrai 1 dall'indice perché l'indicizzazione delle liste in Python inizia da 0
                one_hot_vector[one_hot_dict[value] - 1] = 1
            return one_hot_vector

        data_list = []

        attr_event = dict_attr(OUTPUT_DS_PATH, file="attributi.txt")  # ritorna dizionario di liste (one hot)

        target_std, target_par = dict_target()  # ritorna dizionari target_std e target_par con codice progressivo

        if 'event_name' in att_categorici:
            att_categorici.drop('event_name')
        for G in listGraph:  # ciclo per scorrere i sottografi
            x1 = []
            list_targ_par = []

            # draw(G)
            for i in G.nodes:  # ciclo che scorre i nodi **** altro punto in cui lavorare
                node_features = []
                node_features.extend(attr_event[G.nodes[i]['name_event']])
                node_attrs = G.nodes[i]

                # Aggiungi gli attributi numerici
                for attr in att_numerici:
                    if attr in node_attrs:
                        node_features.append(float(node_attrs[attr]))

                # # Aggiungi gli attributi categorici in formato One-Hot
                for attr in att_categorici:
                    if attr in node_attrs and attr in one_hot_dictionaries:
                        one_hot_vector = one_hot_encode(node_attrs[attr], one_hot_dictionaries[attr])
                        node_features.extend(one_hot_vector)

                x1.append(node_features)
                # print(node_features)

                """
                x1.append([*attr[G.nodes[i]['attrib1']],
                        *G.nodes[i]['attrib2'],
                        # il trick serve per questa che è già una lista e così le concateni tutte al volo
                        *[G.nodes[i]['attrib3']],
                        *[G.nodes[i]['attrib4']],
                        *[G.nodes[i]['attrib5']],
                        *[G.nodes[i]['attrib6']] ]
                        )
                        # aggiunge alla lista il one-hot-encoder (lista) associato all'attributo di quel nodo (activity)
                """

            """
            print(x1)
            print(len(x1))
            """
            x = torch.tensor(x1, dtype=torch.float)
            """
            print("a")
            print(x)
            """
            adj = nx.to_scipy_sparse_array(G)  # prende la matrice di adiacenza del sottografo
            adj = adj.tocoo()
            row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
            col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
            edge_index = torch.stack([row, col], dim=0)
            # one value as target for the subgraph double check if G.graph['target_std'] is the target value
            # #crea un vettore contenente gli archi del sottografo it's e.g. this: tensor([11]) so it's just the
            # target value !

            # y_std = torch.tensor([G.graph['target_std']])
            # assegna il valore numerico all'attività da
            # predirre per quel sottografo secondo logica std
            y_std = torch.tensor([target_std[G.graph[
                'target_std']]])
            # assegna il valore numerico all'attività da predirre per quel sottografo secondo logica std

            # CHANGED!!!
            caseid = G.graph['caseid']
            # NOT NEEDED
            y_par = torch.tensor([target_par[G.graph[
                'target_par']]])
            # assegna il valore numerico all'attività da predirre per quel sottografo secondo logica par

            # print(y_par)

            # x=tensore attributi nodi sottografo corrente,
            # edge_index=descrive collegamenti tra nodi,
            # y=tensore attività da predirre (etichetta)
            # y_par=tensore attività parallele predicibili (etichetta)

            # MIGHT NEED TO REMOVE Y_PAR HERE AS WELL
            # data = Data(x=x, edge_index=edge_index, y=y_std,y_par=y_par,idc=caseid)
            # inserisce queste informazioni nella struttura dati utilizzata da pytorch_geometric
            # per gli elementi del dataset
            data = Data(x=x, edge_index=edge_index, y=y_std,
                        y_par=y_par)
            # inserisce queste informazioni nella struttura dati utilizzata da pytorch_geometric
            # per gli elementi del dataset

            # print(len(edge_index[0]))

            # calcolo il numero dei prefissi:
            flat_tensor = edge_index.view(-1)
            max_value = int(torch.max(flat_tensor))
            if max_value not in prefix:
                prefix.append(max_value)
                with open(join(PREFIX_PATH, 'max_values.txt'), 'a') as file:
                    flat_tensor = edge_index.view(-1)
                    max_value = int(torch.max(flat_tensor)) + 1
                    max_value = 'Num prefix: ' + str(max_value)
                    # Scrivi il valore massimo nel file, seguito da un ritorno a capo
                    file.write(str(max_value) + '\n')

            # print('Num Prefix : ',max_value+1)

            data_list.append(data)  # crea una lista contenente gli elementi del dataset

        # print(data_list)
        data, slices = self.collate(data_list)

        yy = data.y.tolist()
        unique_elements, counts = np.unique(yy, return_counts=True)

        # Stampare gli elementi unici e le frequenze
        # for element, count in zip(unique_elements, counts):
        #     print(f"Label {element}: {count} volte")
        '''

        print(data.y)
        print('-------------------------------------')
        print(data.x)

        '''
        torch.save((data, slices), self.processed_paths[0])  # salva il dataset


# faccio un check di simboli non consentiti per il salvataggio di un file.txt, se presenti sostituisco con _
def sanitize_filename(filename, replacement="_"):
    # Caratteri consentiti nei nomi dei file
    valid_chars = set(string.ascii_letters + string.digits + "._-")

    # Sostituisci i caratteri non validi
    sanitized_filename = ''.join(char if char in valid_chars else replacement for char in filename)

    return sanitized_filename

    # ciclo tutte le variabili categoriche per creare il dizionario del OHE e salvo ogni variabile in un file.txt
    # for col_name in att_categorici:
    #     unique_values = g_dataframe[col_name].unique()
    #     value_to_index = {value: index for index, value in enumerate(unique_values) if value != ''}
    #     one_hot_dictionaries[col_name] = value_to_index
    #     col_name = sanitize_filename(col_name)

    # Salva anche i valori unici se necessario
    # with open(f"{OUTPUT_DS_PATH}/{col_name}_unique.txt", "w") as file:
    #     for value in unique_values:
    #        if value != '':
    #            file.write(str(value) + "\n")


def save_ids():
    input = open(join(OUTPUT_DS_PATH, f'{file_id}.g'), 'r')
    for lines in input.readlines():  # ciclo per la lettura del file riga per riga
        line = lines.split()
        if not line:
            pass

        elif line[0] == "v":
            ids.append(line[4])  # cosa vuole prendere di preciso qui? MR


# prende ogni grafo e lo salva come inverso solo se sono falsi tutti i controlli if
def create_graph():
    save_ids()
    ListGraph = []
    attributes = []  # creo una lista vuota per salvare le activity da andare a scrivere nel file

    s = 0

    def verifyGraph(graph):
        if nx.number_of_isolates(G) != 0:
            # print('error line {}'.format(line)) #controllo che il grafo non contiene nodi isolati
            #pass  # istruzione nulla
        #elif len(nx.get_node_attributes(G, "name_event")) != len(
                #G.nodes()):  # verifica se ci sono nodi senza l'attributo "name_event" nel grafo.
            # print('error line {}'.format(line))
            #pass
        #elif nx.number_connected_components(
                #G.to_undirected()) != 1:  # controllo che ci sia una sola componenti connesse
            # print('error line {}'.format(line))
            pass
        elif len(G.nodes()) < 3:  # controllo che se il grafo ha almeno tre nodi
            # print('error line {}'.format(line))
            pass
        else:
            ListGraph.append(G.reverse())  # salvo il grafo con gli archi invertiti

    # Trova gli indici delle righe che contengono 'XP'
    xp_indices = g_dataframe[g_dataframe['e_v'] == 'XP'].index

    # Itera su ogni gruppo di righe
    for start, end in zip(xp_indices, xp_indices[1:]):
        # Estrai il sottogruppo, escludendo l'indice 'end'
        sub_df = g_dataframe.iloc[start:end]

        for index, row in sub_df.iterrows():

            if row['e_v'] == 'XP':
                # Creo un nuovo grafo
                G = nx.DiGraph()

            if row['e_v'] == 'v':
                # Crea un nuovo nodo
                node_nr = int(float(row['node1']) - 1)
                node_attributes = {}
                node_attributes['idn'] = ids[s]

                # Aggiungi attributi dalle colonne a partire dalla quarta
                # colonna in poi 'e_v', 'node1', 'node2', 'name_event'
                for col_name in sub_df.columns[3:]:
                    # Gestisco l'activity singolarmente, per assicurarmi che venga gestita sempre come una stringa
                    if col_name == 'name_event':
                        value_activity = row[col_name]
                        node_attributes[col_name] = str(value_activity)

                        # salvo tutte le Activity, mi serve per dopo quando devo creare
                        # il relativo dizionario (vedi 'dictattr')
                        if value_activity not in attributes:
                            attributes.append(value_activity)
                    # Gestisco tutti gli altri attributi in modo parametrico, precedentemente ho fatto dei controlli
                    # se gli attributi sono categorici o numerici
                    else:
                        col_value = row[col_name]
                        if col_name in att_categorici:
                            node_attributes[col_name] = str(col_value)
                        elif col_name in att_numerici:
                            node_attributes[col_name] = float(col_value)
                        else:

                            try:
                                node_attributes[col_name] = float(col_value)
                                if col_name not in att_numerici:
                                    att_numerici.append(col_name)
                                    print(
                                        '***************  inserimento  ' + col_name + ' --> in att_numerici ********************')
                            except:
                                node_attributes[col_name] = col_value
                                if col_name not in att_categorici:
                                    att_categorici.append(col_name)
                                    print(
                                        '***************  inserimento  ' + col_name + ' --> in att_categorici ********************')

                # Aggiungi il nodo al grafo
                G.add_node(node_nr, **node_attributes)
                s += 1

            elif row['e_v'] == 'e':
                # Creo un arco
                G.add_edge(int(float(row['node1']) - 1), int(float(row['node2']) - 1))

        verifyGraph(G)

    output = open(join(OUTPUT_DS_PATH, 'attributi.txt'), 'w')
    for att in attributes:
        output.write(att + "\n")
    output.close()

    return ListGraph  # ritorno la lista dei grafi


"""**TARGET**:
* Controllare archi uscenti dal sottografo -> nodi collegati (*esterni al sottografo*)
* Per ogni nodo trovato -> verificare disponibilità tutti archi entranti (*devono già appartenere al sottografo i nodi all'altra estremità*)
"""


# qui toccherà giocarci quando vorremo estendere la caraterizzazione del nodo predetto in output
def define_target(graph, subgraph):
    reverse = graph.reverse()  # inverte le direzioni degli archi del grafo direzionato (completo)
    possible_targets = []  # lista che conterrà i neighbors dei nodi del sottografo
    subgraph_nodes = list(subgraph.nodes())  # lista dei nodi del sottografo
    for node in subgraph_nodes:  # per ogni nodo del sottografo, individua i neighbros e li inserisce in una lista
        possible_targets.extend(list(reverse.neighbors(node)))
    possible_targets = list(set(possible_targets) - set(subgraph_nodes))

    target = possible_targets.copy()
    for node in possible_targets:
        # per ogni possibile nodo target accerta che l'altro estremo degli archi entranti sia già un nodo del
        # sottografo, altrimenti lo elimina dai target
        for node_from, node_to in reverse.in_edges(node):
            if node_from not in subgraph_nodes:
                target.remove(node_to)
                break

    new_t = ''
    for i in range(0,
                   len(target)):  # sostituisce ogni nodo della lista target con la corrispettiva activity (attributo)
        targ_attr = graph.nodes[target[i]]['name_event']
        new_t = new_t + str(targ_attr) + ' '
    target = new_t[:-1]

    return target


#****************NEW_ADD_B****************#
import re

def read_subgraph(file_path):
    subgraphs = []
    with open(file_path, 'r') as file:
        current_graph = None
        node_pattern = re.compile(r'^(\d+) (\w+) (.*)')
        edge_pattern = re.compile(r'^(\d+) (\d+)$')
        for line in file:
            line = line.strip()
            if line == 'XP':
                # Se è presente la stringa "XP", crea un nuovo grafo
                if current_graph:
                    subgraphs.append(current_graph)
                current_graph = nx.Graph()
            else:
                if match := node_pattern.match(line):
                    node_id = int(match.group(1))
                    node_name = match.group(2)
                    if current_graph is None:
                        current_graph = nx.Graph()
                    current_graph.add_node(node_id, label=node_name)
                elif match := edge_pattern.match(line):
                    node1_id = int(match.group(1))
                    node2_id = int(match.group(2))
                    if current_graph is None:
                        current_graph = nx.Graph()
                    current_graph.add_edge(node1_id, node2_id)
        if current_graph:
            subgraphs.append(current_graph)
    return subgraphs

def add_status(cid, event_pos, subg_dir, Subgraph):
    c_id_ = cid.replace("_", "")
    file_path = join(subg_dir, f"{c_id_}_{event_pos}.g")
    if not exists(file_path):
        print(f"Il file {file_path} non esiste.")
        return None
    subg = read_subgraph(file_path)
    for subg_to_add in subg:
        if Subgraph is not None and subg_to_add is not None:
            Subgraph = nx.compose(subg_to_add, Subgraph)
        else:
            print('sottografo non preso')
            continue
    return Subgraph


# Metodo per la creazione dei sottografi nel formato di networkx, a partire dai grafi precedentemente creati.
# Alla fine viene creato un file che contiene il nome delle varie tipologie di activity presenti nel dataset come
# activity da predirre.
# Il file è salvato nel Drive google nella cartella dataset che deve essere creata precedentemente
def create_sub_graph():
    ListGraph = create_graph()
    target_std = []  # creo una lista vuota per salvare le activity da predirre
    target_par = []  # creo una lista vuota per salvare le activity parallele predicibili
    status = [] # creo una lista vuota per salvare lo stato

    ListSubGraph = []  # creo una lista vuota da popolare con i sottografi creati
    not_exist = False
    graph_num = 0
    for graph in ListGraph:  # ciclo per scorrere i grafi
        # print("graph")
        # print(graph)
        graph_num += 1
        print("trace "+str(graph_num)+" out of "+str(len(ListGraph)))
        SubGraph = nx.Graph(target_std="no", target_par='no')
        event_num = max(list(graph.nodes()))
        # crea sottografo con attributi target_std e target_par inizializzati a "no"
        for node in list(graph.nodes()):  # scorro i nodi del grafo corrente
            # si entra solo dopo i primi due nodi perchè sempre in input (mai predetti)
            if len(SubGraph.nodes()) > 1:  # changed min prefix length
                # changed the next line
                # controllo se il sottgrafo che si sta creando ha almeno due nodi all'interno

                # INSERIMENTO DEL TARGET FEATURE //DTL

                # INSERIMENTO DEL ACTIVITY DA PREDIRRE

                #   **********************************************************
                #   target = Days To Late
                #   i don't use it
                # #

                target_t1 = graph.nodes[node]['name_event']
                SubGraph.graph['target_std'] = graph.nodes[node]['name_event']
                # put the target value there for the correct node.
                # assegna come target_std al sottografo il nodo corrente
                # nodevar=node

                if SubGraph.graph['target_std'] not in target_std:
                    # inserisce l'activity solo se non è già inserita nella lista target_std
                    target_std.append(SubGraph.graph['target_std'])
                # *********************************************
                # NOT NEEDED! IT SHOULDN'T USE THIS IN TRAINING

                SubGraph.graph['target_par'] = define_target(graph.copy(), SubGraph)
                # assegna come target_par al sottografo il nodo corrente

                # CHANGED!!!
                SubGraph.graph['caseid'] = graph.nodes[node]['idn']
                if SubGraph.graph['target_par'] not in target_par:
                    # inserisce l'activity solo se non è già inserita nella lista target_par
                    target_par.append(SubGraph.graph['target_par'])

                #ListSubGraph.append(SubGraph.copy().to_undirected())
                # NOTA: la rete lavora sui grafo non diretti (questa cosa è modificabile in teoria)
                # if graph.nodes[node]['attribute']!="END":
                #     ListSubGraph.append(SubGraph.copy().to_undirected())
                # bij mij wordt er wel een voorspelling gemaakt bij de laatste prefix,
                # bij next activity is dat tot de een na laatste

                #*******************NEW_ADD_B**********************
                c_id = graph.nodes[node]['idn']
                st_path = join(STATE_PATH, 'stategraphs')
                if exists(st_path): #and isdir(st_path):
                    SubGraph.graph['status'] = add_status(c_id, event_num, st_path, SubGraph)
                    status.append(SubGraph.graph['status'])
                    #add_status(c_id, event_num, STATE_PATH, SubGraph)
                elif not not_exist:
                    print(f"La cartella {st_path} non esiste! Operazione per l'aggiunta dello stato non eseguita")
                    not_exist = True
                
                ############################################
                #nodi_grafo = list(SubGraph.nodes())
                #print("Nodi del grafo:", nodi_grafo)
                ListSubGraph.append(SubGraph.copy().to_undirected())

            attrs = graph.nodes[node]
            SubGraph.add_node(node, **attrs)

            for neig in graph.neighbors(node):  # aggiungo gli archi per quel nodo al sottografo
                SubGraph.add_edge(neig, node)

    # value_to_index = {value: index for index, value in enumerate(target_par) if value != ''}
    # one_hot_dictionaries['target_par'] = value_to_index

    # # Salva i dizionari in un file JSON
    # with open(f"{OUTPUT_DS_PATH}/one_hot_dictionaries.json", "w") as file:
    #     json.dump(one_hot_dictionaries, file)

    # categoricalAttribute.append('target_par')
    output = open(join(OUTPUT_DS_PATH, 'target_std.txt'), "w")  # apro e scrivo le activity da predire sul file
    # output = open(OUTPUT_DS_PATH+"/target_std_new.txt", "w")  #apro e scrivo le activity da predire sul file

    for item in target_std:  # scrive lista target sul file
        output.write(item + '\n')
    # output.write("END\n")
    output.close()

    # NOT NEEDED
    output = open(join(OUTPUT_DS_PATH, 'target_par.txt'), "w")  # apro e scrivo le activity da predire sul file
    # output = open(OUTPUT_DS_PATH+"/target_par_new.txt", "w")  #apro e scrivo le activity da predire sul file

    for item in target_par:  # scrive lista target sul file
        output.write(item + '\n')
    # output.write("END\n")
    output.close()

    """
    ## ADD_B_
    outpt = open(join(OUTPUT_DS_PATH, 'status.txt'), "w")
    for item in status:
        #outpt.write(item + '\n')
        nx.write_adjlist(item, outpt)
    outpt.close()
    """
    
    return ListSubGraph


# Dal file attributi.txt creato precedentemente si crea un dizionario delle activity che
# contiene per ogni activity il one hot vector corrispondente
# onehot encoding for the activities
def dict_attr(path, file):  # ="attributi.txt"):
    attr = []
    input = open(join(path, file), "r")
    # input = open(OUTPUT_DS_PATH+"/attributi_new.txt", "r")
    for lines in input.readlines():
        lines = lines[:-1]
        attr.append(lines)  # ricrea la lista degli attributi
    input.close()
    s1 = pd.Series(attr)  # crea una serie come valori le attività
    s2 = pd.get_dummies(s1)  # crea dataframe con tante colonne quante le attività e valori solo 0 e 1
    onedictfeat = {}
    # crea dizionario: chiave=chiave dataframe, valore = dizionario (chiave=colonna dataframe, valore=0 o 1)
    s3 = s2.to_dict()
    for a, b in s3.items():
        onedictfeat[a] = list(b.values())  # nuovo dizionario (valore=lista valori con stessa chiave)
    # print("onedictfeat",onedictfeat)
    return onedictfeat


"""Dal file target.txt creato in precedenza si crea un dizionario con il valore di predizione
assegnato ad ogni activity da predirre"""


# dizionario: chiave=target, valore=indice progressivo (da 0)
# NOT NEEDED, just used to determine nr of neurons in output layer.
def dict_target():
    target_std = {}
    i = 0
    input = open(join(OUTPUT_DS_PATH, 'target_std.txt'), "r")
    # input = open(OUTPUT_DS_PATH+"/target_std_new.txt", "r")
    for lines in input.readlines():
        lines = lines[:-1]
        target_std[lines] = i
        i = i + 1
    input.close()
    # print('target std',target_std)

    target_par = {}
    i = 0
    input = open(join(OUTPUT_DS_PATH, 'target_par.txt'), "r")
    # input = open(OUTPUT_DS_PATH+"/target_par_new.txt", "r")
    for lines in input.readlines():
        lines = lines[:-1]
        target_par[lines] = i
        i = i + 1
    input.close()
    # print('target par',target_par)
    return target_std, target_par


def draw(graph):
    pos = nx.circular_layout(graph)
    plt.figure(figsize=(7, 5))
    nodes = nx.draw_networkx_nodes(graph, pos, node_size=250)
    node_labels = nx.get_node_attributes(graph, 'attribute')
    labels = nx.draw_networkx_labels(graph, pos, labels=node_labels)
    edges = nx.draw_networkx_edges(graph, pos)

    plt.axis('off')
    plt.show()


if __name__ == "__main__":

    file_name, file_id = 'complete', '0'
    g_dataframe, att_numerici, att_categorici = get_g_dataframe()
    ids = []
    # Creazione e salvataggio dei dizionari per il One-Hot Encoding
    one_hot_dictionaries = {}
    prefix = []

    G = TraceDataset()
