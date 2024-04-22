import torch
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset
from torch.nn import Linear, Conv1d
from torch_geometric.nn import SAGEConv, SortAggregation
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
from time import time
import random
from shutil import copyfile
from os import makedirs
from os.path import join, isfile, splitext

from cli_functions import select_dataset_paths, select_range_and_split
from config import CK_BEST_TRAIN_PATH, OUTPUT_DS_PATH, CM_PATH, \
    CHECKPOINT_RETE_PATH, CSV_PATH, IMG_BEST_TEST_PATH, IMG_BEST_TRAIN_PATH, \
    LOG_PATH, BASE_PATH, CK_BEST_TEST_PATH, F1_SCORE_PATH
import config

plt.ioff()
args = config.load()


"""classe Pytorch per la creazione o lettura del dataset per allenare la rete.

il file che contiene il dataset sarà creato all'interno della cartella process, questa cartella viene creata all'interno della cartella dataset precedentemente creata nel drive. Il nome del file creato sarà quello passato dal metodo processed_file_names della classe.

Quando la classe viene richiamata prima controlla se all'interno della cartella process è presente il file con il nome passato nel metodo processed_file_names, se presente andrà a leggere il file, se non presente andrà ad attivare il metodo process per la creazione di un nuovo dataset e gli darà il nome passato tramite il metodo processed_file_names.

In questo script viene utilizzato per la lettura del dataset, si consiglia di controllare che il dataset sian presente all'interno della cartella process e inserire all'interno del metodo processed_file_names il nome del dataset da voler utilizzare per allenare la rete.
"""


class TraceDataset(InMemoryDataset):

    def __init__(self, transform=None, pre_transform=None):
        super(TraceDataset, self).__init__(args.data_dir, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['complete_par.pt']

    def process(self):
        print('Dataset preso correttamente.')


"""classe Pytorch per la creazione o lettura del dataset per allenare la rete.

il file che contiene il dataset sarà creato all'interno della cartella process, questa cartella viene creata all'interno della cartella dataset precedentemente creata nella cartell Output.
./Output/dataset/processed

Il nome del file creato sarà quello passato dal metodo processed_file_names creato in creazione_dataset_next_activity.py

Quando la classe viene richiamata prima controlla se all'interno della cartella process è presente il file con il nome passato nel metodo processed_file_names, se presente andrà a leggere il file, se non presente andrà ad attivare il metodo process per la creazione di un nuovo dataset e gli darà il nome passato tramite il metodo processed_file_names.

In questo script viene utilizzato per la lettura del dataset, si consiglia di controllare che il dataset sia presente all'interno della cartella process e inserire all'interno del metodo processed_file_names il nome del dataset da voler utilizzare per allenare la rete.

Il nome del file è preimpostato  a 'complete'

Un modulo PyTorch che implementa il modello SortPool per la classificazione di grafi.

SortPool è una tecnica di pooling per grafi che ordina i nodi in base a una caratteristica (ad esempio, le loro
embedding) prima di selezionare i primi k nodi per formare un grafo fisso di dimensione k. Questo permette di applicare
con successo metodi di convoluzione standard su grafi di dimensione variabile. La classe utilizza convoluzioni spaziali
sui grafi (`SAGEConv`) e convoluzione 1D standard (`Conv1d`) per l'elaborazione dei dati del grafo.

Parameters
----------
dataset : torch_geometric.data.Dataset
    Il dataset contenente i grafi da classificare. Deve fornire `num_features` e `num_classes`.
num_layers : int
    Il numero di layer di convoluzione sui grafi (`SAGEConv`) da utilizzare.
hidden : int
    Il numero di feature nascoste in ciascun layer di convoluzione.
k : int
    Il numero di nodi da mantenere dopo l'operazione di SortPool. NOTA: tipicamente impostato tra 2 e 3 per questo modello.

Methods
-------
reset_parameters()
    Re-inizializza i parametri di tutti i moduli (layer) del modello.
forward(data, k)
    Definisce il passaggio in avanti del modello. Utilizza `SortAggregation` per eseguire il pooling basato su ordinamento,
    seguito da una serie di convoluzioni e trasformazioni lineari.

Attributes
----------
conv1 : torch.nn.Module
    Il primo layer di convoluzione sui grafi.
convs : torch.nn.ModuleList
    Una lista di layer di convoluzione sui grafi aggiuntivi.
conv1d : torch.nn.Conv1d
    Un layer di convoluzione 1D per elaborare le caratteristiche aggregati.
lin1 : torch.nn.Linear
    Un layer lineare per trasformazioni aggiuntive post-convoluzione.
lin2 : torch.nn.Linear
    Il layer lineare finale per la classificazione.

"""


class SortPool(torch.nn.Module):

    # definisce i layer
    def __init__(self, dataset, num_layers, hidden, k):
        super(SortPool, self).__init__()
        self.k = k  # NOTA: A NOI SERVE 2-3!!!!!!!!!!!!!!
        self.conv1 = SAGEConv(dataset.num_features, hidden)  # sageconv layer di convoluzione spaziale sui grafi grafi
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))

        kernel_size = num_layers

        self.conv1d = Conv1d(hidden, 32, kernel_size)  # conv1d layer di convoluzione 1
        self.lin1 = Linear(32 * (self.k - kernel_size + 1),
                           hidden)  # linear layer che applica una trasformazione lineare
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.conv1d.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    # collega i layer
    def forward(self, data, k):
        sort_aggr = SortAggregation(k)

        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))  # relu funzione di attivazione del layer
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = sort_aggr(x, batch)  # layer di pooling
        x = x.view(len(x), self.k, -1).permute(0, 2, 1)
        # modifica della struttura del vettore per poterlo passare al layer conv1d (devono avere n°nodi=k)
        x = F.relu(self.conv1d(x))
        x = x.view(len(x), -1)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=dropout, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


# dizionario: chiave=target, valore=indice progressivo (da 0)
def dict_target():
    target_std = {}
    i = 0
    input = open(join(OUTPUT_DS_PATH, 'target_std.txt'), "r")
    for lines in input.readlines():
        lines = lines[:-1]
        target_std[lines] = i
        i = i + 1
    input.close()
    # print('target std', target_std)

    target_par = {}
    i = 0

    input = open(join(OUTPUT_DS_PATH, 'target_par.txt'), "r")
    for lines in input.readlines():
        lines = lines[:-1]
        target_par[lines] = i
        i = i + 1
    input.close()
    # print('target par', target_par)
    return target_std, target_par


def split_target(G, per):
    dict = {}
    tar_std, _ = dict_target()  # dizionario: chiave=attività (target), valore=codice progressivo
    for x in tar_std.keys():  # crea coppie chiave (codice progressivo) - lista vuota per ogni target
        dict[tar_std[x]] = []

    for x in G:
        dict[int(x.y[0])].append(x)
        # aggiunge alla lista (valore) nel dizionario alla chiave (codice progressivo) il Data che descrive il grafo
    train = []
    test = []
    for x in dict.keys():
        a = []
        a.extend(dict[x])  # inserisce la lista del dizionario con chiave x alla lista a
        # split effettivo
        l = int(len(a) / 100 * per)
        atr = a[:l]
        ate = a[l:]
        train.extend(atr.copy())
        test.extend(ate.copy())
    return train, test


"""metodi per salvare le matrici di confusione, salva sia nel formato di tensore pytorch sia su file in formato txt"""


def print_confusion_matrix_file(cmt, epoch, keyword):
    torch.save(cmt, join(CM_PATH, f'cm_{keyword}_{epoch}.pt'))
    output = open(join(CM_PATH, f'cm_{keyword}_{epoch}.txt'), "w")
    for x in cmt:
        for y in x:
            output.write(str(y.item()))
            output.write("\t")
        output.write("\n")
    output.close()


# Funzione per la selezione di quali prefix-number utilizzare per allenare la rete
# e scelta della percentuale di train/test
def select_parameters():
    prefix_occurrences = {}
    print(G.indices)

    for data in G:
        # Calcola il numero di prefissi
        num_nodes = torch.unique(data.edge_index).size(0)

        # Aggiorna il conteggio delle occorrenze per questo num_edges
        if num_nodes in prefix_occurrences:
            prefix_occurrences[num_nodes] += 1
        else:
            prefix_occurrences[num_nodes] = 1

    # Stampa il risultato
    # for num_edges, occurrences in prefix_occurrences.items():
    #     print(f"Numero di archi: {num_edges}, Occorrenze: {occurrences}")

    return prefix_occurrences


def filter_graphs_by_prefixes(G, min_prefissi, max_prefissi):
    indici_validi = []
    in_tot = 0
    data_list = []

    for indice, data in enumerate(G):
        num_nodes = torch.unique(data.edge_index).size(0)
        in_tot += 1

        if min_prefissi <= num_nodes <= max_prefissi:
            indici_validi.append(indice)

    for indice, data in enumerate(G):
        if indice in indici_validi:
            data_filtered = Data(x=data.x, edge_index=data.edge_index, y=data.y, y_par=data.y_par)
            data_list.append(data_filtered)

    return data_list


# Calcolo di accuracy e F1 score per ogni gruppo di prefissi
def calculate_metrics(df):
    df['y_true'] = df['y_true'].astype(int)
    df['y_pred'] = df['y_pred'].astype(int)
    accuracy = accuracy_score(df['y_true'], df['y_pred'])
    f1 = f1_score(df['y_true'], df['y_pred'], average='weighted')
    return pd.Series({'Accuracy': accuracy, 'F1_Score': f1})


def train_net(k):  # funzione di addestramento della rete
    global model
    model.train(True)  # il true serve per indicare che si devono addestrare i parametri del modello
    running_loss = 0.0
    running_corrects = 0
    all_preds = torch.tensor([])
    ystack_std = torch.tensor([])

    for data in train_loader:  # itera i batch all'interno del dataset
        global optimizer
        optimizer.zero_grad()  # imposta il gradiente a zero
        out = model(data, k)  # risultati di predizione della rete
        loss = criterion(out, data.y)  # calcola la loss
        pred = out.argmax(dim=1)
        # calcola la predizione della rete prendendo quella con la probabilità maggiore.
        # Restituisce il valore progressivo dell'attività predetta
        all_preds = torch.cat((all_preds, pred), dim=0)  # popolo un vettore con tutte le predizioni della rete
        ystack_std = torch.cat((ystack_std, data.y), dim=0)  # popolo un vettore con le activity target (ground True)
        loss.backward()  # derivo il gradiente
        optimizer.step()  # aggiorna i parametri della rete in base al gradiente
        running_loss += loss.item() * data.num_graphs
        # serve per calcolare la loss media dell'epoca di addestramento dato che si calcola la loss per ogni batch
        running_corrects += int(
            (pred == data.y).sum())  # conta quante predizioni sono state corrette per poter poi calcolare l'accuracy
    stacked = torch.stack((ystack_std, all_preds),
                          dim=1)  # creo un vettore di coppie contenenti la predizione della rete e il valore esatto
    cmt = torch.zeros(G.num_classes, G.num_classes,
                      dtype=torch.int64)  # inizializzo a zero la matrice in cui salvo la metrice diconfusione
    for p in stacked:  # popolo la matrice di confusione
        tl, pl = p.tolist()
        cmt[int(tl), int(pl)] = cmt[int(tl), int(pl)] + 1
    return running_loss, running_corrects, cmt


# funzione per il test della rete
def test_net(loader, results_df, epoch, k):
    model.train(False)  # passando il false dico al programma di non addestrare il modello
    running_loss = 0.0
    running_corrects = 0

    all_preds = torch.tensor([])
    ystack_std = torch.tensor([])

    for data in loader:  # itera i batch all'interno del dataset
        out = model(data, k)  # risultati di predizione della rete
        loss = criterion(out, data.y)  # calcola la loss
        pred = out.argmax(dim=1)  # calcola la predizione della rete prendendo quella con la probabilità maggiore
        all_preds = torch.cat((all_preds, pred), dim=0)  # popolo un vettore con tutte le predizioni della rete
        ystack_std = torch.cat((ystack_std, data.y), dim=0)  # popolo un vettore con le activity target
        # serve per calcolare la loss media dell'epoca di addestramento dato che si calcola la loss per ogni batch
        running_loss += loss.item() * data.num_graphs
        running_corrects += int(
            (pred == data.y).sum())  # conta quante predizioni sono state corrette per poter poi calcolare l'accuracy

        unique_edge = torch.unique(data.edge_index)
        num_prefix = unique_edge.shape[0]

        batch_results = pd.DataFrame({'Num_Prefix': [num_prefix] * len(pred),
                                      'y_pred': int(pred.cpu()),
                                      'y_true': int(data.y.cpu())})
        results_df = pd.concat([results_df, batch_results], ignore_index=True)

    # occurrences_group = results_df.groupby('Num_Prefix').size().reset_index(name='Occurrences')
    grouped_results = results_df.groupby('Num_Prefix').apply(calculate_metrics)
    grouped_results = grouped_results.sort_values(by='Num_Prefix')
    accuracy_all = accuracy_score(ystack_std, all_preds)
    f1_all = f1_score(ystack_std, all_preds, average='weighted')

    # Concatena le informazioni in una stringa
    file_content = grouped_results.to_string(
        index=False) + f'\n\nAccuracy All: {accuracy_all:.4f}\nF1 Score All: {f1_all:.4f}\n'

    # Calcola le occorrenze nella colonna 'Num_Prefix'
    occurrences_count = results_df['Num_Prefix'].value_counts()

    # Concatena le informazioni nel file_content
    file_content += f'\n\nOccurrences Count:\n{occurrences_count.to_string()}'

    # Salva il contenuto nel file di testo
    with open(join(CSV_PATH, f'{epoch}_CSV.txt'), 'w') as file:
        file.write(file_content)

    # Grafico
    plt.figure(figsize=(10, 6))

    # Traccia Accuratezza e Punteggio F1
    plt.plot(grouped_results.index, grouped_results['Accuracy'], label='Accuracy', marker='o')
    plt.plot(grouped_results.index, grouped_results['F1_Score'], label='F1 Score', marker='x')

    # Traccia 'samples' (Occorrenze)
    # plt.plot(grouped_results.index, occurrences_group['Occurrences'], label='Samples', marker='^')

    # Imposta i tick sull'asse delle x
    plt.xticks(range(results_df['Num_Prefix'].min(), results_df['Num_Prefix'].max() + 1,
                     1))  # Imposta i tick da 2 a 20, con incrementi di 0.5

    # Aggiungi testo con i valori di accuracy_all e f1_all
    plt.text(0.5, 0.9, f'Accuracy All: {accuracy_all:.4f}', transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top')
    plt.text(0.5, 0.85, f'F1 Score All: {f1_all:.4f}', transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top')

    # Aggiungi etichette e legenda
    plt.xlabel('Numero di Prefissi')
    plt.ylabel('Metriche')
    plt.title('Andamento delle Metriche in Funzione dei Prefissi')
    plt.legend()

    # Imposta lo sfondo bianco e rimuovi la griglia
    plt.gca().set_facecolor('white')
    plt.grid(False)

    plt.savefig(join(F1_SCORE_PATH, f'{epoch}_PrefixF1Score.png'))  # salva l'immagini della matrice di confusione

    stacked = torch.stack((ystack_std, all_preds),
                          dim=1)  # creo un vettore di coppie contenenti la predizione della rete e il valore esatto
    cmt = torch.zeros(G.num_classes, G.num_classes,
                      dtype=torch.int64)  # inizializzo a zero la matrice in cui salvo la metrice diconfusione
    for p in stacked:  # popolo la matrice di confusione
        tl, pl = p.tolist()
        cmt[int(tl), int(pl)] = cmt[int(tl), int(pl)] + 1

    return running_loss, running_corrects, cmt


# funzione che serve per creare l'immagine della matrice di confusione
def plot_confusion_matrix(cm, title, normalize=False, cmap=plt.cm.Blues):
    # input=open('target.txt')
    input = open(join(OUTPUT_DS_PATH, 'target_std.txt'))
    classes = []

    for lines in input.readlines():
        lines = lines[:-1]
        classes.append(lines)
    input.close()

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    else:
        # print('Confusion matrix, without normalization')
        pass

    # print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# funzione che serve per far ripartire l'addestramento della rete da un checkpoint salvato
def resume_from_checkpoint(path_to_checkpoint):
    if isfile(path_to_checkpoint):
        # Caricamento del checkpoint
        checkpoint = torch.load(path_to_checkpoint)
        # Ripristino dello stato del sistema
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt'])
        print("Caricato il checkpoint '{}' (epoca {})".format(path_to_checkpoint, checkpoint['epoch']))
    else:
        start_epoch = 0
    return start_epoch


# serve per riprendere il valore della miglior loss di test degli addestramenti precedenti
def resume_best_loss_test(path_to_checkpoint):
    if isfile(path_to_checkpoint):
        b = torch.load(path_to_checkpoint)
        best_loss_test = b['loss']
        return best_loss_test
    else:
        best_loss_test = 0
        return best_loss_test


# serve per riprendere il valore della miglior loss di train degli addestramenti precedenti
def resume_best_loss_train(path_to_checkpoint):
    if isfile(path_to_checkpoint):
        b = torch.load(path_to_checkpoint)
        best_loss_train = b['loss']
        return best_loss_train
    else:
        best_loss_train = 0
        return best_loss_train


def run_epochs(num_layers, hidden, k, lr):
    # apre un file per il salvataggio del log della rete, per ogni epoca salva i valori di
    # accuracy edi loss per il train e il test
    # Stampa le informazioni in forma tabellare
    print('+---------------+------------+')
    print('| {:<13} | {:<10} |'.format('Num Layers', num_layers))
    print('| {:<13} | {:<10} |'.format('Hidden', hidden))
    print('| {:<13} | {:<10} |'.format('k', k))
    print('| {:<13} | {:<10} |'.format('Learning Rate', lr))
    print('+---------------+------------+')

    # modificare con il checkpoint da cui far partire la rete e modificare con il path in cui si trova
    start_epoch = resume_from_checkpoint(join(CHECKPOINT_RETE_PATH, 'checkpoint_50.pth.tar'))
    # NOTA QUESTO 50 è FITTIZIO, BISOGNA VEDERE IL FILE SALVATO PER IL CORRETTO NOME
    best_loss_test = resume_best_loss_test(join(CK_BEST_TEST_PATH, 'best_model.pth.tar'))
    best_loss_train = resume_best_loss_train(join(CK_BEST_TRAIN_PATH, 'best_model.pth.tar'))

    file_log = open(join(LOG_PATH,
                         f'log_split_target_{percentuale_split}_{100 - percentuale_split}_epoch_{start_epoch}_{start_epoch + delta_epoch - 1}.txt'),
                    'w')
    global model
    global optimizer
    model = SortPool(dataset=G, num_layers=num_layers, hidden=hidden, k=k)  # definizione del modello della rete
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # definizione della variabile dove vengono salvati i parametri di ottimizzazione della rete
    start_time = time()

    for epoch in range(start_epoch, start_epoch + delta_epoch):  # ciclo delle epoche di addestramento
        train_loss, train_acc, cmt_train = train_net(k=k)  # avvia il training della rete

        # Path dove salvo le metriche di valutazione F1 in funzione dei prefissi
        test_loss, test_acc, cmt_test = test_net(test_loader, results_df, epoch, k)
        # avvia il test della rete

        lossTr.append(train_loss / len(train_loader.dataset))
        # popola delle liste che contengono i valori di loss e accuracy della rete peril train e per il test
        lossTe.append(test_loss / len(test_loader.dataset))
        accTr.append(train_acc / len(train_loader.dataset))
        accTe.append(test_acc / len(test_loader.dataset))

        print_confusion_matrix_file(cmt_train, epoch, 'train')  # salva le matrici di confusione
        print_confusion_matrix_file(cmt_test, epoch, 'test')
        # plt.figure(figsize=(15,15))       #crea la figura in cui salvare la matrice di confusione nel formato png
        plot_confusion_matrix(cmt_test, "Confusion matrix TEST epoch : " + str(epoch))
        # modificare con il path dove si vogliono salvare le immagini della matrice di confusione

        plt.savefig(join(CM_PATH, f'confusion_matrix_split_target_{percentuale_split}_{100 - percentuale_split}_epoch_{epoch}test.png'))
        # salva l'immagini della matrice di confusione
        plt.figure(figsize=(15, 15))
        plot_confusion_matrix(cmt_train, "Confusion matrix TRAIN epoch : " + str(epoch))
        # modificare con il path dove si vogliono salvare le immagini della matrice di confusione

        plt.savefig(join(CM_PATH, f'confusion_matrix_split_target_{percentuale_split}_{100 - percentuale_split}_epoch_{epoch}train.png'))
        # print(f'Epoch: {epoch:03d}, Train Acc: {(train_acc/len(train_loader.dataset)):.4f},
        # Test Acc: {(test_acc/len(test_loader.dataset)):.4f}, Train Loss: {(train_loss/len(train_loader.dataset)):.4f},
        # Test Loss: {(test_loss/len(test_loader.dataset)):.4f} ')
        file_log.write(
            f'Epoch: {epoch:03d}, Train Acc: {(train_acc / len(train_loader.dataset)):.4f},'
            f'Test Acc: {(test_acc / len(test_loader.dataset)):.4f},'
            f'Train Loss: {(train_loss / len(train_loader.dataset)):.4f},'
            f'Test Loss: {(test_loss / len(test_loader.dataset)):.4f} \n')
        # scrive il file di log

        # TEST: SE LOSS MIGLIORA
        if test_loss < best_loss_test or epoch == 0:  # condizione per decidere la miglior loss di test
            best_loss_test = test_loss
            print(
                f'Epoch: {epoch:03d}, Train Acc: {(train_acc / len(train_loader.dataset)):.4f},'
                f'Test Acc: {(test_acc / len(test_loader.dataset)):.4f},'
                f'Train Loss: {(train_loss / len(train_loader.dataset)):.4f},'
                f'Test Loss: {(test_loss / len(test_loader.dataset)):.4f}   **** BEST EPOCH TEST ****')

            plt.figure(figsize=(15, 15))
            plot_confusion_matrix(cmt_test, "Confusion matrix BEST TEST epoch : " + str(epoch))
            # modificare con il path dove si vogliono salvare le immagini della matrice di confusione
            plt.savefig(join(IMG_BEST_TEST_PATH, 'test.png'))
            plt.figure(figsize=(15, 15))
            plot_confusion_matrix(cmt_train, "Confusion matrix train epoch: " + str(epoch))
            plt.savefig(join(IMG_BEST_TEST_PATH, 'train.png'))
            state = {'epoch': epoch, 'loss': test_loss, 'state_dict': model.state_dict(),
                     'opt': optimizer.state_dict()}
            torch.save(state, join(CK_BEST_TEST_PATH, 'best_model.pth.tar'))  # salva info utili sulla best epoch
            best_loss_test = test_loss
            triggertimes = 0

        else:
            triggertimes += 1

            print(
                f'Epoch: {epoch:03d}, Train Acc: {(train_acc / len(train_loader.dataset)):.4f},'
                f'Test Acc: {(test_acc / len(test_loader.dataset)):.4f},'
                f'Train Loss: {(train_loss / len(train_loader.dataset)):.4f},'
                f'Test Loss: {(test_loss / len(test_loader.dataset)):.4f} ')

        # TRAIN: SE LOSS MIGLIORA
        if train_loss < best_loss_train or epoch == 0:  # condizione per decidere la miglior loss di test
            best_loss_train = train_loss
            print(
                f'Epoch: {epoch:03d}, Train Acc: {(train_acc / len(train_loader.dataset)):.4f},'
                f'Test Acc: {(test_acc / len(test_loader.dataset)):.4f},'
                f'Train Loss: {(train_loss / len(train_loader.dataset)):.4f},'
                f'Test Loss: {(test_loss / len(test_loader.dataset)):.4f}   **** BEST EPOCH TRAIN ****')

            plt.figure(figsize=(15, 15))
            plot_confusion_matrix(cmt_test, "Confusion matrix test epoch : " + str(epoch))
            # modificare con il path dove si vogliono salvare le immagini della matrice di confusione
            plt.savefig(join(IMG_BEST_TRAIN_PATH, 'test.png'))
            plt.figure(figsize=(15, 15))
            plot_confusion_matrix(cmt_train, "Confusion matrix BEST TRAIN epoch: " + str(epoch))
            plt.savefig(join(IMG_BEST_TRAIN_PATH, 'train.png'))
            state = {'epoch': epoch, 'loss': train_loss, 'state_dict': model.state_dict(),
                     'opt': optimizer.state_dict()}
            torch.save(state, join(CK_BEST_TRAIN_PATH, 'best_model.pth.tar'))
            # salva info utili sulla best epoch
            best_loss_train = train_loss

        else:
            print(
                f'Epoch: {epoch:03d}, Train Acc: {(train_acc / len(train_loader.dataset)):.4f},'
                f'Test Acc: {(test_acc / len(test_loader.dataset)):.4f},'
                f'Train Loss: {(train_loss / len(train_loader.dataset)):.4f},'
                f'Test Loss: {(test_loss / len(test_loader.dataset)):.4f} ')

        # early stopping
        if triggertimes >= patience:
            print('Early stopping!\nBest loss = {}'.format(best_loss_test))
            break

        plt.close('all')

    file_log.close()
    end_time = time()
    time_sec = end_time - start_time
    hours = int(time_sec) // 3600
    minutes = int(time_sec % 3600) // 60
    seconds = int(time_sec % 60)
    if hours > 0:
        print(f"--- Time for training and testing --> {hours} hours, {minutes} minutes and {seconds} seconds ---")
    elif hours == 0 and minutes > 0:
        print(f"--- Time for training and testing --> {minutes} minutes and {seconds} seconds ---")
    else:
        print(f"--- Time for training and testing --> {seconds} seconds ---")

    state = {'epoch': start_epoch + delta_epoch, 'state_dict': model.state_dict(),
             'opt': optimizer.state_dict()}  # creo un dizionario per salvare il checkpoint della rete
    # modificare con il path dove si vogliono salvare i checkpoint
    torch.save(state, join(CHECKPOINT_RETE_PATH, f'{start_epoch + delta_epoch - 1}.pth.tar'))
    # salvo il checkpoint della rete


dataset_path, path_txt_par, path_txt_std = select_dataset_paths()  # Seleziona i percorsi dei file del dataset
# Verifica il tipo di file
_, file_extension_dataset = splitext(dataset_path)
_, file_extension_txt_par = splitext(path_txt_par)
_, file_extension_txt_std = splitext(path_txt_std)

target_path_dataset = join(OUTPUT_DS_PATH, 'processed', 'complete_par.pt')
target_path_txt_par = join(OUTPUT_DS_PATH, 'target_par.txt')
target_path_txt_std = join(OUTPUT_DS_PATH, 'target_std.txt')

# Verifica se l'estensione è '.pt'
if (file_extension_dataset == '.pt') & (file_extension_txt_par == '.txt') & (file_extension_txt_std == '.txt'):
    if not (dataset_path == target_path_dataset) & (path_txt_par == target_path_txt_par) & (
            path_txt_std == target_path_txt_std):
        try:
            # Copia i file
            copyfile(dataset_path, target_path_dataset)
            copyfile(path_txt_par, target_path_txt_par)
            copyfile(path_txt_std, target_path_txt_std)
            print("File copiati con successo.")
        except PermissionError:
            print("Errore di permesso: non è possibile scrivere nel percorso specificato.")
        except FileNotFoundError:
            print("Errore: la directory di destinazione non esiste.")
        except Exception as e:
            print(f"Si è verificato un errore: {e}")
    else:
        print(f"Files di default presi da {OUTPUT_DS_PATH}")
accTr = []  # variabili per salvare i valori di accuracy e loss per la creazione dei grafici
accTe = []
lossTr = []
lossTe = []
dropout = 0.1  # 0,5 #0.05
model = None
optimizer = None

# per = 67 # percentuale per lo split del dataset tra train e test NON MODIFICARE
delta_epoch = 100  # epoche di addestramento
# add early stopping
patience = 10
triggertimes = 0
# ************

# Imposto il seed
seed = 12
# Imposta il seed per PyTorch
torch.manual_seed(seed)
# Imposta il seed per altre librerie
np.random.seed(seed)
random.seed(seed)

# Inizializzazione del DataFrame
results_df = pd.DataFrame(columns=['Num_Prefix', 'y_pred', 'y_true'])
# Creo il dataset
G = TraceDataset()

min_prefissi_selezionato, max_prefissi_selezionato, percentuale_split, search_grid = select_range_and_split(
    select_parameters())
print("Valori selezionati:")
print(f"Min prefissi: {min_prefissi_selezionato}")
print(f"Max prefissi: {max_prefissi_selezionato}")
print(f"Percentuale di split train/test: {percentuale_split}%")
print(f"Search Grid: {'Attivata' if search_grid else 'Disattivata'}")

G_filtrato = filter_graphs_by_prefixes(G, min_prefissi_selezionato, max_prefissi_selezionato)

train, test = split_target(G_filtrato, percentuale_split)  # divisione del dataset tra train e test
random.shuffle(train)
random.shuffle(test)

train_loader = DataLoader(train, batch_size=args.batch_size)
# definizione delle variabili che conterranno in dataset di train o test per l'allenamento della rete
test_loader = DataLoader(test, batch_size=1)
# batch_size serve per l'allenamento della rete quando si hanno grafi di dimensione e struttura variabile
criterion = torch.nn.CrossEntropyLoss()  # definizione del criterio per l'allenamento della rete

# apre un file per il salvataggio del log della rete, per ogni epoca salva i valori di accuracy e
# di loss per il train ed il test

if not search_grid:
    run_epochs(args.num_layers, args.batch_size, args.k, args.learning_rate)
else:
    # Definizione dei parametri per la grid search
    k_values = [3, 5, 7, 30]
    num_layers_values = [2, 3, 5, 7]
    lr_values = [1e-2, 1e-3, 1e-4]

    # Ciclo di grid search
    for k in k_values:
        for num_layers in num_layers_values:
            for lr in lr_values:
                # Controllo per combinazioni non valide
                if k < num_layers:
                    continue

                # Creazione di una cartella unica per questa combinazione
                combination_folder = join(BASE_PATH, f"results_k{k}_layers{num_layers}_lr{lr}")
                makedirs(combination_folder, exist_ok=True)
                makedirs(join(combination_folder, 'Immagini', 'cm_epoch'), exist_ok=True)
                makedirs(join(combination_folder, 'Immagini', 'best_train'), exist_ok=True)
                makedirs(join(combination_folder, 'Immagini', 'best_test'), exist_ok=True)
                makedirs(join(combination_folder, 'best_test'), exist_ok=True)
                makedirs(join(combination_folder, 'best_train'), exist_ok=True)
                makedirs(join(combination_folder, 'log'), exist_ok=True)

                run_epochs(num_layers, args.batch_size, k, lr)
