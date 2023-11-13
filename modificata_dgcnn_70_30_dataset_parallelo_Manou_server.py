

"""
Dal file target.txt creato in precedenza si crea un dizionario con il valore di predizione assegnato ad ogni activity da predirre
"""


# dizionario: chiave=target, valore=indice progressivo (da 0)
# def dictarget():
#     target_std = {}
#     i = 0
#     input = open("./dataset/" + chosen_data + "/target_std.txt", "r")
#     for lines in input.readlines():
#         lines = lines[:-1]
#         target_std[lines] = i
#         i = i + 1
#     input.close()
#     #print('target std', target_std)

#     target_par = {}
#     i = 0
#     input = open("./dataset/" + chosen_data + "/target_par.txt", "r")
#     for lines in input.readlines():
#         lines = lines[:-1]
#         target_par[lines] = i
#         i = i + 1
#     input.close()
#     #print('target par', target_par)
#     return target_std, target_par


"""classe Pytorch per la creazione o lettura del dataset per allenare la rete.

il file che contiene il dataset sarà creato all'interno della cartella process, questa cartella viene creata all'interno della cartella dataset precedentemente creata nel drive. Il nome del file creato sarà quello passato dal metodo processed_file_names della classe.

Quando la classe viene richiamata prima controlla se all'interno della cartella process è presente il file con il nome passato nel metodo processed_file_names, se presente andrà a leggere il file, se non presente andrà ad attivare il metodo process per la creazione di un nuovo dataset e gli darà il nome passato tramite il metodo processed_file_names.

In questo script viene utilizzato per la lettura del dataset, si consiglia di controllare che il dataset sian presente all'interno della cartella process e inserire all'interno del metodo processed_file_names il nome del dataset da voler utilizzare per allenare la rete.
"""

import torch
import pandas as pd
import numpy as np
#import networkx as nx
#from torch_geometric.data import InMemoryDataset, Data
import os
import config 

args=config.load()
file_name = "complete_par"


"""
definizione della DGCNN
presa dagli esempi di pythrch geometric al link https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/sort_pool.py
"""

import torch
import torch.nn.functional as F
from torch.nn import Linear, Conv1d
from torch_geometric.nn import SAGEConv, global_sort_pool


class SortPool(torch.nn.Module):
    # definisce i layer
    def __init__(self, dataset, num_layers, hidden):
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
        #MOD MR
        self.lin2 = Linear(hidden, 6)  # Per classificazione multiclasse
        #self.lin2 = Linear(hidden, 1) #i think this is the last layer so change unit to 1! #dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.conv1d.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    # collega i layer
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))  # relu funzione di attivazione del layer
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_sort_pool(x, batch, self.k)  # layer di pooling
        x = x.view(len(x), self.k, -1).permute(0, 2,
                                               1)  # modifica della struttura del vettore per poterlo passare al layer conv1d (devono avere n°nodi=k)
        x = F.relu(self.conv1d(x))
        x = x.view(len(x), -1)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=dropout, training=self.training)
        x = self.lin2(x)

        #MOD MR
        return F.log_softmax(x, dim=1)  # Per classificazione multiclasse
        #return x #shouldn't do activation for regression. original: F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


"""divisione del dataset in train e test in base alla percentuale passata e bilanciata rispetto al target"""


def Split_Target(G,per):
    #checks only for complete cases 
    train = []
    test = []
    
    for x in G:
        train.append(x)
    split=int(len(train)*(per/100))    
    test=train[split:]
    train=train[:split]
    for x in test:
        if x.x.shape[0]==3: #start of a new case 
            break
        else:
            train.append(x)
            #remove from test
            test.remove(x)
    return (train, test)

#include validation set
def Split_Target_valid(G,per,per2):
    #checks only for complete cases 
    train = []
    valid=[]
    test = []
    
    for x in G:
        train.append(x)
    split=int(len(train)*(per/100))    
    test=train[split:]
    train=train[:split]
    for x in test:
        if x.x.shape[0]==3: #start of a new case 
            break
        else:
            train.append(x)
            #remove from test
            test.remove(x)
    split2=int(len(train)*(per2/100)) #keep 80% in train
    valid=train[split2:]
    train=train[:split2]
    for x in valid:
        if x.x.shape[0]==3: #start of a new case 
            break
        else:
            train.append(x)
            #remove from valid
            valid.remove(x)
            
    return (train, valid, test)
#leaving perc to zero everything is put in the train set and you can do with train what you want for cross validation

def Split_Target2(G,per):
    dict = {}
    #line underneath should be changed. (make sure dict target returns the y without changing anything.)
    #tar_std, _ = dictarget()  # dizionario: chiave=attività (target), valore=codice progressivo
    targets=[]
    for x in G:
        targets.append(x.y.item())
    dftarget=pd.DataFrame(targets, columns=['target'])
    dftarget['bins']=pd.qcut(dftarget['target'], q=8)
    for x in dftarget['bins'].unique():  # crea coppie chiave (codice progressivo) - lista vuota per ogni target
        dict[x] = []
    idx=0
    for x in G:
        
        dict[dftarget['bins'].iloc[idx]].append(
            x)  # aggiunge alla lista (valore) nel dizionario alla chiave (codice progressivo) il Data che descrive il grafo
        idx+=1
    train = []
    test = []
    
    for x in dict.keys():
        a = []
        a.extend(dict[x])  # inserisce la lista del dizionario con chiave x alla lista a
        # split effettivo
        #how to make sure that there are complete cases in the set only...?
        l = int(len(a) / 100 * per) #take percentage of x'es with that bin for target 
        atr = a[:l] #train
        ate = a[l:] #test
        for x in ate:
            if x.x.shape[0]==3: #start of a new case 
                break
            else:
                atr.append(x)
                #remove from ate
                ate.remove(x)
                
        train.extend(atr.copy())
        test.extend(ate.copy())
    return (train, test)
"""metodi per salvare le matrici di confusione, salva sia nel formato di tensore pytorch sia su file in formato txt"""


def printcfile_test(cmt, epoch):
    if not os.path.exists('dataset/cm'):
        os.mkdir('/dataset/cm')
    if os.path.isdir('cm'):
        torch.save(cmt, "./dataset/cm/cm_test_" + str(epoch) + ".pt")
        output = open("./dataset/cm/cm_test_" + str(epoch) + ".txt", "w")
        for x in cmt:
            for y in x:
                output.write(str(y.item()))
                output.write("\t")
            output.write("\n")
        output.close()


def printcfile_train(cmt, epoch):
    if not os.path.exists('cm'):
        os.mkdir('./cm')
    if os.path.isdir('cm'):
        torch.save(cmt, "./dataset/cm/cm_train_" + str(epoch) + ".pt")
        output = open("./dataset/cm/cm_train_" + str(epoch) + ".txt", "w")
        for x in cmt:
            for y in x:
                output.write(str(y.item()))
                output.write("\t")
            output.write("\n")
        output.close()


import torch
from torch_geometric.loader import DataLoader
import torch_geometric.datasets
from torch_geometric import utils
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Usa il backend che non richiede una GUI.
import itertools
import numpy as np

import time

import random
import os

# ************
import shutil

chosen_data = ""  # "/bpi12"
k = args.k #15 30
batch_size = args.batch_size #32
batch_size_valid=args.batch_size_valid
hidden=64 #args.batch_size_valid#args.batch_size #this is in original file... 
dropout = args.dropout #0.5,0.05
lr = args.learning_rate# 0.01  #0.1
num_ly = args.num_layers # 2 #5
# ************

per = args.per #perc train 
per2=80 #perc for train-valid split 
delta_epoch = args.num_epochs

# ************


accTr = []  # variabili per salvare i valoti di accuracy e loss per la creazione dei grafici
accTe = []
lossTr = []
lossTe = []

#NO PREPROCESSING
#import dotG_enrichment_Manou_server #this imports the big algorithm first. then enriches it.
#can stay, but remove target par.txt, target std.txt, attributi.txt, processed folder
from creazione_dataset_parallelo_Manou_server import TraceDataset as TD

PATH='./dataset/'

file=PATH+'/processed/'+file_name+'.pt'
if os.path.isfile(file):
  os.remove(file) 
G = TD()
#G = TraceDataset()  # inizializzazione del dataset
"""
print("this")
print(type(G))
for elem in G:
    print("a")
    print(elem)
    print(elem.x)
    print(elem.edge_index)
"""
train, valid, test = Split_Target_valid(G,per,per2) # divisione del dataset tra train e test
#do a split on train set, get 67% of traces for training and 33% for testing. need to know where cases end.

# split=int(len(G)*2/3) #67% training 
# train_t=train[:split]
# #make sure there are only finished cases in there, so there are no prefixes of one case in both train and test
# for x in train[split:]:
#     if x.x.shape[0]==3: #start of a new case 
#         break
#     else:
#         train_t.append(x)
# test=train[len(train_t):]    
# train=train_t

# for x in test:
#     print(x.x)

random.shuffle(train)
random.shuffle(valid)
random.shuffle(test)
import math
batch_test=math.ceil(len(test)/2)

train_loader = DataLoader(train, batch_size=batch_size,
                          shuffle=True)  # definizione delle variabili che conterranno in dataset di train o test per l'allenamento della rete
valid_loader= DataLoader(valid, batch_size=batch_size_valid,
                         shuffle=True) 
test_loader = DataLoader(test, batch_size=batch_test,
                         shuffle=True)  # batch_size serve per l'allenamento della rete quando si hanno grafi di dimensione e struttura variabile

save_folder = f"{args.checkpoint_dir}/"
# torch.save(train_loader,save_folder+"train_dataloader_dgcnn_bs64.pth")
# torch.save(valid_loader,save_folder+"valid_dataloader_dgcnn_bs64.pth")
# torch.save(test_loader,save_folder+"test_dataloader_dgcnn_bs1.pth") #changed to 1

idsavecheck=str(test_loader.dataset[0].idc)
idsavefile=open(save_folder+"idcheck.txt", "w")
idsavefile.write('test save id : {}'.format(idsavecheck))
idsavefile.flush()
idsavefile.close()
#check the model! I made changes, I guess it should be fine. changed nr units to 1 for last layer, and removed activation in last step
#18 features in G, i guess this is for around 15 activities, and the 3 other features 
model = SortPool(dataset=G, num_layers=num_ly, hidden=hidden) #hidden was batchsize!  # definizione del modello della rete
optimizer = torch.optim.Adam(model.parameters(),
                             lr=lr)  # definizione della variabile dove vengono salvati i parametri di ottimizzazione della rete

# MOD MR
criterion = torch.nn.NLLLoss()

#criterion = torch.nn.L1Loss()  #changed to mae # definizione del criterio per l'allenamento della rete

'''
vecchia versione con alcune modifiche:


def train():  # funzione di addestramento della rete
    model.train(True)  # il true serve per indicare che si devono addestrare i paramentri del modello
    running_loss = 0.0
    running_corrects = 0
    all_preds = torch.tensor([]) #predictions
    ystack_std = torch.tensor([]) #actuals

    for data in train_loader:  # itera i batch all'interno del dataset
        """
        print(data[0].x)
        print(data[0].x.shape)
        print(data[0].edge_index)
        print(data[0].edge_index.shape)
        print(data.x[0])
        csob
        """

        optimizer.zero_grad(set_to_none=True)  # imposta il gradiente a zero
        out = model(data)  # risultati di predizione della rete
        #changed!!
        act=data.y.view(data.y.shape[0],1)

        MOD MR nella funzione di loss veniva passato out e act
        #
        loss = criterion(out, data.y)#data.y)  # calcola la loss
        pred = out #are 64 predictions, as this is the batch size as it's just a number and not a category. it's still a tensor in this way... #out.argmax(
            #dim=1)  # calcola la predizione della rete prendendo quella con la probabilità maggiore. Restituisce il valore progressivo dell'attività predetta
        #torch.cat : Concatenates the given sequence of seq tensors in the given dimension

        all_preds = torch.cat((all_preds, pred), dim=0)  # popolo un vettore con tutte le predizioni della rete
        ystack_std = torch.cat((ystack_std, data.y), dim=0)  # popolo un vettore con le activity target (ground True)

        loss.backward()  # derivo il gradiente
        optimizer.step()  # aggiorna i parametri della rete in base al gradiente
        running_loss += loss.item() * data.num_graphs  # serve per calcolare la loss media dell'epoca di addestramento dato che si calcola la loss per ogni batch
        #print('pred',[i[0] for i in pred.tolist()])
        #print('data_y',data.y.tolist())
        
        #this is not working and i don't know how it should work
        # running_corrects += int(
        #     ([int(i[0]) for i in pred.tolist()] == data.y.tolist()).sum())  #don't know if i can make an integer of a torch thing. # conta quante predizioni sono state corrette per poter poi calcolare l'accuracy
    stacked = torch.stack((ystack_std, all_preds.reshape([len(all_preds)])),
                          dim=1)  # creo un vettore di coppie contenenti la predizione della rete e il valore esatto
    
    #don't need the cmt. SHOULD BE CHANGED
    # cmt = torch.zeros(G.num_classes, G.num_classes,
    #                   dtype=torch.int64)  # inizializzo a zero la matrice in cui salvo la metrice diconfusione
    # for p in stacked:  # popolo la matrice di confusione
    #     tl, pl = p.tolist()
    #     cmt[int(tl), int(pl)] = cmt[int(tl), int(pl)] + 1
        
    return running_loss,stacked#, running_corrects#, cmt




'''

##nuova versione MOD MR
def train():
    model.train(True)
    running_loss = 0.0
    all_preds = torch.tensor([])  # predictions
    ystack_std = torch.tensor([])  # actuals

    for data in train_loader:
        optimizer.zero_grad(set_to_none=True)
        out = model(data)
        loss = criterion(out, data.y)  # Non modificare data.y con view()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data.num_graphs

        # Usa argmax per ottenere le classi predette dall'output del modello
        _, preds = torch.max(out, 1)
        all_preds = torch.cat((all_preds, preds.float()), dim=0)  # Assicurati che sia dello stesso tipo
        ystack_std = torch.cat((ystack_std, data.y.float()), dim=0)  # Assicurati che sia dello stesso tipo

    # Ora all_preds e ystack_std hanno la stessa lunghezza e possono essere impilati
    stacked = torch.stack((ystack_std, all_preds), dim=1)

    return running_loss, stacked



'''

                                                versione precedente





def test(loader):  # funzione per il test della rete
    model.train(False)  # passando il false dico al programma di non addestrare il modello
    running_loss = 0.0
    running_corrects = 0

    all_preds = torch.tensor([])
    ystack_std = torch.tensor([])

    for data in loader:  # itera i batch all'interno del dataset
        out = model(data)  # risultati di predizione della rete
        act=data.y.view(data.y.shape[0],1)
        loss = criterion(out,act)# data.y)  # calcola la loss
        pred = out #.argmax(dim=1)  # calcola la predizione della rete prendendo quella con la probabilità maggiore
        all_preds = torch.cat((all_preds, pred), dim=0)  # popolo un vettore con tutte le predizioni della rete
        ystack_std = torch.cat((ystack_std, data.y), dim=0)  # popolo un vettore con le activity target
        running_loss += loss.item() * data.num_graphs  # serve per calcolare la loss media dell'epoca di addestramento dato che si calcola la loss per ogni batch
        # running_corrects += int(
        #     ([int(i[0]) for i in pred.tolist()] == data.y.tolist()).sum())  # conta quante predizioni sono state corrette per poter poi calcolare l'accuracy
    stacked = torch.stack((ystack_std,  all_preds.reshape([len(all_preds)])),
                          dim=1)  # creo un vettore di coppie contenenti la predizione della rete e il valore esatto
    
    #AGAIN NOT NEEDED
    # cmt = torch.zeros(G.num_classes, G.num_classes,
    #                   dtype=torch.int64)  # inizializzo a zero la matrice in cui salvo la metrice diconfusione
    # for p in stacked:  # popolo la matrice di confusione
    #     tl, pl = p.tolist()
    #     cmt[int(tl), int(pl)] = cmt[int(tl), int(pl)] + 1
        
    return running_loss,stacked#, running_corrects #, cmt

'''


#   ********************************************* MOD MR **************************
# 
# 
# 
# 
# 
# 
# #
def test(loader):
    model.eval()  # Imposta il modello in modalità valutazione
    running_loss = 0.0
    all_preds = torch.tensor([])
    ystack_std = torch.tensor([])

    with torch.no_grad():  # Disattiva il calcolo dei gradienti per risparmiare memoria e calcoli
        for data in loader:
            out = model(data)
            loss = criterion(out, data.y)  # Assicurati che data.y sia un vettore 1D con le etichette corrette
            running_loss += loss.item() * data.num_graphs

            # Usa argmax per ottenere le classi predette dall'output del modello
            _, preds = torch.max(out, 1)
            all_preds = torch.cat((all_preds, preds.float()), dim=0)
            ystack_std = torch.cat((ystack_std, data.y.float()), dim=0)

    # Calcola la matrice di confusione o altre metriche di valutazione qui se necessario

    return running_loss, torch.stack((ystack_std, all_preds), dim=1)

'''

vecchia versione


def plot_confusion_matrix(cm, title, normalize=False,
                          cmap=plt.cm.Blues):  # funzione che serve per creare l'immagine della matrice di confusione
    # input=open('target.txt')
    input = open('./dataset/target_std.txt')
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
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

'''

#   ************************************* MOD MR ***********************
# 
# 
# 
# 
# 
# 
# 
# #

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Questa funzione stampa e visualizza la matrice di confusione.
    La normalizzazione può essere applicata impostando `normalize=True`.
    """
    # Calcola la matrice di confusione
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Leggi le classi dal file
with open('./dataset/target_par.txt', 'r') as file:
    classes = file.read().splitlines()

# Assumi che y_true e y_pred siano le liste o tensori delle etichette reali e predette
# y_true = [...]
# y_pred = [...]

# Chiama la funzione plot_confusion_matrix
#plot_confusion_matrix(y_true, y_pred, classes=classes, normalize=False)


def resume_from_checkpoint(
        path_to_checkpoint):  # funzione che serve per far ripartire l'addestramento della rete da un checkpoint salvato
    if os.path.isfile(path_to_checkpoint):
        # Caricamento del checkpoint
        checkpoint = torch.load(path_to_checkpoint)
        # Ripristino dello stato del sistema
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt'])
        #print("Caricato il checkpoint '{}' (epoca {})".format(path_to_checkpoint, checkpoint['epoch']))
    else:
        start_epoch = 1
    return start_epoch


def resume_best_loss_test(
        path_to_checkpoint):  # serve per riprendere il valore della miglior loss di test degli addestramenti precedenti
    if os.path.isfile(path_to_checkpoint):
        b = torch.load(path_to_checkpoint)
        best_loss_test = b['loss']
        return best_loss_test
    else:
        best_loss_test = 0;
        return best_loss_test


def resume_best_loss_train(
        path_to_checkpoint):  # serve per riprendere il valore della miglior loss di train degli addestramenti precedenti
    if os.path.isfile(path_to_checkpoint):
        b = torch.load(path_to_checkpoint)
        best_loss_train = b['loss']
        return best_loss_train
    else:
        best_loss_train = 0;
        return best_loss_train



# #modificare con il checkpoint da cui far partire la rete e modificare con il path in cui si trova
if not os.path.exists(args.checkpoint_dir):
    os.mkdir(args.checkpoint_dir)
if not os.path.exists(args.checkpoint_dir+"/best_test") or not os.path.exists(args.checkpoint_dir+"/best_train"): 
  os.mkdir(args.checkpoint_dir+"/best_test")
  os.mkdir(args.checkpoint_dir+"/best_train")
if os.path.isdir(args.checkpoint_dir):
  #!!!
  #check saved file for good names of files!  
  start_epoch = resume_from_checkpoint(args.checkpoint_dir +'/checkpoint_99.pth.tar')  #NOTA QUESTO 50 è FITTIZIO, BISOGNA VEDERE IL FILE SALVATO PER IL CORRETTO NOME
  best_loss_test = resume_best_loss_test(args.checkpoint_dir+'/best_test/best_model.pth.tar')
  best_loss_train = resume_best_loss_train(args.checkpoint_dir+'/best_train/best_model.pth.tar')

if not os.path.exists(args.checkpoint_dir+"/log"): 
  os.mkdir(args.checkpoint_dir+"/log")
if os.path.isdir(args.checkpoint_dir+"/log"):
  file_log = open(args.checkpoint_dir+"/log/log_split_target_"+str(per)+'_'+str(100-per)+'_epoch_'+str(start_epoch)+'_'+str(start_epoch+delta_epoch-1)+'.txt', 'w')    #apre un file per il salvataggio del log della rete, per ogni epoca salva i valori di accuracy edi loss per il train ed il test

#add early stopping 
patience=10
triggertimes=0
start_time = time.time()
start_epoch = 0
best_loss_test=0
best_loss_train=0
best_outcome_train=[]
best_outcome_test=[]
#each epoch is printed three times, not done three times :) 
import time 
t1=time.time()
for epoch in range(start_epoch, start_epoch + delta_epoch): #0-100  # ciclo delle epoche di addestramento
    t_starte=time.time()
    train_loss, outcome_train= train()#, train_acc = train()  # avvia il training della rete
    test_loss,outcome_test= test(valid_loader) #changed to validation set ! #, test_acc = test(test_loader) #why is test file not using the trained model? # avvia il test della rete
    lossTr.append(train_loss / len(
        train_loader.dataset))  # popola delle liste che contengono i valori di loss e accuracy della rete peril train e per il test
    lossTe.append(test_loss / len(valid_loader.dataset)) #forgot to change this ! 
    t_ende=time.time()



    #   *****************************************   MOD MR  ************************************

    
    # Calcola la matrice di confusione per l'epoca corrente
    y_true_test = outcome_test[:, 0].tolist()
    y_pred_test = outcome_test[:, 1].tolist()
    cmt_test = confusion_matrix(y_true_test, y_pred_test)

    # Plot della matrice di confusione
    plt.figure(figsize=(15, 15))
    plot_confusion_matrix(y_true_test, y_pred_test, classes=classes, title="Confusion matrix TEST epoch : " + str(epoch))
    plt.savefig(args.checkpoint_dir + "/Immagini/cm_epoch/test_epoch_" + str(epoch) + ".png")
    plt.close()

    # Calcola la matrice di confusione per l'epoca corrente (train)
    y_true_train = outcome_train[:, 0].tolist()
    y_pred_train = outcome_train[:, 1].tolist()
    cmt_train = confusion_matrix(y_true_train, y_pred_train)

    # Plot della matrice di confusione
    plt.figure(figsize=(15, 15))
    plot_confusion_matrix(y_true_train, y_pred_train, classes=classes, title="Confusion matrix TEST epoch : " + str(epoch))
    plt.savefig(args.checkpoint_dir + "/Immagini/cm_epoch/train_epoch_" + str(epoch) + ".png")
    plt.close()
    


    #   *************************************************************************************




    # accTr.append(train_acc / len(train_loader.dataset))
    # accTe.append(test_acc / len(test_loader.dataset))

    # printcfile_train(cmt_train, epoch)           #salva le matrici di confusione
    # printcfile_test(cmt_test, epoch)
    # plt.figure(figsize=(15,15))       #crea la figura in cui salvare la matrice di confusione nel formato png
    # plot_confusion_matrix(cmt_test, "Confusion matrix TEST epoch : "+str(epoch))
    # modificare con il path dove si vogliono salvare le immagini della matrice di confusione
    if not os.path.exists(args.checkpoint_dir+"/Immagini/"):
        os.mkdir(args.checkpoint_dir+"/Immagini/")
    if not os.path.exists(args.checkpoint_dir+"/Immagini/best_test") or not os.path.exists(args.checkpoint_dir+"/Immagini/best_train") or not os.path.exists(
            args.checkpoint_dir+"/Immagini/cm_epoch"):  # crea immagini e le relative sottocartelle
        os.mkdir(args.checkpoint_dir+"/Immagini/cm_epoch")
        os.mkdir(args.checkpoint_dir+"/Immagini/best_test")
        os.mkdir(args.checkpoint_dir+"/Immagini/best_train")
    if os.path.isdir(args.checkpoint_dir+"/Immagini/cm_epoch"):
        
        #print(f'Epoch: {epoch:03d}, Train Loss: {(train_loss/len(train_loader.dataset)):.4f}, Test Loss: {(test_loss/len(test_loader.dataset)):.4f} ') #Train Acc: {(train_acc/len(train_loader.dataset)):.4f}, Test Acc: {(test_acc/len(test_loader.dataset)):.4f},
        file_log.write(
            f'Epoch: {epoch:03d}, Train Loss: {(train_loss / len(train_loader.dataset)):.4f}, Test Loss: {(test_loss / len(valid_loader.dataset)):.4f} \n')  # scrive il file di log #Train Acc: {(train_acc / len(train_loader.dataset)):.4f}, Test Acc: {(test_acc / len(test_loader.dataset)):.4f},
        file_log.flush()
        file_log.write('time epoch = {} \n'.format(t_ende-t_starte))
        file_log.flush()
        # TEST: SE LOSS MIGLIORA
        if test_loss < best_loss_test or epoch == 0:  # condizione per decidere la miglior loss di test
            #print(
            #    f'Epoch: {epoch:03d}, Train Loss: {(train_loss / len(train_loader.dataset)):.4f}, Test Loss: {(test_loss / len(test_loader.dataset)):.4f}   **** BEST EPOCH TEST ****') #Train Acc: {(train_acc / len(train_loader.dataset)):.4f}, Test Acc: {(test_acc / len(test_loader.dataset)):.4f},

            # plt.figure(figsize=(15, 15))
            # plot_confusion_matrix(cmt_test, "Confusion matrix BEST TEST epoch : " + str(epoch))
            # modificare con il path dove si vogliono salvare le immagini della matrice di confusione
            # if os.path.isdir("./Immagini/best_test"):
            #     plt.savefig('./Immagini/best_test/test.png')
            #     plt.figure(figsize=(15, 15))
            #     plot_confusion_matrix(cmt_train, "Confusion matrix train epoch: " + str(epoch))
            #     plt.savefig('./Immagini/best_test/train.png')
            state = {'epoch': epoch, 'loss': test_loss, 'state_dict': model.state_dict(), 'opt': optimizer.state_dict()}
            torch.save(state, args.checkpoint_dir+'/best_test/best_model.pth.tar')  # salva info utili sulla best epoch
            best_loss_test = test_loss
            best_outcome_test=outcome_test
            triggertimes=0

            #   *********************************   MOD MR  ******************************************** 
            # 
            # 
            # 
            # #

            # Salva la matrice di confusione per il miglior test
            plt.figure(figsize=(15, 15))
            plot_confusion_matrix(y_true_test, y_pred_test, classes=classes, title="Confusion matrix BEST TEST epoch : " + str(epoch))
            plt.savefig(args.checkpoint_dir + "/Immagini/best_test/cm_test_epoch_" + str(epoch) + ".png")
            plt.close()

            #   *******************************************************************************************
        
        else:
            triggertimes+=1
        #else:
            # print(
            #     f'Epoch: {epoch:03d},  Train Loss: {(train_loss / len(train_loader.dataset)):.4f}, Test Loss: {(test_loss / len(test_loader.dataset)):.4f} ') #Train Acc: {(train_acc / len(train_loader.dataset)):.4f}, Test Acc: {(test_acc / len(test_loader.dataset)):.4f},

        # TRAIN: SE LOSS MIGLIORA
        if train_loss < best_loss_train or epoch == 0:  # condizione per decidere la miglior loss di test
            # print(
            #     f'Epoch: {epoch:03d}, Train Loss: {(train_loss / len(train_loader.dataset)):.4f}, Test Loss: {(test_loss / len(test_loader.dataset)):.4f}   **** BEST EPOCH TRAIN ****') # Train Acc: {(train_acc / len(train_loader.dataset)):.4f}, Test Acc: {(test_acc / len(test_loader.dataset)):.4f},

            # plt.figure(figsize=(15, 15))
            # plot_confusion_matrix(cmt_test, "Confusion matrix test epoch : " + str(epoch))
            # # modificare con il path dove si vogliono salvare le immagini della matrice di confusione
            # if os.path.isdir("./Immagini/best_train"):
            #     plt.savefig('./Immagini/best_train/test.png')
            #     plt.figure(figsize=(15, 15))
            #     plot_confusion_matrix(cmt_train, "Confusion matrix BEST TRAIN epoch: " + str(epoch))
            #     plt.savefig('./Immagini/best_train/train.png')
            state = {'epoch': epoch, 'loss': train_loss, 'state_dict': model.state_dict(),
                     'opt': optimizer.state_dict()}
            torch.save(state, args.checkpoint_dir+'/best_train/best_model.pth.tar')  # salva info utili sulla best epoch
            best_loss_train = train_loss
            best_outcome_train=outcome_train

            #   *********************************   MOD MR  ******************************************** 
            # 
            # 
            # 
            # #

            # Salva la matrice di confusione per il miglior train
            plt.figure(figsize=(15, 15))
            plot_confusion_matrix(y_true_train, y_pred_train, classes=classes, title="Confusion matrix BEST TRAIN epoch : " + str(epoch))
            plt.savefig(args.checkpoint_dir + "/Immagini/best_train/cm_train_epoch_" + str(epoch) + ".png")
            plt.close()
            #   **********************************************************************************
        #early stopping 
        if triggertimes>=patience:
            print('Early stopping!\nBest loss = {}'.format(best_loss_test))
            break
        # else:
        #     print(
        #         f'Epoch: {epoch:03d},  Train Loss: {(train_loss / len(train_loader.dataset)):.4f}, Test Loss: {(test_loss / len(test_loader.dataset)):.4f} ') #Train Acc: {(train_acc / len(train_loader.dataset)):.4f}, Test Acc: {(test_acc / len(test_loader.dataset)):.4f},

file_log.close()
#print("--- %s seconds ---" % (time.time() - start_time))
state = {'epoch': start_epoch + delta_epoch, 'state_dict': model.state_dict(),
         'opt': optimizer.state_dict()}  # creo un dizionario per salvare il checkpoint della rete
# modificare con il path dove si vogliono salvare i checkpoint
torch.save(state, args.checkpoint_dir+'/checkpoint_' + str(
    start_epoch + delta_epoch - 1) + '.pth.tar')  # salvol il ceckpoint della rete
t2=time.time()
#why are only the checkpoints saved and not the model itself?

#everything that is predicted in test set is the same value.
#decreasing the learning rate from 0.01 to 0.0001 helped a bit.
best_outcome_test=best_outcome_test.data.data.tolist()
best_outcome_train=best_outcome_train.data.data.tolist()
bestlosstest=sum([abs(best_outcome_test[i][0]-best_outcome_test[i][1]) for i in range(len(best_outcome_test))])/len(best_outcome_test)

#was a mistake in save folder 
save_folder = f"{args.checkpoint_dir}/"
pred_filepath='{}/Predictions_validation_{}_{}_run.txt'.format(save_folder,args.dataset,args.learning_rate)
with open(pred_filepath,'w') as file:
    file.write('\n training time in hours: {}'.format((t2-t1)/3600))
    file.write('\n best parameters: lr_rate = {}, neurons = {},  nrepochs = {}'.format(args.learning_rate,hidden,args.num_epochs))
    file.write('\n best validation average mae (loss) = {}'.format(bestlosstest))
    file.write('\n best actual values, best predicted values')
    for w in range(len(best_outcome_test)):
        file.write('\n {}'.format(best_outcome_test[w]))
    file.close()

#results test 
outfile=open(save_folder+"resultslog.txt","w")

outfile.write('open trained model...\n')
outfile.flush()
model_test= SortPool(dataset=G, num_layers=num_ly, hidden=hidden)
checkpoint=torch.load(args.checkpoint_dir+'/best_test/best_model.pth.tar')
model_test.load_state_dict(checkpoint['state_dict'])




'''
def test_results(loader):  # funzione per il test della rete
    with torch.no_grad():
        model_test.train(False)  # passando il false dico al programma di non addestrare il modello
        #model_test.eval()
        running_loss = 0.0
        running_corrects = 0
    
        all_preds = torch.tensor([])
        ystack_std = torch.tensor([])
        
        i=0
        for data in loader:  # itera i batch all'interno del dataset
            out = model_test(data)  # risultati di predizione della rete
            act=data.y.view(data.y.shape[0],1)
            loss = criterion(out,act)# data.y)  # calcola la loss
            #pred = out #.argmax(dim=1)  # calcola la predizione della rete prendendo quella con la probabilità maggiore
            all_preds = torch.cat((all_preds, out), dim=0)  # popolo un vettore con tutte le predizioni della rete
            ystack_std = torch.cat((ystack_std, data.y), dim=0)  # popolo un vettore con le activity target
            #running_loss += loss.item() * data.num_graphs  # serve per calcolare la loss media dell'epoca di addestramento dato che si calcola la loss per ogni batch
            i+=1
            outfile.write('{} in loader\n'.format(i))
            outfile.flush()
            # running_corrects += int(
            #     ([int(i[0]) for i in pred.tolist()] == data.y.tolist()).sum())  # conta quante predizioni sono state corrette per poter poi calcolare l'accuracy
        stacked = torch.stack((ystack_std,  all_preds.reshape([len(all_preds)])),
                              dim=1)  # creo un vettore di coppie contenenti la predizione della rete e il valore esatto
        
    #AGAIN NOT NEEDED
    # cmt = torch.zeros(G.num_classes, G.num_classes,
    #                   dtype=torch.int64)  # inizializzo a zero la matrice in cui salvo la metrice diconfusione
    # for p in stacked:  # popolo la matrice di confusione
    #     tl, pl = p.tolist()
    #     cmt[int(tl), int(pl)] = cmt[int(tl), int(pl)] + 1
        
    return stacked#running_loss,stacked#, running_corrects #, cmt

    
'''




#   ********************************** MOD MR ******************************
# 
# 
# 
# 
# 
# 
# 
# 
# #


def test_results(loader):
    model_test.eval()  # Imposta il modello in modalità valutazione
    all_preds = []
    ystack_std = []

    with torch.no_grad():  # Disattiva il calcolo dei gradienti per risparmiare memoria e calcoli
        for data in loader:
            out = model_test(data)  # risultati di predizione della rete
            loss = criterion(out, data.y)  # Calcola la loss usando le etichette originali non ridimensionate
            
            # Usa argmax per ottenere le classi predette dall'output del modello
            _, preds = torch.max(out, 1)
            all_preds.append(preds)
            ystack_std.append(data.y)

    # Concatena tutti i batch insieme
    all_preds = torch.cat(all_preds, dim=0)
    ystack_std = torch.cat(ystack_std, dim=0)

    # Crea un tensore di coppie contenenti la predizione della rete e il valore esatto
    stacked = torch.stack((ystack_std, all_preds), dim=1)
    
    return stacked

t1=time.time()
outfile.write('do predictions...\n')
outfile.flush()
results_test=test_results(test_loader)
t2=time.time()
outfile.write('set predictions to list format...\n')
outfile.flush()
best_outcome_test_final=results_test.data.data.tolist()
outfile.write('nr of prefixes in testset: {} \n'.format(len(best_outcome_test_final)))
outfile.flush()
outfile.write('calculate loss...\n')
outfile.flush()
bestlosstest_final=sum([abs(best_outcome_test_final[i][0]-best_outcome_test_final[i][1]) for i in range(len(best_outcome_test_final))])/len(best_outcome_test_final)

outfile.write('writing predictions txt file...\n')
outfile.flush()
pred_filepath='{}/Predictions_test_{}_{}_run.txt'.format(save_folder,args.dataset,args.learning_rate)
with open(pred_filepath,'w') as file:
    file.write('\n prediction time in hours: {}'.format((t2-t1)/3600))
    file.write('\n best parameters: lr_rate = {}, neurons = {},  nrepochs = {}'.format(args.learning_rate,hidden,args.num_epochs))
    file.write('\n best test average mae (loss) = {}'.format(bestlosstest_final))
    file.write('\n best actual values, best predicted values')
    for w in range(len(best_outcome_test_final)):
        file.write('\n {}'.format(best_outcome_test_final[w]))
    file.close()

outfile.write('create best trace info list...\n')
outfile.flush()
best_trace_info_dgcnn=[]
for i in range(len(best_outcome_test_final)):
    l=[]
    l.append(str(test_loader.dataset[i].idc))
    l.append(np.asarray(test_loader.dataset[i].x))
    l.append(abs(best_outcome_test_final[i][0]-best_outcome_test_final[i][1]))
    l.append(best_outcome_test_final[i][1]) #pred
    l.append(best_outcome_test_final[i][0]) #target
    #prefix length
    pr_len=len(np.asarray(test_loader.dataset[i].x)) 
    l.append(pr_len)
    best_trace_info_dgcnn.append(l)

import pickle
outfile.write('dump list...\n')
outfile.flush()
with open(save_folder+"test_best_trace_info_dgnn.ob", 'wb') as bti:
    pickle.dump(best_trace_info_dgcnn, bti)
    
outfile.close()