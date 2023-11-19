# -*- coding: utf-8 -*-
"""
Only information is added to the g file, the g is not made here. 

With time features from Andrea!
"""
import gc
import os
import config
#will make the first graph
import Copy_of_NewBig_server as bg

args=config.load() 
"""**Features**"""

from pm4py.util import xes_constants as xes
import pm4py
from pm4py.objects.log import obj as log_instance


def get_all_event_attributes_from_log(log):
    """
    Get all events attributes from the log

    Parameters
    -------------
    log
        Log

    Returns
    -------------
    all_attributes
        All trace attributes from the log
    """
    all_attributes = set()
    for trace in log:
        for event in trace:
            all_attributes = all_attributes.union(set(event.keys()))
    if xes.DEFAULT_TRANSITION_KEY in all_attributes:
        all_attributes.remove(xes.DEFAULT_TRANSITION_KEY)
    return all_attributes

def get_all_trace_attributes_from_log(log):
    """
    Get all trace attributes from the log

    Parameters
    ------------
    log
        Log

    Returns
    ------------
    all_attributes
        All trace attributes from the log
    """
    all_attributes = set()
    for trace in log:
        all_attributes = all_attributes.union(set(trace.attributes.keys()))
    if xes.DEFAULT_TRACEID_KEY in all_attributes:
        all_attributes.remove(xes.DEFAULT_TRACEID_KEY)
    return all_attributes

# trace_attr = get_all_trace_attributes_from_log(a)
# print('trace:' + str(trace_attr))
# event_attr = get_all_event_attributes_from_log(a)
# print('event:' + str(event_attr))

"""mostra attributi selezioanbili"""


#<HARDCODED> names2 = feature_strutturali + lista_param_eventi + lista_param_tracce
#IMPORTANTE!! la posizione degli attributi in names2 deve rispettare l'ordine degli attributi nel file .g!!!
#IMPORTANT!! the postions of the attributes in names2 must reflect the order og the attributes in the .g file!!!

#own p2plog
lista_param_tracce = ['concept:name']
lista_param_eventi = ['concept:name', 'time:timestamp']


"""
cerca 
# <hardcoded> COMMENTARE QUI per farlo funzionare su un altro dataset; attributo specifico di DBI12

per trovare porzioni codice da modificare / commentare per heldesk<-->BPI12
"""

"""*add_info_G_file*.py"""

#function that add information to the .g file, taking it from the event log associated
#the information are: timestamp (by the event) and the amount-required (by the trace of the event)

from os import listdir
import sys
# Path della cartella che contiene i file .xes
path_xes_dir = "./Input/xes/"
# Path della cartella che contiene i file .g
path_r_dir = "./Input/g/"
# output path
path_w = "./Output/Pre-cage/"


xes_list = []
from os.path import isfile, join
try:
    xes_list = [join(path_xes_dir,f) for f in listdir(path_xes_dir) if isfile(join(path_xes_dir, f))]
except:
    sys.exit("  Error - " + path_xes_dir + " not exist!")

#need to have this gfile as input, together with the xes file. is gfile result from prom? it results from a petri net?
g_list = []
from os.path import isfile, join
try:
    g_list = [join(path_r_dir,f) for f in listdir(path_r_dir) if isfile(join(path_r_dir, f))]
except:
    sys.exit("  Error - " + path_r_dir + " not exist!")


'''
.g file structure:
    XP
    v 1  ASUBMITTED
    v 2  APARTLYSUBMITTED
    ....
    e 1 2 ASUBMITTED__APARTLYSUBMITTED
'''

def write_function(wfile, text):
    wfile.writelines(text)
    
def v_case(wfile, word, index, log):
    #tmp contains the value of the concept:name;
    #this is necessary to compare the "concept:name" of log file and the concept:name of .g file
    tmp = ' '.join(word[2:]) #CHANGED TO GET CONCEPT:NAME
    #note: count = event_index + 1, to avoid same information by same "concept:name" value
    count = 1
    #with this for cycle we scan each event refers to index (refered to the trace)
    for event_index, _ in enumerate(log[index]):
        #when "concept:name" == tmp && the node count is right, we write in destination file the line + additional info
        #NOTE: we add 2 space for the execution of "temporal_calculated_features.py" file
        #HARDCODED sul campo chiamato concept:name

        #print(index)

        if (tmp == log[index][event_index]['concept:name'] and word[1] == str(count)):
          params_tracks = ""
          for elem in lista_param_tracce:
            params_tracks += str(log[index].attributes[elem]) + " "
          params_tracks = params_tracks.strip()
          # ******************************************************************************************
          # 
          # # MOD MR 
          # ho aggiunto il replace in quanto con il file di log PermitLog_SE il concept:name ha degli spazi e crea problemi quando va a leggere il merged.g
          #
          # df = pd.read_csv(path_read, sep=" ", names=names2, dtype={'name_track':str} )
          # v 2  Starttrip 2016-10-0423:59:59.999000+02:00 travel permit 76455 
          # nell'esempio di sopra con read_csv divide in 3 colonne (sbagliato) travel permit 76455 #
          # 
          # #
          params_tracks = params_tracks.replace(' ', '_')
          params_elems = ""
          for elem in lista_param_eventi:
            try:
              s=str(log[index][event_index][elem])
              s=s.replace(" ","") #changed this to make one word of concept:name
              params_elems += s + " "
              #se non presente viene impostato il valore -1
            except KeyError:
              params_elems += "-1" + " "
          params_elems = params_elems.strip()
          a = (word[0] + " " + word[1] + " " + " "  + params_elems + " " + params_tracks + "\n");  
          write_function(wfile, a)
        count += 1

def e_case(wfile, word):
    a = word[0] + " " + word[1] + " " + word[2] + " " +"".join(word[3:])+ "\n" #changed this to make one word of concept:name
    write_function(wfile, a)

def add_info(path_w, path_r, log):
    #When we find 'XP' we increase the trace index by 1. 
    #Each trace in .g start with 'XP' --> for the firs time we need to set trace_index= -1
    trace_index = -1;
    w = open(path_w, "a") #file in which we append trace + additionals information
    r = open(path_r, "r") #file .g to which informations is to be added

    #scan line by line the .g file
    for line in r:
        #split line into a list where each word is a list item
        riga = line.split()
        
        #control if line is empty; if it's true, the line isn't empty
        if line.strip() :
            #if the list contain 'XP' we are in a new trace, --> we increase trace_index by 1
            if riga[0] == 'XP':
                write_function(w,'XP\n')
                trace_index = trace_index + 1;
            #if the first element of list is 'v' --> the line refers to a node (event) --> we have to add additional informations
            if riga[0] =='v':
                v_case(w, riga, trace_index, log)
            elif riga[0] =='e':
                #if the first element of list is 'e' --> the line refers to an edge --> no additional informations. We only write the line in the destination file
                e_case(w, riga)
        else:
            #empty line case
            write_function(w,' \n')
    r.close() 
    w.close() 

#import xes file
from pm4py.objects.log.importer.xes import importer as xes_importer

# each rows of logs contains one xes files

logs = []
for f in xes_list:
  logs.append(xes_importer.apply(f))

# dynamic call of add_info function based on previous logs list

for i in range(len(logs)):
  add_info(path_w + str(i) + ".g", g_list[i], logs[i])

if not os.path.exists('./dataset'):
    os.mkdir('./dataset')
original= r'./Output/Pre-cage/0.g'
target= r'./dataset/0.g'
import shutil
shutil.copyfile(original,target)


#append the 4 file.g in only one file "merged.g"
filenames = []
for i in range(len(logs)):
  filenames.append(path_w + str(i) + ".g")

# cl=open(filenames[0], 'wb') 
# cl.close()

# Open merged.g in write mode 
with open(path_w + 'merged.g', 'w') as outfile: 
  
    # Iterate through list 
    for names in filenames: 
  
        # Open each file in read mode 
        with open(names) as infile: 
            outfile.write(infile.read()) 
        outfile.write('')
        #outfile.close()











#     ************************************************************************
#  
#                 END CREATE .G FILE
# 
#                 START temporal_and_ohe**.py
# 
#                 ADD A TEMPORAL FEATURES:
# 
#                 #start time, 
#                 #norm time, 
#                 #prev event time,
#                 #trace time                
# 
# 
#     ************************************************************************



import pandas as pd
import numpy as np
import datetime
#import networkx as nx
from tensorflow.keras.utils import to_categorical
import math

#os.system("head -1000 " + PATH + "Output/Pre-cage/merged.g > " + PATH + "Output/Pre-cage/merged_short.g")

path_read = "./Output/Pre-cage/merged.g" #all the 4 .g partition merged

path_write =  "./dataset/complete.g"

# cl=open(path_write, 'wb') 
# cl.close()
#NOTE: you have to check if the .g file has the correct fields before the execution; see the .g example!!
#NOTE: each 'v' row must have 2 space before concept:name to make 'node2' = NaN
'''
.g file structure:
    XP
    v 1  ASUBMITTED 2011-09-30 22:38:44.546000+00:00 20000
    v 2  APARTLYSUBMITTED 2011-09-30 22:38:44.880000+00:00 20000
    ....
    e 1 2 ASUBMITTED__APARTLYSUBMITTED
    ...
    
'''

e_v_column = 0

#Aggiunge _event agli attributi relativi agli attributi e track agli attributi relativi alle tracce

for i in range(len(lista_param_eventi)):
  for j in range(len(lista_param_tracce)):
    if lista_param_eventi[i] == lista_param_tracce[j]:
      lista_param_eventi[i] = lista_param_eventi[i] + "_event"
      lista_param_tracce[j] = lista_param_tracce[j] + "_track"

#e li unisce in una riga unica
names2 = ["e_v", "node1", "node2"] + lista_param_eventi + lista_param_tracce

for i in range(len(names2)):
  names2[i] = names2[i].split(':')[-1]

#<HARDCODED> sui nomi timestamp e REG_DATE *****ANDREA****

df = pd.read_csv(path_read, sep=" ", names=names2, dtype={'name_track':str} ) #accede alle variabili lista_param_eventi e lista_param_tracce dello script add_info_g_file.py ////////////// legge il .g file prendendo solo le colonne presenti nella variabile names2

#until here it's okay!
df.timestamp = df.timestamp.apply(lambda x: str(x)[:18])

#df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d%H:%M:%S', errors='coerce')
df['timestamp']=pd.to_datetime(df['timestamp'], format='%Y-%m-%d%H:%M:%S', utc=True) #add utc to be sure (,utc=True)
#df['timestamp']=pd.to_datetime(df['timestamp'], format='%Y-%m-%d%H:%M:%S', utc=True)

tmp=df.timestamp  #Salvo in una variabile di appoggio tutti i timestamp per inserirli in una nuova colonna 'finish'
df['finish']=tmp
df.drop(['timestamp'], axis=1, inplace = True)


"""
#8 is the number of characters until the second, in order to remove milliseconds and timezone
df.REG_DATE_TIME = df.REG_DATE_TIME.apply(lambda x: str(x)[:8]) # <hardcoded> COMMENTARE QUI per farlo funzionare su un altro dataset; attributo specifico di DBI12

tmp = df.REG_DATE_DATE + " " + df.REG_DATE_TIME # <hardcoded> COMMENTARE QUI per farlo funzionare su un altro dataset; attributo specifico di DBI12
df['REG_finish'] = tmp #add datetime column
df.drop(['REG_DATE_DATE', 'REG_DATE_TIME'], axis=1, inplace = True) #remove date, time columns # <hardcoded> COMMENTARE QUI per farlo funzionare su un altro dataset; attributo specifico di DBI12
df['REG_finish'] = pd.to_datetime(df['REG_finish'], format='%Y-%m-%d %H:%M:%S')#change format datetime
"""

#</HARDCODED>
#we assign, for each node, the start time as the maxim (finish) time of its predecessor 
'''
we'll use these 2 dataframe in the following ways:
    - edge_dataframe: to temporally store the edges relative to each trace
    - g_dataframe: will contain the old df dataframe, with the new column "start". 

the pipeline will be:
    1. add nodes in the graph
    2. add edges in the graph, and fill the edge_dataframe
    3. add to g_dataframe nodes in the graph
    4. add to g_dataframe edges from edge_dataframe
    5. iterate 
'''
#arrivati a questo punto il dataframe ha la seguente struttura:
#edge_dataframe = pd.DataFrame({"e_v" : [], "node1" : [], "node2" : [], "name_event" : [], "resource" : [], "transaction" : [], "name_track" : [], "AMOUNT_REQ" : [], "finish" : [], "REG_finish" : []})

# g_dataframe_columns = dict()
# for item in df.columns:
#   g_dataframe_columns.update({item : []})

# g_dataframe    = pd.DataFrame(g_dataframe_columns)
# edge_dataframe = pd.DataFrame(g_dataframe_columns)

# #the start and end times are not updated well yet

# DG = nx.DiGraph() #Direct graph
# for index,row in df.iloc[:].iterrows():
#     if (row['e_v'] == 'XP'):
#         DG.clear()
#         DG = nx.DiGraph()
#         g_dataframe = g_dataframe.append({'e_v': ''}, ignore_index = True) # add empty line to the dataframe
#         g_dataframe = g_dataframe.append({'e_v': 'XP'}, ignore_index = True) #add XP to the dataframe
#     elif (row['e_v'] == 'v'): #node case
#         #add the node to the graph with its informations
#         qwe = dict()
#         for i in range(len(row)):
#             qwe.update({df.columns[i] :  row[i]})
#         qwe.pop("e_v")
#         qwe.pop("node2")
#         DG.add_nodes_from([(row['node1'], qwe)])
#     elif(row['e_v'] == 'e'):#edge case
#         #add the edge to the graph 
#         DG.add_edge(row['node1'], row['node2'])
#         #set a dummy date for the edge's row, that we'll replace
#         time0 = datetime.datetime.fromisoformat("2021-01-01T00:00:00") 
#         edge_dataframe = edge_dataframe.append({'e_v': 'e',
#                                                 'node1': row['node1'],
#                                                 'node2': row['node2'],
#                                                 'name_event': row['name_event'], 
#                                                 'start': time0, 
#                                                 'finish': time0}, #i don't see where those times are replaced...
#                                                ignore_index = True)
#     elif(pd.isnull(row['e_v']) ): #once added edge and node to the graph, we have to calculate the start time of each node
#         for node, data in DG.nodes(data = True): # scan each node in the graph
#             maxim = datetime.datetime.min #set a "zero" datetime
#             predecessor_list = list(DG.predecessors(node)) # set of predecesso of each node in the graps
            
#             if not (predecessor_list == []): #if list isn't empty
#                 for element in predecessor_list: #scan each predecessor
                    
#                     if DG.nodes[element]['finish'] > maxim:
#                         maxim = DG.nodes[element]['finish'] 
#             else: #if there are no predecessoo (first event) we define start time = finish time
#                 maxim = DG.nodes[node]['finish']
#             #once defined the start time of the event, we add the node to the dataframe
#             dizionario2 = dict({'e_v': 'v'})
#             dizionario2.update(DG.nodes[node])
#             dizionario2.update({'start': [maxim]}) #the start time is the time stamp of the last event because that is the completion timestamp.
#             df2 = pd.DataFrame(dizionario2)
#             g_dataframe = g_dataframe.append(df2, ignore_index = True)
#         #once added all the node, we add to the dataframe all the edges
#         g_dataframe = g_dataframe.append(edge_dataframe, ignore_index = True)
#         #delete all rows of the edge_dataframe dataframe
#         ''' dirty but mantain the structure'''
#         edge_dataframe = edge_dataframe.iloc[0:0] 

#create g_dataframe myself
print_file=open('./log.txt','w')
print_file.write('Start with g_dataframe...\n')
print_file.flush()

g_dataframe=pd.DataFrame(columns=list(df.columns))  #viene creato il g_dataframe con stesso numero/nome delle colonne df
g_dataframe.loc[0]=np.array(np.nan*len(g_dataframe.columns))
g_dataframe=pd.concat((g_dataframe,df),ignore_index=True)
g_dataframe['e_v']=g_dataframe['e_v'].fillna('')
g_dataframe=g_dataframe[:-1]
df_shift=pd.DataFrame(columns=list(df.columns))
df_shift.loc[0]=np.array(np.nan*len(df_shift.columns))
# df_shift.loc[0]=np.zeros(len(df_shift.columns))
df_shift.loc[1]=np.array(np.nan*len(df_shift.columns))
df_shift=pd.concat((df_shift,df),ignore_index=True)
df_shift['e_v']=df_shift['e_v'].fillna('')
df_shift=df_shift[:-1]
g_dataframe['start']=df_shift['finish'].copy()  #non è spiegato, presumo che viene fatto per traslare verso il basso la colonna timestamp originale e successivamente per valutare se prendere il valore della stessa riga del timestamp o eventualmente quella precedente
g_dataframe['start']=g_dataframe.apply(lambda x: x['finish'] if x['node1'] == 1 else x['start'], axis=1)

#changed!!
del [[df,df_shift]]
gc.collect()
df,df_shift=pd.DataFrame(),pd.DataFrame()

g_dataframe['finish'].fillna(pd.NaT,inplace=True)
#try this as well 
g_dataframe['start'].fillna(pd.NaT,inplace=True)
g_dataframe['start'].replace(0, pd.NaT,inplace=True)

#maybe not necessary...
g_dataframe['finish']=pd.to_datetime(g_dataframe['finish'], format='%Y-%m-%d %H:%M:%S',errors='coerce')
g_dataframe['start']=pd.to_datetime(g_dataframe['start'], format='%Y-%m-%d %H:%M:%S',errors='coerce')

# EXTRACT TRACE TIME, PREV EVENT TIME not normalized and NORM TIME
# def get_normalized_date_of_week(date):
#     day_o_week = (datetime.datetime.weekday(date))
#     norm_timeOfWeek = (day_o_week*24*60)+((date.hour*60) + date.minute)#--> ora assoluta (in minuti) dalla mezzanotte di domenica
#     norm_timeOfWeek = norm_timeOfWeek/10080
#     return norm_timeOfWeek

# # norm_time_list = [] #will contain the finish time in g_dataframe, but normalized. but finish time is never filled???
# # #base contain the column "finish" of the dataframe g_dataframe
# # base = g_dataframe['finish']

# #do normalization myself
# def normalize(v): 
#     try:
#         v = get_normalized_date_of_week(v)
#     except:
#         norm_date=np.nan 
#     return v

# g_dataframe['norm_time']=g_dataframe.apply(lambda x: normalize(x['finish']),axis=1)
print_file.write('Start with feature engineering...\n')
print_file.write('norm_time...\n')
print_file.flush()

'''
# Calcola la colonna 'norm_time' in base al giorno della settimana e all'ora

#Infine, dividiamo il tempo totale in minuti per 10080, che è il numero totale di minuti in una settimana (7 giorni * 24 ore * 60 minuti).
#Questa divisione normalizza il tempo in modo che il risultato sia compreso tra 0 (inizio della settimana) e 1 (fine della settimana)

'''
g_dataframe['norm_time']=((g_dataframe['finish'].dt.dayofweek*24*60)+((g_dataframe['finish'].dt.hour*60) + g_dataframe['finish'].dt.minute))/10080

#the base column will be normalized 
#what does he want to do with the nan values?
# for index in range(len(base)):

#     try:
#         norm_date = get_normalized_date_of_week(base[index])
#     except:
#         norm_date=np.nan 
#     norm_time_list.append(norm_date)
# #add the normalized time to g_dataframe
# g_dataframe['norm_time'] = pd.DataFrame(norm_time_list)


# delta_trace = []  # will contain trace time
# delta_action = [] # will contain prev event time
# begin_time = 0
# previous_time = 0
# base = g_dataframe.values

# search the finish and start column position

# i=0
# while g_dataframe.columns[i] != "finish":
#   i=i+1
# finish_column = i


# i=0
# while g_dataframe.columns[i] != "start":
#   i=i+1
# start_column = i

# for index in range((base.shape[e_v_column])): #scan all rows in dataframe
#     complete_date = base[index][finish_column] #finish time
#     #if there is a new XP the row is referd to another trace --> set the start time =  finish time
#     if base[index-1][e_v_column] == 'XP' and base[index][e_v_column] == 'v':
#         begin_time = base[index][finish_column] #start time of the trace
#         previous_time = base[index][finish_column] #start time of the event
#         delta_trace.append(0)  #set the trace time = 0
#         delta_action.append(0) #set the prev_event time = 0
#     else:
#         try:
#             a = complete_date - begin_time # trace time
#             delta_trace.append((a.days*24*60) + (a.seconds//60))
#         except:
#             delta_trace.append(np.nan)
#         try:
#             previous_time = base[index][start_column] # start time of the event
#             a = complete_date - previous_time # time it took the event to finish
#             delta_action.append((a.days*24*60) + (a.seconds//60))
#         except:
#             delta_action.append(np.nan)
            
# def eventtime(v,z):
#     complete_date=v
#     previous_date=z
#     try:
#         a = complete_date - previous_date
#         #=(a.days*24*60) + (a.seconds//60)
#     except:
#         a=pd.NaT
#     return a 
       
       
# begintimes=g_dataframe.groupby('name_track',sort=False,as_index=False)['finish'].agg('min')

# def begin(v):
#     try:
#         value=begintimes.loc[begintimes['name_track']==v,'finish'].iloc[0]
#     except:
#         value=np.nan
#     return value
# g_dataframe['begin']=g_dataframe.apply(lambda x: begin(x['name_track']),axis=1)
# g_dataframe['trace_time']=g_dataframe.apply(lambda x: eventtime(x['finish'],x['begin']), axis=1)  
# g_dataframe['prev_event_time']=g_dataframe.apply(lambda x: eventtime(x['finish'],x['start']), axis=1) 
# g_dataframe.drop('begin',inplace=True,axis=1)

# a = np.asarray(g_dataframe['trace_time'])
# b = np.asarray(g_dataframe['prev_event_time'])
# a = a.reshape((a.shape[0],1))
# b = b.reshape((a.shape[0],1))
# g_dataframe['trace_time']= pd.DataFrame(a)
# g_dataframe['prev_event_time']=pd.DataFrame(b)

print_file.write('trace_time...\n')
print_file.flush()

g_dataframe['trace_time']=g_dataframe.groupby('name_track',sort=False)['finish'].transform('min')
g_dataframe['trace_time']=g_dataframe['finish']-g_dataframe['trace_time']
g_dataframe['trace_time']=(g_dataframe['trace_time'].dt.days*24*60)+(g_dataframe['trace_time'].dt.seconds//60)
#changed!!
#g_dataframe['prev_event_time']=g_dataframe.apply(lambda x: eventtime(x['finish'],x['start']), axis=1)
print_file.write('prev_event_time...\n')
print_file.flush()
g_dataframe['prev_event_time']=g_dataframe['finish']-g_dataframe['start'] 
g_dataframe['prev_event_time']=(g_dataframe['prev_event_time'].dt.days*24*60)+(g_dataframe['prev_event_time'].dt.seconds//60)

#adds the newly calculated values to the dataframe
# g_dataframe['trace_time'] = pd.DataFrame(a)
# g_dataframe['prev_event_time'] = pd.DataFrame(b)

#add target to nodes in the graph!!
print_file.write('Start with extra own features...\n')
print_file.flush()
print_file.write('read data frame...\n')
print_file.flush()



# **********************************************************************************
#    
# 
# 
# 
# 
# 
# 
#     QUESTO TARGETFRAME NON LO UTILIZZO IN QUANTO NON FACCIO REGRESSIONE E NON HO BISOGNO DI CALCOLARMI IL TARGET CON IL DAYS TO LATE.
#     MOD MR 
# 
# 
# 
# 
# 
# 
# # *********************************************************************************************

#targetframe=pd.read_csv(f"./{args.csv_name}",low_memory=False)
targetframe = bg.df_targetframe
#targetframe = convert_xes_to_csv(log_file, path_csv)
col_name = 'Case ID'
if col_name in targetframe.columns:
    targetframe[col_name] = targetframe[col_name].astype(str)
else:
    print(f"La colonna {col_name} non esiste.")


##############___________MOD_B___________##################
# pulizia di tutte le colonne con il numero di 0 > 60/70 % or Nan > 5% eliminare
col_eliminate = []
soglia_0 = int(len(targetframe) * 0.66) # Calcola la soglia per il numero massimo di zeri consentiti
colonne_da_eliminare_0 = targetframe.columns[(targetframe == 0).sum() > soglia_0] # Elimina le colonne con il 66% di zeri
targetframe = targetframe.drop(columns=colonne_da_eliminare_0)
col_eliminate = colonne_da_eliminare_0.to_list()

soglia_nan = int(len(targetframe) * 0.05) # Calcola la soglia per il numero massimo di NaN consentiti
colonne_da_eliminare_nan = targetframe.columns[targetframe.isna().sum() > soglia_nan] # Elimina le colonne con il 5% di NaN
targetframe = targetframe.drop(columns=colonne_da_eliminare_nan)
col_eliminate += colonne_da_eliminare_nan.to_list()

col_unique_val = [col for col in targetframe.columns if targetframe[col].nunique() == 1]
targetframe = targetframe.drop(columns=col_unique_val)
col_eliminate += col_unique_val

with open("col_eliminate.txt", "w") as file:
    file.write("Colonne eliminate per numero di 0 > 66%:\n")
    file.write("\n".join(colonne_da_eliminare_0) + "\n\n")
    file.write("Colonne eliminate per numero di NaN > 5%:\n")
    file.write("\n".join(colonne_da_eliminare_nan) + "\n")
    file.write("Colonne eliminate con stesso valore per ogni riga:\n")
    file.write("\n".join(col_unique_val) + "\n")

print("Colonne eliminate per zeri:", colonne_da_eliminare_0)
print("Colonne eliminate per NaN:", colonne_da_eliminare_nan)
print("Colonne eliminate per valore unico:", col_unique_val)



# ************************************************************************************
# 
# 
# 
#                 # MOD MR
#                 # Parametrizzo tutte le colonne  
# 
# 
# 
# 
# 
# #


from GUI import SimpleGui as SG

clm=targetframe.columns.values

#'Case ID' - Activity - Complete Timestamp
#Attributi obbligatori da usare
attribute = ['Case ID', 'Activity', 'Timestamp', 'Variant', 'Variant Index']

# Crea una lista delle colonne da rimuovere
columns_to_remove = [col for col in targetframe.columns if any(attr in col for attr in attribute)]

# Calcola la differenza tra le colonne originali e quelle da rimuovere
selected_columns = [col for col in targetframe.columns if col not in columns_to_remove]

clm = targetframe.columns.values
clm = np.sort(clm)

# Selezionare le colonne interessate
selected_columns = SG(sorted(selected_columns))



'''


# Lista dei prefissi delle colonne da considerare
prefixes = ['bu_', 'plant_', 'item_', 'vendor_', 'MatnrShort_']

# Filtrare le colonne con i prefissi
filtered_columns = [col for col in targetframe.columns if any(col.startswith(prefix) for prefix in prefixes)]

# Aggiungere colonne al g_dataframe
for col in filtered_columns:
    bins_col_name = f'bins_{col}'
    targetframe[bins_col_name] = targetframe[col].idxmax(axis=1)
    targetframe.drop(list(targetframe.filter(like=col).columns)[:-1], axis=1, inplace=True)

# Spostare 'Days too late' in ultima posizione
dtl = targetframe['Days too late']
targetframe.drop('Days too late', axis=1, inplace=True)
targetframe['Days too late'] = dtl

'''

# Raggruppare per 'Case ID' e calcolare il massimo
#targetframe = targetframe.groupby('Case ID', sort=False, as_index=False).agg('max')



print_file.write('Set values, sizes and array for all features + target and add to g_dataframe...\n')
print_file.flush()
sizes=np.array(g_dataframe.groupby('name_track',sort=False,as_index=False).size()['size'])

idxss=list(np.where(~g_dataframe['name_track'].isnull()))[0] #Trova gli indici delle righe in cui la colonna 'name_track' di g_dataframe non è nulla.



# Set values, sizes, and array per tutte le colonne selezionate
for i in selected_columns:
    print_file.write(f'Aggiungi {i} a g_dataframe...\n')
    print_file.flush()
    arr = sum([[s] * n for s, n in zip(targetframe[i], sizes)], [])
   
    g_tmp = targetframe[i]
    g_dataframe[i] = [np.nan] * len(g_dataframe)
    arr = pd.Series(arr)
    arr.index = idxss
    g_dataframe[i] = arr




# targetframe=pd.read_csv(f"./{args.csv_name}", usecols=([i for i in range(0,79)]),dtype={'Case ID': str},low_memory=False)
# targetframe=targetframe.drop(['Weekday','matgroup_4', 'matgroup_7', 'matgroup_others']+list(targetframe.filter(like='Weekday_').columns)+list(targetframe.filter(like='ACTIVITY_EN_').columns), axis=1)
# print_file.write('add bins for categories...\n')
# print_file.flush()
# ## Discretizzazione dei valori

# for c in ['bu_','plant_', 'item_', 'vendor_','MatnrShort_']:
#   targetframe['bins_{}'.format(c)]=targetframe.filter(like=c).idxmax(axis=1) 
#   targetframe.drop(list(targetframe.filter(like=c).columns)[:-1],axis=1,inplace=True)

# dtl=targetframe['Days too late']
# targetframe.drop('Days too late',axis=1,inplace=True)
# targetframe['Days too late']=dtl  #viene fatto per inserire la colonna dtl come ultima posizione

# print_file.write('groupby target frame...\n')
# print_file.flush()
# # targetframe=targetframe.groupby('caseID_hash',sort=False,as_index=False)['Days too late'] #.agg('max')
# targetframe=targetframe.groupby('Case ID',sort=False,as_index=False).agg('max')#['Days too late'] #.agg('max')
# idx=list(range(3,7))+list(range(10,len(targetframe.columns)))
# columns=list(np.array(targetframe.columns)[idx])

# print_file.write('Set values, sizes and array for all features + target and add to g_dataframe...\n')
# print_file.flush()
# sizes=np.array(g_dataframe.groupby('name_track',sort=False,as_index=False).size()['size'])
# idxss=list(np.where(~g_dataframe['name_track'].isnull()))[0] #Trova gli indici delle righe in cui la colonna 'name_track' di g_dataframe non è nulla.

# for i in columns:
#     print_file.write('add {} to g_dataframe...\n'.format(i))
#     print_file.flush()
#     arr=sum([[s] * n for s, n in zip(targetframe[i], sizes)], [])
#     if i=='Days too late':
#         g_dataframe['target']=[np.nan]*len(g_dataframe)
#         arr=pd.Series(arr)
#         arr.index=idxss
#         g_dataframe['target']=arr
#     else:
#         g_tmp = targetframe[i]
#         g_dataframe[i]=[np.nan]*len(g_dataframe)
#         arr=pd.Series(arr)
#         arr.index=idxss
#         g_dataframe[i]=arr
        


#
# 
# 
# 
#     PARTE DI BLOCCO COMMENTATA FINO A QUI
# 
# 
# 
# 
# MOD MR
# 
# 
# 
# 
# 
# 
# 
# 
# # ************************************************************


# values=targetframe.agg('max')['Days too late']

# print_file.write('Set array...\n')
# print_file.flush()
# # arr=sum([[s] * n for s, n in zip(values, sizes)], [])
# # import itertools
# # arr=list(itertools.chain(*arr))
# print_file.write('Set idxss...\n')
# print_file.flush()


# print_file.write('add target to gframe...\n')
# print_file.flush()


# conditions=[(g_dataframe['name_track']==i) for  i in targetframe['caseID_hash']]
# values=targetframe['Days too late']
#changed 
#del targetframe
gc.collect()
#targetframe=pd.DataFrame()


# g_dataframe['target']=np.select(conditions,values)
# g_dataframe['target'].replace(0,np.nan,inplace=True)

# def target(v):
#     try:
#         value=targetframe.loc[targetframe['caseID_hash'] == v, 'Days too late'].iloc[0]
#     except:
#         value=np.nan
#     return value
    
# g_dataframe['target']=g_dataframe.apply(lambda x: target(x['name_track']),axis=1)          



#add target to nodes in the graph!!
# g_dataframe['target']=np.zeros(len(g_dataframe))
# targetframe=pd.read_csv('targetframe.csv', dtype={'case:concept:name': str})
# for i in range(len(g_dataframe)):
#     if g_dataframe['e_v'].iloc[i]=='' or g_dataframe['e_v'].iloc[i]=='XP':
#         g_dataframe['target'].iloc[i]=np.nan
#     elif g_dataframe['e_v'].iloc[i]=='e':
#         pass     
#     else:
#         #give value in target column where case id equals value of name_track
#         g_dataframe['target'].iloc[i]=targetframe.loc[targetframe['case:concept:name'] == g_dataframe['name_track'].iloc[i], 'Days too late'].iloc[0]


#add the "trace_time" "prev_event_time" to g_dataframe

#dictionary that contains the list of parameters as key and 0 or 1 to mark wich
#parameters have to be normalized.
# 0 = no actions this features has to be added to blacklist at the end of the script (except for the first four),
# 1 = continuous attribute,
# 2 = categorical attribute
# IMPORTANT! e_v; node1; node2; name_event are structural feature. These haven't be added to blacklist. These four must be the same for every dataset!!!
#<HARDCODED> Adattare il dizionario in base agli attributi del dataset
#<HARDCODED> Adapt the dictionary to the set of attributes of the dataset

#own p2plog



'''

In base a cosa imposto 0 1 2 ?
come posso gestirlo?

'''
to_norm = {'e_v': 0, 'node1': 0, 'node2': 0, 'name_event': 0, 'name_track':0,
                    'finish': 0,
                    'start': 0, 'norm_time': 1, 'trace_time': 1,
                    'prev_event_time': 1, 'target':0}

#why not onehot encode the activities??

#</HARDCODED>

# does the normalization and the encoding
#bool var that must be true if cat_features are selected to be encoded
cat_encoding = False
for i in range(len(to_norm.values())):
  if list(to_norm.values())[i] == 2 and cat_encoding is False:
    cat_encoding = True

print_file.write('Start with normalizing andreas features...\n')
print_file.flush()
for i in ['norm_time','trace_time','prev_event_time']:  # Normalizzazione delle feature norm_time - trace_time - prev_event_time

    g_dataframe[i] = g_dataframe[i].div(g_dataframe[i].max()).round(15)

if cat_encoding is True:
  #name_track casted to string
  #HARDCODED castare a string tutti gli attributi interi categorici
  #HARDCODED cast to string all the set of integer categorical attributes
  #<hardcoded>
  # <hardcoded> COMMENTARE QUI per farlo funzionare su un altro dataset; attributo specifico di DBI12
  # <hardcoded> here you must write the attributes
  cast_to_string = ['name_track']  # bpi12
  cast_to_string = ['name_track', 'name_event', 'variant-index', 'ActivityID']  # helpdesk
  #se te lo perdi ti da errore con la key e ti dice che non può eseguire l'op su una non stringa
  # if you miss this it is hard.
  #</hardcoded>
  for item in cast_to_string:
    g_dataframe[item] = g_dataframe[item].astype(str)
  #list of lists with unique names in the columns
  list_unici = []
  #list of lists with swapped values between value and his index in the list
  list_categorical = []
  for i in range(len(to_norm.values())):
    if list(to_norm.values())[i] == 2: #set up a list of lists for 
                                        #categorical values
      #creates 2 equal lists, one for unique names, the other for indexes
      list_unici.append(list(g_dataframe[list(to_norm.keys())[i]].unique()))                                   
      list_categorical.append(list(g_dataframe[list(to_norm.keys())[i]].unique()))
      for x in list_categorical:
        #delete 'nan' value from first element of each list
        if str(x[0]) == 'nan':
          del x[0]
        for y in x: #replace value with his index in order to have values from 1
                    #to n (to make 'to_categorical' works)
          x[x.index(y)] = x.index(y)
  #delete first element of each list because it is 'nan'
  for i in list_unici:
    del i[0]

  ohe_list = []

  for x in list_categorical: # normalize for each of the cat. attr. chosen 
    ohe_list.append(to_categorical(x, len(x), dtype=int))

  def find_index(list_unici, elem):
    index = list_unici.index(elem)
    return index

  #counter used to sync the two lists above
  ctr = 0
  for i in range(len(to_norm.values())):
    if list(to_norm.values())[i] == 2:
      all_values = g_dataframe[(list(to_norm.keys())[i])]
      #list to ensure to not look for already swapped values (otherwise it'll throw errors)
      swapped = []
      for k in g_dataframe[(list(to_norm.keys())[i])]:
        k = str(k)  
        if k != 'nan' and k not in swapped:
          swapped.append(k)
          index = find_index(list_unici[ctr], k)
          tmp = (ohe_list[ctr].tolist())[index]
          swapped.append(str(tmp).replace(",",""))

          #swap the g_dataframe value with the encoded one
          g_dataframe[list(to_norm.keys())[i]].replace({k: str(tmp).replace(",","")}, inplace = True)
      ctr += 1
      
  g_dataframe['name_track'] = g_dataframe['name_track'].replace('nan', '')


#casting time column as string
g_dataframe[['finish', 'start']] = g_dataframe[['finish', 'start']].astype(str)
#add blank row before XPs
g_dataframe['e_v'].replace('XP', '\nXP',inplace=True)


#-------------- DA PARAMETRIZZARE --------------------------------------

#TOGLIERE LE DATE

#remove all null cells in all data's field
# g_dataframe['finish'] = g_dataframe[['finish','e_v']].apply(lambda x: '' if (x['e_v']== '' or x['e_v']== 'e' or x['e_v']== 'XP') else x['finish'], axis=1)
# g_dataframe['start'] = g_dataframe[['start','e_v']].apply(lambda x: '' if (x['e_v']== '' or x['e_v']== 'e' or x['e_v']== 'XP') else x['start'], axis=1)
# g_dataframe['norm_time'] = g_dataframe[['norm_time','e_v']].apply(lambda x: '' if (x['e_v']== '' or x['e_v']== 'e' or x['e_v']== 'XP') else x['norm_time'], axis=1)
# g_dataframe['trace_time'] = g_dataframe[['trace_time','e_v']].apply(lambda x: '' if (x['e_v']== '' or x['e_v']== 'e' or x['e_v']== 'XP') else x['trace_time'], axis=1)
# g_dataframe['prev_event_time'] = g_dataframe[['prev_event_time','e_v']].apply(lambda x: '' if (x['e_v']== '' or x['e_v']== 'e' or x['e_v']== 'XP') else x['prev_event_time'], axis=1)
# g_dataframe['target'] = g_dataframe[['target','e_v']].apply(lambda x: '' if (x['e_v']== '' or x['e_v']== 'e' or x['e_v']== 'XP') else x['target'], axis=1)
# #reorder columns
# g_dataframe = g_dataframe[g_dataframe.columns]

# lista_param_eventi
#sobstitute all null as ''
#changed!!
g_dataframe.fillna('',inplace=True)
g_dataframe.replace({'NaT': ''}, inplace=True)

#recompose the string for the final .g file

# tmp=''
#<HARDCODED>
# <hardcoded> COMMENTARE QUI per farlo funzionare su un altro dataset; attributo specifico di DBI12
#<HARDCODED> what you write in the black list is what it will be eliminated as attributes in the output.
blacklist=['finish', 'start', 'name_track']

# blacklist=['finish','REG_finish','start','transition','name_track'] # 4 bpi12

# blacklist = ['creator', 'variant', 'variant-index','ActivityID','name_track','resource', 'finish', 'start', 'transition'] # 4 helpdesk
# blacklist = ['name_track', 'finish', 'start'] # 4 BPI2020
#</HARDCODED>
#print(g_dataframe)
g_dataframe.drop(columns=blacklist, axis = 1, inplace = True)


print('start writing the complete.g file ...')
# for index, row in g_dataframe.iterrows():
#   for item in g_dataframe.columns:
#     tmp+=str(row[item])+' '
#   tmp = tmp.strip()
#   tmp+='\n'
# tmp = tmp.replace("nan", "")

#changed !!
print_file.write('Start writing the complete.g file...\n')
print_file.flush()
g_dataframe[list(g_dataframe.columns)]=g_dataframe[list(g_dataframe.columns)].astype(str)
g_dataframe['tmp']=g_dataframe[list(g_dataframe.columns)].T.agg(' '.join)
g_dataframe['tmp']=g_dataframe['tmp'].str.strip()
tmp=g_dataframe['tmp'][1:].str.cat(sep='\n')
tmp+='\n'
tmp=tmp.replace("nan", "")

#OUTPUT
print_file.write('Actual writing the complete.g file...\n')

print_file.close()
w = open(path_write, "w") 
w.writelines(tmp)
w.close()




def get_gDataFrame():
  g_dataframe_tmp = g_dataframe.iloc[:, 4:-1]  # Seleziona tutte le righe (:), dalla terza colonna in poi (2:)
  return g_dataframe_tmp


'''
Esempio di cosa restituisce get_gDataFrame senza nessuna colonna selezionata dall'utente:

           norm_time trace_time prev_event_time
0                                              
1                                              
2  0.285643417005655        0.0             0.0
3  0.285643417005655        0.0             0.0
4  0.285742633197738        0.0             0.0


'''