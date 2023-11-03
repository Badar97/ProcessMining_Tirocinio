 # Guida d'utilizzo
 ### Tirocinio Process Mining a.a 2022/2023
 #### Badar Ali Waqar - Marco Romanelli
 
 
 ## Installazione Librerie
 Assicurarsi che tutte le librerie siano installate, le principali, con la loro versione, vengono installate con il comando:
  ```zsh 
 $ pip install -r requirements.txt
  ```

 ## Avvio di NewBig
 Eseguire `Copy_of_NewBig_server.py` il quale prende in input un file .xes presente in `./Input/xes/`, e lo converte in csv, inoltre crea altri file necessari per i step successivi:
 - le reti di petri: `dgcnn_log_sample_net.pnml`, `dgcnn_log_sample_net_startend.pnml`
 - gli Instance Graph: creati attraverso l'algoritmo BIG
 

 ## Enrichment
 Eseguire il file `dotG_enrichment.py` il quale prende in input gli instance graph presenti in `./Input/g/` li unisce in `merged.g` e successivamente esegue l'arricchimento con gli attributi necessari salvandoli in `complete.g`.
