versione 1.1

Script funzionante e parametrizzato.
SearchGrid non integrata.

# HOW TO USE:

 ## Installazione Librerie
 Assicurarsi che tutte le librerie siano installate, le principali, con la loro versione, vengono installate con il comando:
  ```zsh 
 $ pip install -r requirements.txt
  ```

## Scelta file.xes
Nella cartella 'Input/xes' sono presenti diversi file .xes, per scegliere il file xes specifico bisogna andare su file config.py e a linea 39 in corrispondenza del '--xes_name' va inserito nel parametro default il nome del file .xes desiderato.

Una volta scelto quale file.xes startare lo script modificata_dgcnn_70_30_dataset_Manou_server.py

Durante l'elaborazione verrà generata una GUI (accade nel file dotG_enrichment_Manou_server.py) per la scelta delle feature da utilizzare.

Al termine dell'elaborazione, nella cartella log, viene generato il file .txt con i relativi risultati dell'addestramento.
