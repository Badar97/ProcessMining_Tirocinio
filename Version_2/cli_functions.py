import sys
import config
args = config.load()


def select_range_and_split(prefix_occurrences):
    min_prefissi = min(prefix_occurrences.keys())
    max_prefissi = max(prefix_occurrences.keys())

    print("Occorrenze dei prefissi:")
    for k, v in prefix_occurrences.items():
        print(f"Num. prefix : {k}, Occurences: {v}")

    try:
        min_prefissi_selezionato = int(input(f"Inserisci il valore minimo del range di prefissi (min: {min_prefissi},"
                                             f"max: {max_prefissi}): "))
        max_prefissi_selezionato = int(input(f"Inserisci il valore massimo del range di prefissi (min: {min_prefissi},"
                                             f"max: {max_prefissi}): "))
        percentuale_split = int(input("Inserisci la percentuale di split train/test: "))
        search_grid = input("Vuoi attivare la Search Grid? (Y/N): ").strip().lower() == "y"
    except (Exception,):
        print('Valori non inseriti.')
        sys.exit()

    if min_prefissi_selezionato > max_prefissi_selezionato:
        print("Il valore minimo del range di prefissi non può essere maggiore del massimo.")
        sys.exit(1)

    print(f"Range di prefissi selezionato: {min_prefissi_selezionato} a {max_prefissi_selezionato}")
    print(f"Percentuale di split train/test: {percentuale_split}%")
    print(f"Search Grid: {'Attivata' if search_grid else 'Disattivata'}")

    return min_prefissi_selezionato, max_prefissi_selezionato, percentuale_split, search_grid


def select_features(colonne_da_sel, colonne_eliminate, parametro_bool):
    # Opzioni per la selezione
    opzioni = ["Next Activity Prediction"]
    if parametro_bool:
        opzioni.insert(0, "Regressione")

    print("Seleziona le features:")
    for i, col in enumerate(colonne_da_sel):
        print(f"{i}. {col}")
    selezione_da_sel = input("Inserisci gli indici delle colonne da selezionare, separati da virgola (es. 0,2,4): ")
    colonne_selezionate_da_sel = [colonne_da_sel[int(i)] for i in selezione_da_sel.split(',') if i.isdigit()]

    print("\nFeatures sconsigliate (per mancanza dati):")
    for i, col in enumerate(colonne_eliminate):
        print(f"{i}. {col}")
    selezione_elim = input("Inserisci gli indici delle colonne eliminate da selezionare,"
                           "separati da virgola (es. 0,1): ")
    colonne_selezionate_elim = [colonne_eliminate[int(i)] for i in selezione_elim.split(',') if i.isdigit()]

    print("\nScegli un'opzione:")
    for i, opzione in enumerate(opzioni):
        print(f"{i}. {opzione}")
    scelta_opzione = input("Inserisci l'indice dell'opzione scelta: ")
    opzione_scelta = opzioni[int(scelta_opzione)]

    colonne_selezionate = colonne_selezionate_da_sel + colonne_selezionate_elim
    tutte_le_colonne = colonne_da_sel + colonne_eliminate
    non_selezionate = list(set(tutte_le_colonne) - set(colonne_selezionate))

    print(f"\nColonne selezionate: {colonne_selezionate}")
    print(f"Colonne non selezionate: {non_selezionate}")
    print(f"Opzione scelta: {opzione_scelta}")
    train_rete = False  # Questo valore rimane invariato come nell'esempio GUI

    conferma = input("Confermi la selezione? (s/n): ")
    if conferma.lower() != 's':
        sys.exit("Selezione annullata.")

    return colonne_selezionate, non_selezionate, opzione_scelta, train_rete


def select_dataset_paths():
    dataset_path = input("Inserisci il percorso del file del dataset pre-elaborato (complete_par.pt): ")
    target_par = input("Inserisci il percorso del file target_par.txt: ")
    target_std = input("Inserisci il percorso del file target_std.txt: ")

    if target_par == "":
        target_par = args.target_par

    if target_std == "":
        target_std = args.target_std

    if dataset_path == "":
        dataset_path = args.complete_par

    print(f"Percorso dataset selezionato: {dataset_path}")
    print(f"Percorso file target_par.txt selezionato: {target_par}")
    print(f"Percorso file target_std.txt selezionato: {target_std}")

    return dataset_path, target_par, target_std
