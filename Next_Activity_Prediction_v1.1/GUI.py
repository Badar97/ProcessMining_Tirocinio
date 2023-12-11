import PySimpleGUI as sg


"""
def SimpleGui(clms):

    selezionate = None
    colonne = clms
    # Lista delle colonne
    # colonne = ["Colonna 1", "Colonna 2", "Colonna 3", "Colonna 4", "Colonna 5"]

    layout = [
        [sg.Text("Seleziona le colonne da salvare:")],
        [sg.Listbox(colonne, select_mode=sg.LISTBOX_SELECT_MODE_MULTIPLE, size=(20, 6), key="-COLONNE-")],
        [sg.Button("Esegui"), sg.Button("Annulla")]
    ]

    window = sg.Window("Seleziona Colonne", layout)

    while True:
        event, values = window.read()

        if event == sg.WINDOW_CLOSED or event == "Annulla":
            break
        elif event == "Esegui":
            selezionate = values["-COLONNE-"]
            sg.popup(f"Colonne selezionate: {selezionate}")
            break

    window.close()
    return selezionate
"""

def SimpleGui(colonne_da_sel, colonne_eliminate):
    col_elim_str = '\n'.join(colonne_eliminate)
    layout = [
        [sg.Text("Seleziona le colonne da salvare:")],
        [sg.Listbox(colonne_da_sel, select_mode=sg.LISTBOX_SELECT_MODE_MULTIPLE, size=(30, 10), key="-COLONNE_DA_SEL-", enable_events=True)],
        [sg.Text("Colonne non selezionabili:")],
        [sg.Multiline(col_elim_str, size=(30, 10), key="-COLONNE_ELIMINATE-", text_color="red")],
        [sg.Button("Esegui"), sg.Button("Annulla")]
    ]

    window = sg.Window("Selezione Colonne", layout)

    while True:
        event, values = window.read()

        if event == sg.WINDOW_CLOSED or event == "Annulla":
            break
        elif event == "Esegui":
            colonne_selezionate = values["-COLONNE_DA_SEL-"]
            non_selezionate = list(set(colonne_da_sel) - set(colonne_selezionate))
            colonne_eliminate += non_selezionate
            col_elim_str = '\n'.join(colonne_eliminate)

            # Abilita la visualizzazione delle colonne eliminate
            window["-COLONNE_ELIMINATE-"].update(value=col_elim_str)

            sg.popup(f"Colonne selezionate: {colonne_selezionate}\n\nColonne eliminate: {colonne_eliminate}")

            break

    window.close()
    return colonne_selezionate, non_selezionate
