import PySimpleGUI as sg

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