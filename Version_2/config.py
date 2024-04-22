from os import makedirs
from os.path import join, abspath, exists, dirname
from shutil import rmtree
from argparse import ArgumentParser

LOGS_WITH_NO_START_END = ['BPI12']

BASE_PATH = dirname((abspath(__file__)))
INPUT_PATH = join(BASE_PATH, 'Input')
INPUT_XES_PATH = join(INPUT_PATH, 'xes')
INPUT_G_PATH = join(INPUT_PATH, 'g')

OUTPUT_PATH = join(BASE_PATH, 'Output')
OUTPUT_DS_PATH = join(OUTPUT_PATH, 'dataset')
OUTPUT_PN_PATH = join(OUTPUT_PATH, 'petri_nets')

PREFIX_PATH = join(BASE_PATH, 'Prefix')

# to create
LOG_PATH = join(BASE_PATH, 'log')
CSV_PATH = join(BASE_PATH, 'Csv')
IMG_PATH = join(BASE_PATH, 'IMG')
IMMAGINI_PATH = join(BASE_PATH, 'Immagini')
IMG_CM_EPOCH = join(IMMAGINI_PATH, 'cm_epoch')
IMG_BEST_TEST_PATH = join(IMMAGINI_PATH, 'best_test')
IMG_BEST_TRAIN_PATH = join(IMMAGINI_PATH, 'best_train')
CM_PATH = join(IMG_PATH, 'cm_epoch')
CHECKPOINT_RETE_PATH = join(BASE_PATH, 'checkpoint_rete')
CK_BEST_TRAIN_PATH = join(CHECKPOINT_RETE_PATH, 'best_train')
CK_BEST_TEST_PATH = join(CHECKPOINT_RETE_PATH, 'best_test')
F1_SCORE_PATH = join(BASE_PATH, 'AndamentoF1Score')


# pulizia delle directory
def clean_directories():
    # input_to_clean = ['csv', 'g', 'testg']
    input_to_clean = ['csv', 'g']
    # recreate input folders
    for to_clean in input_to_clean:
        complete_path = join(INPUT_PATH, to_clean)
        if exists(complete_path):
            rmtree(complete_path)
        makedirs(complete_path)

    output_to_clean = ['Output']
    for to_clean in output_to_clean:
        complete_path = join(OUTPUT_PATH, to_clean)
        if exists(complete_path):
            rmtree(complete_path)

    output_to_create = ['petri_nets', 'Pre-cage',
                        join('dataset', 'processed'),
                        join('checkpoints', 'best_test'),
                        join('checkpoints', 'best_train'),
                        join('checkpoints', 'immagini', 'best_test'),
                        join('checkpoints', 'immagini', 'best_train'),
                        join('checkpoints', 'immagini', 'cm_epoch'),
                        'register'
                        ]
    for to_create in output_to_create:
        complete_path = join(OUTPUT_PATH, to_create)
        if not exists(complete_path):
            makedirs(complete_path)

        if exists(PREFIX_PATH):
            rmtree(PREFIX_PATH)
        makedirs(PREFIX_PATH)

    bla = [LOG_PATH, CSV_PATH, IMG_PATH, IMMAGINI_PATH, IMG_CM_EPOCH,
           IMG_BEST_TEST_PATH, IMG_BEST_TRAIN_PATH, CM_PATH, CHECKPOINT_RETE_PATH,
           CK_BEST_TRAIN_PATH, CK_BEST_TEST_PATH, F1_SCORE_PATH]

    for complete_path in bla:
        if exists(complete_path):
            rmtree(complete_path)
        makedirs(complete_path)


def load():
    parser = ArgumentParser()

    parser.add_argument('--attr_list',
                        default=['bu', 'nrchanges', 'ttmotif', 'plant', 'matgroup', 'polines', 'vendor', 'item'])
    parser.add_argument('--data_dir', default=join(OUTPUT_PATH, 'dataset'))
    parser.add_argument('--csv_name', default='testCsv.csv')
    parser.add_argument('--xes_name',  default='testXes.xes')
    parser.add_argument('--net_name', default=join(OUTPUT_PATH, 'petri_nets'))
    parser.add_argument('--checkpoint_dir', default=join(OUTPUT_PATH, 'checkpoints'))
    parser.add_argument('--dataset', default='p2p')
    parser.add_argument('--nrcases', default=4000, type=int)

    # training
    parser.add_argument('--patience', default=10, type=int, help="nr of epochs with no improvement (early stopping)")
    parser.add_argument('--per', default=67, type=int, help='percentage training data for train-test split')
    # model
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--num_runs', default=1, type=int)
    parser.add_argument('--num_neurons', default=100, type=int)
    parser.add_argument('--lr_run', default=1, type=int, help="0 lr=1e-03, 1 lr=1e-04, 2 lr=1e-05")
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--batch_size_valid', default=64, type=int)
    parser.add_argument('--k', default=7, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--num_layers', default=3, type=int)

    # dataset pre-elaborato
    parser.add_argument('--complete_par', default=join(OUTPUT_DS_PATH, 'processed', 'complete_par.pt'), type=str)
    parser.add_argument('--target_par', default=join(OUTPUT_DS_PATH, 'target_par.txt'), type=str)
    parser.add_argument('--target_std', default=join(OUTPUT_DS_PATH, 'target_std.txt'), type=str)

    args = parser.parse_args()
    return args
