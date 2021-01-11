import argparse
import sys
from data_loader import *
from LSTMClique import LSTMClique
from LSTMSentAvg import LSTMSentAvg
from LSTMParSeq import LSTMParSeq
from train_neural_models import *

sys.path.insert(0,os.getcwd())

dirname, filename = os.path.split(os.path.abspath(__file__))
root_dir = "/".join(dirname.split("/")[:-1])

run_dir = os.path.join(root_dir, "runs")

parser = argparse.ArgumentParser()

# data
parser.add_argument("--task", type=str, default="class")  # class [classification], perm [binary permutation], score_pred [mean score prediction], minority [minority binary classification]

# model params
parser.add_argument("--model_type", type=str, default="clique") # clique, doc_seq
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--dropout", type=float, default=0)
parser.add_argument("--lstm_dim", type=int, default=100)
parser.add_argument("--hidden_dim", type=int, default=200, help="hidden layer dimension")
parser.add_argument("--clique", type=int, default=3) # number of sentences in each clique (clique model only)
parser.add_argument("--l2_reg", type=float, default=0)

# training
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--train_data_limit", type=int, default=-1) # for debugging with subset of data
parser.add_argument("--lr_decay", type=str, default="none")

# vectors
parser.add_argument("--vector_type", default="glove", help="specify vector type glove/word2vec/none")
parser.add_argument("--glove_path", type=str, default="data/GloVe/glove.840B.300d.txt")
parser.add_argument("--embedding_dim", type=int, default=300, help="vector dimension")
parser.add_argument("--case_sensitive", action="store_true", help="activate this flag if vectors are case-sensitive (don't lower-case the data)")

# per-experiment settings
parser.add_argument("--model_name", type=str)
parser.add_argument("--data_dir", default="data/", help="path to the data directory")
parser.add_argument("--train_corpus", type=str)
parser.add_argument("--test_corpus", type=str)


args = parser.parse_args()
if args.model_name is None:
    print("Specify name of experiment")
    sys.exit(0)
if args.train_corpus is None:
    print("Specify train corpus")
    sys.exit(0)
if args.test_corpus is None:
    args.test_corpus = args.train_corpus

params = {
    'top_dir': root_dir,
    'run_dir': run_dir,
    'model_name': args.model_name,
    'data_dir': args.data_dir,
    'train_corpus': args.train_corpus,
    'test_corpus': args.test_corpus,
    'task': args.task,
    'train_data_limit': args.train_data_limit,
    'lr_decay': args.lr_decay,
    'model_type': args.model_type,
    'glove_file': args.glove_path,
    'vector_type': args.vector_type,
    'embedding_dim': args.embedding_dim,  # word embedding dim
    'case_sensitive': args.case_sensitive,
    'learning_rate': args.learning_rate,
    'dropout': args.dropout,  # 1 = no dropout, 0.5 = dropout
    'hidden_dim': args.hidden_dim,
    'lstm_dim': args.lstm_dim,
    'clique_size': args.clique,
    'l2_reg': args.l2_reg,
    'batch_size': args.batch_size,
    'num_epochs': args.num_epochs,
}

if not os.path.exists(params['run_dir']):
    os.mkdir(params['run_dir'])
model_dir = os.path.join(params['run_dir'], params["model_name"])
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
params['model_dir'] = model_dir

# save parameters
with open(os.path.join(model_dir, params['model_name'] + '.params'), 'w') as param_file:
    for key, parameter in params.items():
        param_file.write("{}: {}".format(key, parameter) + "\n")
        print((key, parameter))

start = time.time()
if params['vector_type'] == 'glove':
    params['vector_path'] = params['glove_file']

# load data
data = Data(params)
vectors = None
if params['vector_type'] != 'none':
    vectors, vector_dim = data.load_vectors()
    params['embedding_dim'] = vector_dim

if params['task'] == 'class' or params['task'] == 'score_pred' or params['task'] == 'minority':
    training_docs = data.read_data_class(params, 'train')
    test_docs = data.read_data_class(params, 'test')
else:
    training_docs = data.read_data_perm(params, 'train')
    test_docs = data.read_data_perm(params, 'test')
# dev_docs = None
if params['vector_type'] == 'none':  # init random vectors
    vectors = data.rand_vectors(len(data.word_to_idx))

if params['model_type'] == 'clique':
    model = LSTMClique(params, data)
    train(params, training_docs, test_docs, data, model)
elif params['model_type'] == 'sent_avg':
    model = LSTMSentAvg(params, data)
    train(params, training_docs, test_docs, data, model)
elif params['model_type'] == 'par_seq':
    model = LSTMParSeq(params, data)
    train(params, training_docs, test_docs, data, model)
