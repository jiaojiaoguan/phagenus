#coding=utf-8

from get_dataset import get_loader
from model import Transformer
import torch.nn as nn
import torch.optim as optim
from get_test_result import check_accuracy_dropout
import torch.distributed
import argparse
from preprocessing import preprocessing_data

parser = argparse.ArgumentParser(description="""Phagenus is a python library for bacteriophages genus-level classification.
                               It is a transformer-based model and rely on protein-based vocabulary to convert DNA sequences into sentences for prediction.""")

parser.add_argument('--contigs', help='FASTA file of contigs',  default = 'test_contigs.fa')
parser.add_argument('--out', help='name of the output file',  type=str, default = 'prediction_output.csv')
parser.add_argument('--len', help='minimun length of contigs', type=int, default=3000)
parser.add_argument('--threads', help='threads of Diamond', type=str, default='2')
parser.add_argument('--reject', help='threshold to reject prophage',  type=float, default = 0.035)
parser.add_argument('--midfolder', help='folder to store the intermediate files', type=str, default='midfolder/')
parser.add_argument('--batch_size',help="batch size",type=int,default=64)
parser.add_argument('--lr',help='learning rate',type=float,default=0.001)
parser.add_argument('--num_workers',help='number worker',type=int,default=2)
parser.add_argument('--dropout',help="dropout",type=float,default=0.5)
parser.add_argument("--sim",help="input the high similarity or low similarty",type=str,default="high")

inputs = parser.parse_args()

input_data=inputs.contigs
out_fn = inputs.midfolder
similarity=inputs.sim
length=inputs.len
threads=inputs.threads

#processing the inputdata and convert the contig into protein cluster sentences
preprocessing_data(input_data,out_fn,similarity,length,threads)

#using the Phagenus to predict
if similarity=="high":
    src_vocab_size = 70175
elif similarity=="low":
    src_vocab_size = 71356

tar_vocab_size = 508
max_length = 300
num_layers = 1

load_model = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
test_loader = get_loader(
    annotation_file=f"{out_fn}/test_protein_sentence.csv",
    batch_size=inputs.batch_size,
    num_workers=inputs.num_workers,
    length_match=src_vocab_size-1,
    max_length=max_length
)

# Initialize network
model = Transformer(
    src_vocab_size=src_vocab_size,
    tar_vocab_size=tar_vocab_size,
    max_length=max_length,
    dropout=inputs.dropout,
    num_layers=num_layers,
    heads=12,
    embed_size=768,
)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=inputs.lr)

if torch.cuda.device_count() > 1:
    print(f'Use {torch.cuda.device_count()} GPUs!\n')
    model = nn.DataParallel(model)
model.to(device)

print(f"Loading checkpoint...model.pth.tar")
checkpoint = torch.load("data/model_protein_transformer_distance_30_combine_label.pth.tar")
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])

_ = check_accuracy_dropout(loader=test_loader, model=model,midfolder=out_fn+"/",var_cutoff=inputs.reject,output=inputs.out)
print("*******************")
print("*****ALL DONE!*****")
print("*******************")
