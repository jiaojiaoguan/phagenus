from get_dataset import get_loader
from model import Transformer
import torch
import torch.nn as nn
import torch.optim as optim
from test import check_accuracy
import torch.distributed
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import time

def save_checkpoint(state, filename):
    torch.save(state, filename)

# Hyperparameters
src_vocab_size = 70215
#labels
tar_vocab_size = 493
max_length = 300
learning_rate = 0.001
num_workers = 6
batch_size = 256
num_epochs = 200
dropout = 0.5
num_layers = 1
load_model = False

root_folder = "./"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
train_loader = get_loader(
    root_folder=root_folder,
    annotation_file="train_protein.csv",
    batch_size=batch_size,
    shuffle=True,
    drop=True,
    num_workers=num_workers,
    buqi=src_vocab_size-1,
    max_length=max_length

)

val_loader = get_loader(
    root_folder=root_folder,
    annotation_file="val_protein.csv",
    batch_size=batch_size,
    num_workers=num_workers,
    buqi=src_vocab_size-1,
    max_length=max_length

)
test_loader = get_loader(
    root_folder=root_folder,
    annotation_file="test_protein.csv",
    batch_size=batch_size,
    num_workers=num_workers,
    buqi=src_vocab_size-1,
    max_length=max_length

)

# Initialize network
model = Transformer(
    src_vocab_size=src_vocab_size,
    tar_vocab_size=tar_vocab_size,
    max_length=max_length,
    dropout=dropout,
    num_layers=num_layers,
    heads=12,
    embed_size=768,

)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if torch.cuda.device_count() > 1:
    print(f'Use {torch.cuda.device_count()} GPUs!\n')
    model = nn.DataParallel(model)
model.to(device)

# Load model
if load_model:
    print(f"Loading checkpoint...model.pth.tar")
    checkpoint = torch.load(root_folder+"model_protein_transformer_combine.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    # max_accuracy = check_accuracy(loader=val_loader, model=model, device=device, task='val',save_result_path="")

else:
    max_accuracy = 0
    max_f1=0

y_true = []
y_pred = []
val_acc_all = []

# Train Network
for epoch in range(num_epochs):
    t=time.time()
    losses = []

    for batch_idx, (data, targets,contig) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)
        # Forward
        scores = model(data)
        _, predictions = scores.max(dim=1)
        loss = criterion(scores, targets)

        #ece loss
        # loss += masked_ECE(scores, targets)

        losses.append(loss.item())
        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient descent or adam step
        optimizer.step()

        for i in torch.arange(0, targets.shape[0]):
            y_true.append(targets[i].cpu().numpy())
            y_pred.append(predictions[i].cpu().numpy())

    mean_loss = sum(losses) / len(losses)
    f1 = f1_score(y_true, y_pred, average='macro')
    acc_train = accuracy_score(y_true, y_pred)
    f1_val,acc_val = check_accuracy(loader=val_loader, model=model, device=device, task='val',save_result_path="")

    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(mean_loss),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

    if acc_val > max_accuracy:
        max_accuracy = acc_val
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(state=checkpoint, filename=root_folder+'model_protein_transformer_combine4_label_acc_maxlength_300.pth.tar')