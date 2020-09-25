from __future__ import division
from __future__ import print_function

import math
import sys
import time
import argparse

import torch.optim as optim
from Modelmlp import MTPool
from utils import *
import torch.nn.functional as F

datasets = ["ArticularyWordRecognition", "CharacterTrajectories", "FaceDetection", "Heartbeat", "MotorImagery",
            "NATOPS", "PEMS-SF", "PenDigits", "SelfRegulationSCP2", "SpokenArabicDigits"]

parser = argparse.ArgumentParser()

# dataset settings
parser.add_argument('--data_path', type=str, default="./dataset/Preprocess/",
                    help='the path of data.')
parser.add_argument('--dataset', type=str, default="PEMS-SF",#PEMS-SF", #NATOPS
                    help='time series dataset. Options: See the datasets list')
parser.add_argument('--gnn', type=str, default="GNN",
                    help='GNN or GIN')
parser.add_argument('--relation', type=str, default="dynamic",
                    help='dynamic or corr')
parser.add_argument('--pooling', type=str, default="CoSimPool",
                    help='CoSimPool or DiffPool')

# cuda settings
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--use_cuda', type=int, default=0, help='cpu or gpu.')

# Training parameter settings
parser.add_argument('--epochs', type=int, default=10000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-5,
                    help='Initial learning rate. default:[0.00001]')
parser.add_argument('--wd', type=float, default=1e-3,
                    help='Weight decay (L2 loss on parameters). default: 5e-3')
parser.add_argument('--stop_thres', type=float, default=1e-9,
                    help='The stop threshold for the training error. If the difference between training losses '
                         'between epoches are less than the threshold, the training will be stopped. Default:1e-9')

args = parser.parse_args()
args.cuda = False#not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))

print("Loading dataset", args.dataset, "...")
# Model and optimizer
model_type = "MTPool"
if model_type == "MTPool":

    features, labels, idx_train, idx_val, idx_test, nclass \
                                    = load_raw_ts(args.data_path, dataset=args.dataset)

    print("Data shape:", features.size())
    model = MTPool(use_cuda=args.cuda,
    			   dataset_path=args.data_path,
                   dataset=args.dataset,
                   graph_method=args.gnn,
                   relation_method=args.relation,
                   pooling_method=args.pooling
                   )
    # cuda
    if args.cuda:
        model.cuda()
        features, labels, idx_train = features.cuda(), labels.cuda(), idx_train.cuda()
    input = (features, labels, idx_train, idx_val, idx_test)

# init the optimizer
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.wd)


# training function
def train():
    loss_list = [sys.maxsize]
    test_best_possible, best_so_far = 0.0, sys.maxsize
    for epoch in range(args.epochs):

        t = time.time()
        model.train()
        optimizer.zero_grad()

        output = model(input)

        loss_train = F.cross_entropy(output[idx_train], torch.squeeze(labels[:len(idx_train)]))
        loss_train = loss_train

        loss_list.append(loss_train.item())
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        output = model(input, test=True)
        loss_val = F.cross_entropy(output, torch.squeeze(labels[len(idx_train):]))
        acc_val = accuracy(output, labels[len(idx_train):])

        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.8f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'lv: {:.4f}'.format(loss_val.item()),
              'av: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))

        if acc_val.item() > test_best_possible:
            test_best_possible = acc_val.item()
        if best_so_far > loss_train.item():
            best_so_far = loss_train.item()
            test_acc = acc_val.item()
    print("test_acc: " + str(test_acc))
    print("best possible: " + str(test_best_possible))

# test function
def test():
    output = model(input,test=True)
    #print(output[idx_test])
    loss_test= F.cross_entropy(output, torch.squeeze(labels[len(idx_train):]))
    acc_test = accuracy(output, labels[len(idx_train):])
    print(args.dataset, "Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

# Train model
t_total = time.time()
train()
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
