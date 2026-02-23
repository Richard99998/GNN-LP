import argparse
import os

import numpy as np
import torch
from pandas import read_csv

from models import GCNPolicy


parser = argparse.ArgumentParser()
parser.add_argument("--data", help="number of training data", default=1000)
parser.add_argument("--gpu", help="gpu index", default="0")
parser.add_argument("--embSize", help="embedding size of GNN", default="6")
parser.add_argument("--epoch", help="num of epoch", default="10000")
parser.add_argument("--type", help="what's the type of the model", default="fea", choices=["fea", "obj", "sol"])
args = parser.parse_args()


def process(model, dataloader, optimizer, task_type="fea"):
    c, ei, ev, v, n_cs, n_vs, n_csm, n_vsm, cand_scores = dataloader
    batched_states = (c, ei, ev, v, n_cs, n_vs, n_csm, n_vsm)

    optimizer.zero_grad()
    logits = model(batched_states, training=True)
    loss = torch.mean((cand_scores - logits) ** 2)
    loss.backward()
    optimizer.step()

    return_loss = loss.item()
    errs = None
    err_rate = None

    if task_type == "fea":
        errs_fp = torch.sum((logits > 0.5) & (cand_scores < 0.5)).item()
        errs_fn = torch.sum((logits < 0.5) & (cand_scores > 0.5)).item()
        errs = errs_fp + errs_fn
        err_rate = errs / cand_scores.shape[0]

    return return_loss, errs, err_rate


max_epochs = int(args.epoch)
lr = 0.0003
seed = 0

trainfolder = "./data-training/"
n_Samples = int(args.data)
n_Cons_small = 10
n_Vars_small = 50
n_Eles_small = 100

embSize = int(args.embSize)
os.makedirs("./saved-models/", exist_ok=True)
model_path = f"./saved-models/{args.type}_d{n_Samples}_s{embSize}.pkl"

if args.type == "fea":
    varFeatures = read_csv(trainfolder + "VarFeatures_all.csv", header=None).values[: n_Vars_small * n_Samples, :]
    conFeatures = read_csv(trainfolder + "ConFeatures_all.csv", header=None).values[: n_Cons_small * n_Samples, :]
    edgFeatures = read_csv(trainfolder + "EdgeFeatures_all.csv", header=None).values[: n_Eles_small * n_Samples, :]
    edgIndices = read_csv(trainfolder + "EdgeIndices_all.csv", header=None).values[: n_Eles_small * n_Samples, :]
    labels = read_csv(trainfolder + "Labels_feas.csv", header=None).values[:n_Samples, :]
elif args.type == "obj":
    varFeatures = read_csv(trainfolder + "VarFeatures_feas.csv", header=None).values[: n_Vars_small * n_Samples, :]
    conFeatures = read_csv(trainfolder + "ConFeatures_feas.csv", header=None).values[: n_Cons_small * n_Samples, :]
    edgFeatures = read_csv(trainfolder + "EdgeFeatures_feas.csv", header=None).values[: n_Eles_small * n_Samples, :]
    edgIndices = read_csv(trainfolder + "EdgeIndices_feas.csv", header=None).values[: n_Eles_small * n_Samples, :]
    labels = read_csv(trainfolder + "Labels_obj.csv", header=None).values[:n_Samples, :]
else:
    varFeatures = read_csv(trainfolder + "VarFeatures_feas.csv", header=None).values[: n_Vars_small * n_Samples, :]
    conFeatures = read_csv(trainfolder + "ConFeatures_feas.csv", header=None).values[: n_Cons_small * n_Samples, :]
    edgFeatures = read_csv(trainfolder + "EdgeFeatures_feas.csv", header=None).values[: n_Eles_small * n_Samples, :]
    edgIndices = read_csv(trainfolder + "EdgeIndices_feas.csv", header=None).values[: n_Eles_small * n_Samples, :]
    labels = read_csv(trainfolder + "Labels_solu.csv", header=None).values[: n_Vars_small * n_Samples, :]

nConsF = conFeatures.shape[1]
nVarF = varFeatures.shape[1]
nEdgeF = edgFeatures.shape[1]
n_Cons = conFeatures.shape[0]
n_Vars = varFeatures.shape[0]

torch.manual_seed(seed)
gpu_index = int(args.gpu)
device = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")

varFeatures = torch.tensor(varFeatures, dtype=torch.float32, device=device)
conFeatures = torch.tensor(conFeatures, dtype=torch.float32, device=device)
edgFeatures = torch.tensor(edgFeatures, dtype=torch.float32, device=device)
edgIndices = torch.tensor(edgIndices, dtype=torch.int64, device=device).T
labels = torch.tensor(labels, dtype=torch.float32, device=device)
train_data = (conFeatures, edgIndices, edgFeatures, varFeatures, n_Cons, n_Vars, n_Cons_small, n_Vars_small, labels)

if args.type == "sol":
    model = GCNPolicy(embSize, nConsF, nEdgeF, nVarF, isGraphLevel=False)
else:
    model = GCNPolicy(embSize, nConsF, nEdgeF, nVarF)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_init, _, _ = process(model, train_data, optimizer, task_type=args.type)
epoch = 0
count_restart = 0
err_best = 2
loss_best = 1e10

while epoch <= max_epochs:
    train_loss, errs, err_rate = process(model, train_data, optimizer, task_type=args.type)

    if args.type == "fea":
        print(f"EPOCH: {epoch}, TRAIN LOSS: {train_loss}, ERRS: {errs}, ERRATE: {err_rate}")
        if err_rate < err_best:
            model.save_state(model_path)
            print("model saved to:", model_path)
            err_best = err_rate
    else:
        print(f"EPOCH: {epoch}, TRAIN LOSS: {train_loss}")
        if train_loss < loss_best:
            model.save_state(model_path)
            print("model saved to:", model_path)
            loss_best = train_loss

    if epoch == 200 and count_restart < 3 and (train_loss > loss_init * 0.8 or (err_rate is not None and err_rate > 0.5)):
        print("Fail to reduce loss, restart...")
        if args.type == "sol":
            model = GCNPolicy(embSize, nConsF, nEdgeF, nVarF, isGraphLevel=False)
        else:
            model = GCNPolicy(embSize, nConsF, nEdgeF, nVarF)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_init, _, _ = process(model, train_data, optimizer, task_type=args.type)
        epoch = 0
        count_restart += 1

    epoch += 1

print("Count of restart:", count_restart)
print(model)
