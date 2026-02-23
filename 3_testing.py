import argparse

import torch
from pandas import read_csv

from models import GCNPolicy


parser = argparse.ArgumentParser()
parser.add_argument("--data", help="number of training data", default=1000)
parser.add_argument("--dataTest", help="number of test data", default=1000)
parser.add_argument("--gpu", help="gpu index", default="0")
parser.add_argument("--embSize", help="embedding size of GNN", default="16")
parser.add_argument("--type", help="what's the type of the model", default="fea", choices=["fea", "obj", "sol"])
parser.add_argument("--set", help="which set you want to test on?", default="train", choices=["test", "train"])
parser.add_argument("--loss", help="loss function used in testing", default="mse", choices=["mse", "l2"])
args = parser.parse_args()


def process(model, dataloader, task_type="fea", loss="mse", n_Vars_small=50):
    c, ei, ev, v, n_cs, n_vs, n_csm, n_vsm, cand_scores = dataloader
    batched_states = (c, ei, ev, v, n_cs, n_vs, n_csm, n_vsm)
    with torch.no_grad():
        logits = model(batched_states, training=False)

    if task_type == "fea":
        errs_fp = torch.sum((logits > 0.5) & (cand_scores < 0.5)).item()
        errs_fn = torch.sum((logits < 0.5) & (cand_scores > 0.5)).item()
        return (errs_fp + errs_fn) / cand_scores.shape[0]

    if task_type == "obj":
        if loss == "mse":
            return torch.mean((cand_scores - logits) ** 2).item()
        return torch.mean(torch.abs(cand_scores - logits) / (torch.abs(cand_scores) + 1.0)).item()

    if loss == "mse":
        return torch.mean((cand_scores - logits) ** 2).item()

    length_sol = logits.shape[0]
    cand_scores = cand_scores.view(int(length_sol / n_Vars_small), n_Vars_small)
    logits = logits.view(int(length_sol / n_Vars_small), n_Vars_small)
    err = torch.linalg.norm(cand_scores - logits, dim=1)
    norm = torch.linalg.norm(cand_scores, dim=1) + 1.0
    return torch.mean(err / norm).item()


datafolder = "./data-training/" if args.set == "train" else "./data-testing/"
n_Samples_test = int(args.dataTest)
n_Cons_small = 10
n_Vars_small = 50
n_Eles_small = 100

embSize = int(args.embSize)
n_Samples = int(args.data)
model_path = f"./saved-models/{args.type}_d{n_Samples}_s{embSize}.pkl"

if args.type == "fea":
    varFeatures = read_csv(datafolder + "VarFeatures_all.csv", header=None).values[: n_Vars_small * n_Samples_test, :]
    conFeatures = read_csv(datafolder + "ConFeatures_all.csv", header=None).values[: n_Cons_small * n_Samples_test, :]
    edgFeatures = read_csv(datafolder + "EdgeFeatures_all.csv", header=None).values[: n_Eles_small * n_Samples_test, :]
    edgIndices = read_csv(datafolder + "EdgeIndices_all.csv", header=None).values[: n_Eles_small * n_Samples_test, :]
    labels = read_csv(datafolder + "Labels_feas.csv", header=None).values[:n_Samples_test, :]
elif args.type == "obj":
    varFeatures = read_csv(datafolder + "VarFeatures_feas.csv", header=None).values[: n_Vars_small * n_Samples_test, :]
    conFeatures = read_csv(datafolder + "ConFeatures_feas.csv", header=None).values[: n_Cons_small * n_Samples_test, :]
    edgFeatures = read_csv(datafolder + "EdgeFeatures_feas.csv", header=None).values[: n_Eles_small * n_Samples_test, :]
    edgIndices = read_csv(datafolder + "EdgeIndices_feas.csv", header=None).values[: n_Eles_small * n_Samples_test, :]
    labels = read_csv(datafolder + "Labels_obj.csv", header=None).values[:n_Samples_test, :]
else:
    varFeatures = read_csv(datafolder + "VarFeatures_feas.csv", header=None).values[: n_Vars_small * n_Samples_test, :]
    conFeatures = read_csv(datafolder + "ConFeatures_feas.csv", header=None).values[: n_Cons_small * n_Samples_test, :]
    edgFeatures = read_csv(datafolder + "EdgeFeatures_feas.csv", header=None).values[: n_Eles_small * n_Samples_test, :]
    edgIndices = read_csv(datafolder + "EdgeIndices_feas.csv", header=None).values[: n_Eles_small * n_Samples_test, :]
    labels = read_csv(datafolder + "Labels_solu.csv", header=None).values[: n_Vars_small * n_Samples_test, :]

nConsF = conFeatures.shape[1]
nVarF = varFeatures.shape[1]
nEdgeF = edgFeatures.shape[1]
n_Cons = conFeatures.shape[0]
n_Vars = varFeatures.shape[0]

gpu_index = int(args.gpu)
device = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")

varFeatures = torch.tensor(varFeatures, dtype=torch.float32, device=device)
conFeatures = torch.tensor(conFeatures, dtype=torch.float32, device=device)
edgFeatures = torch.tensor(edgFeatures, dtype=torch.float32, device=device)
edgIndices = torch.tensor(edgIndices, dtype=torch.int64, device=device).T
labels = torch.tensor(labels, dtype=torch.float32, device=device)
data = (conFeatures, edgIndices, edgFeatures, varFeatures, n_Cons, n_Vars, n_Cons_small, n_Vars_small, labels)

if args.type == "sol":
    model = GCNPolicy(embSize, nConsF, nEdgeF, nVarF, isGraphLevel=False)
else:
    model = GCNPolicy(embSize, nConsF, nEdgeF, nVarF)
model = model.to(device)
model.restore_state(model_path, map_location=device)

err = process(model, data, task_type=args.type, loss=args.loss, n_Vars_small=n_Vars_small)
print(model)
print(f"MODEL: {model_path}, DATA-SET: {datafolder}, NUM-DATA: {n_Samples_test}, LOSS: {args.loss}, ERR: {err}")
