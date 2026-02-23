import argparse
import os

import torch
from pandas import read_csv

from models import GCNPolicy


parser = argparse.ArgumentParser()
parser.add_argument("--dataTest", help="number of test data", default=1000)
parser.add_argument("--gpu", help="gpu index", default="0")
parser.add_argument("--embSize", help="embedding size of GNN", default=None)
parser.add_argument("--type", help="what's the type of the model", default="fea", choices=["fea", "obj", "sol"])
parser.add_argument("--set", help="which set you want to test on?", default="train", choices=["test", "train"])
parser.add_argument("--loss", help="loss function used in testing", default="mse", choices=["mse", "l2"])
parser.add_argument("--folderModels", help="the folder of the saved models", default="./saved-models/")
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

device = torch.device(f"cuda:{int(args.gpu)}" if torch.cuda.is_available() else "cpu")

model_list = []
for model_name in os.listdir(args.folderModels):
    model_path = args.folderModels + model_name
    model_type = model_name.split("_")[0]
    if model_type != args.type:
        continue
    n_Samples = int(model_name.split("_")[1][1:]) if args.set == "train" else n_Samples_test
    embSize = int(model_name.split("_")[2][1:-4])
    if args.embSize is not None and embSize != int(args.embSize):
        continue
    model_list.append((model_path, embSize, n_Samples))

if args.type == "fea":
    varFeatures_np = read_csv(datafolder + "VarFeatures_all.csv", header=None).values
    conFeatures_np = read_csv(datafolder + "ConFeatures_all.csv", header=None).values
    edgFeatures_np = read_csv(datafolder + "EdgeFeatures_all.csv", header=None).values
    edgIndices_np = read_csv(datafolder + "EdgeIndices_all.csv", header=None).values
    labels_np = read_csv(datafolder + "Labels_feas.csv", header=None).values
elif args.type == "obj":
    varFeatures_np = read_csv(datafolder + "VarFeatures_feas.csv", header=None).values
    conFeatures_np = read_csv(datafolder + "ConFeatures_feas.csv", header=None).values
    edgFeatures_np = read_csv(datafolder + "EdgeFeatures_feas.csv", header=None).values
    edgIndices_np = read_csv(datafolder + "EdgeIndices_feas.csv", header=None).values
    labels_np = read_csv(datafolder + "Labels_obj.csv", header=None).values
else:
    varFeatures_np = read_csv(datafolder + "VarFeatures_feas.csv", header=None).values
    conFeatures_np = read_csv(datafolder + "ConFeatures_feas.csv", header=None).values
    edgFeatures_np = read_csv(datafolder + "EdgeFeatures_feas.csv", header=None).values
    edgIndices_np = read_csv(datafolder + "EdgeIndices_feas.csv", header=None).values
    labels_np = read_csv(datafolder + "Labels_solu.csv", header=None).values

for model_path, embSize, n_Samples in model_list:
    varFeatures = torch.tensor(varFeatures_np[: n_Vars_small * n_Samples, :], dtype=torch.float32, device=device)
    conFeatures = torch.tensor(conFeatures_np[: n_Cons_small * n_Samples, :], dtype=torch.float32, device=device)
    edgFeatures = torch.tensor(edgFeatures_np[: n_Eles_small * n_Samples, :], dtype=torch.float32, device=device)
    edgIndices = torch.tensor(edgIndices_np[: n_Eles_small * n_Samples, :], dtype=torch.int64, device=device).T
    if args.type == "sol":
        labels = torch.tensor(labels_np[: n_Vars_small * n_Samples, :], dtype=torch.float32, device=device)
    else:
        labels = torch.tensor(labels_np[:n_Samples, :], dtype=torch.float32, device=device)

    nConsF = conFeatures.shape[1]
    nVarF = varFeatures.shape[1]
    nEdgeF = edgFeatures.shape[1]
    n_Cons = conFeatures.shape[0]
    n_Vars = varFeatures.shape[0]
    data = (conFeatures, edgIndices, edgFeatures, varFeatures, n_Cons, n_Vars, n_Cons_small, n_Vars_small, labels)

    if args.type == "sol":
        model = GCNPolicy(embSize, nConsF, nEdgeF, nVarF, isGraphLevel=False)
    else:
        model = GCNPolicy(embSize, nConsF, nEdgeF, nVarF)
    model = model.to(device)
    model.restore_state(model_path, map_location=device)

    err = process(model, data, task_type=args.type, loss=args.loss, n_Vars_small=n_Vars_small)
    print(f"MODEL: {model_path}, DATA-SET: {datafolder}, NUM-DATA: {n_Samples}, LOSS: {args.loss}, ERR: {err}")
