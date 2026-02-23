import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F


class BipartiteGraphConvolution(nn.Module):
    """Partial bipartite graph convolution (either left-to-right or right-to-left)."""

    def __init__(self, emb_size, right_to_left=False):
        super().__init__()

        self.emb_size = emb_size
        self.right_to_left = right_to_left

        self.feature_module_left = nn.Sequential(
            nn.LazyLinear(self.emb_size, bias=True),
            nn.ReLU(),
            nn.Linear(self.emb_size, self.emb_size),
        )
        self.feature_module_edge = nn.Identity()
        self.feature_module_right = nn.Sequential(
            nn.LazyLinear(self.emb_size, bias=False),
            nn.ReLU(),
            nn.Linear(self.emb_size, self.emb_size),
        )
        self.output_module = nn.Sequential(
            nn.LazyLinear(self.emb_size),
            nn.ReLU(),
            nn.Linear(self.emb_size, self.emb_size),
        )

    def forward(self, inputs):
        left_features, edge_indices, edge_features, right_features, scatter_out_size = inputs

        if self.right_to_left:
            scatter_dim = 0
            prev_features = self.feature_module_left(left_features)
            neighbour_features = self.feature_module_right(right_features)
            joint_features = self.feature_module_edge(edge_features) * neighbour_features[edge_indices[1]]
        else:
            scatter_dim = 1
            prev_features = self.feature_module_right(right_features)
            neighbour_features = self.feature_module_left(left_features)
            joint_features = self.feature_module_edge(edge_features) * neighbour_features[edge_indices[0]]

        conv_output = torch.zeros(
            int(scatter_out_size),
            self.emb_size,
            dtype=joint_features.dtype,
            device=joint_features.device,
        )
        conv_output.index_add_(0, edge_indices[scatter_dim], joint_features)

        output = self.output_module(torch.cat([conv_output, prev_features], dim=1))
        return output


class GCNPolicy(nn.Module):
    """Our bipartite Graph Convolutional neural Network (GCN) model."""

    def __init__(self, embSize, nConsF, nEdgeF, nVarF, isGraphLevel=True):
        super().__init__()

        self.emb_size = embSize
        self.cons_nfeats = nConsF
        self.edge_nfeats = nEdgeF
        self.var_nfeats = nVarF
        self.is_graph_level = isGraphLevel

        self.cons_embedding = nn.Sequential(
            nn.LazyLinear(self.emb_size),
            nn.ReLU(),
        )
        self.edge_embedding = nn.Identity()
        self.var_embedding = nn.Sequential(
            nn.LazyLinear(self.emb_size),
            nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution(self.emb_size, right_to_left=True)
        self.conv_c_to_v = BipartiteGraphConvolution(self.emb_size)
        self.conv_v_to_c2 = BipartiteGraphConvolution(self.emb_size, right_to_left=True)
        self.conv_c_to_v2 = BipartiteGraphConvolution(self.emb_size)

        self.output_module = nn.Sequential(
            nn.LazyLinear(self.emb_size),
            nn.ReLU(),
            nn.Linear(self.emb_size, 1, bias=False),
        )

    def forward(self, inputs, training=False):
        (
            constraint_features,
            edge_indices,
            edge_features,
            variable_features,
            n_cons_total,
            n_vars_total,
            n_cons_small,
            n_vars_small,
        ) = inputs

        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        constraint_features = F.relu(
            self.conv_v_to_c((constraint_features, edge_indices, edge_features, variable_features, n_cons_total))
        )
        variable_features = F.relu(
            self.conv_c_to_v((constraint_features, edge_indices, edge_features, variable_features, n_vars_total))
        )
        constraint_features = F.relu(
            self.conv_v_to_c2((constraint_features, edge_indices, edge_features, variable_features, n_cons_total))
        )
        variable_features = F.relu(
            self.conv_c_to_v2((constraint_features, edge_indices, edge_features, variable_features, n_vars_total))
        )

        if self.is_graph_level:
            variable_features = variable_features.view(int(n_vars_total / n_vars_small), n_vars_small, self.emb_size)
            variable_features_mean = torch.mean(variable_features, dim=1)
            constraint_features = constraint_features.view(int(n_cons_total / n_cons_small), n_cons_small, self.emb_size)
            constraint_features_mean = torch.mean(constraint_features, dim=1)
            final_features = torch.cat([variable_features_mean, constraint_features_mean], dim=1)
        else:
            final_features = variable_features

        return self.output_module(final_features)

    def save_state(self, path):
        with open(path, "wb") as f:
            pickle.dump({k: v.detach().cpu() for k, v in self.state_dict().items()}, f)

    def restore_state(self, path, map_location=None):
        with open(path, "rb") as f:
            state_dict = pickle.load(f)
        if map_location is not None:
            state_dict = {k: v.to(map_location) for k, v in state_dict.items()}
        self.load_state_dict(state_dict)
