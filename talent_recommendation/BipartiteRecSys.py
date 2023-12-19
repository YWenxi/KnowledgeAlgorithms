import torch
import torch_geometric.data
from torch import nn
import torch.nn.functional as F

from bidict import bidict
import yaml
import json

from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected, RandomLinkSplit

import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.nn.conv import MessagePassing, GCNConv
from torch_geometric.utils import degree

import knowledge_algorithms.deep_learning.pytorch
from knowledge_algorithms.knowledge_graph.neo4j import Neo4jAPI
from knowledge_algorithms.knowledge_graph.processors import SequenceEncoder, IdentityEncoder

from typing import Union


# Neural Graph Collaborate Filtering
class NGCFConv(MessagePassing):
    def __init__(self, latent_dim, dropout, bias=True, **kwargs):
        super().__init__(aggr="add", **kwargs)
        self.dropout = dropout

        self.lin_1 = nn.Linear(latent_dim, latent_dim, bias=bias)
        self.lin_2 = nn.Linear(latent_dim, latent_dim, bias=bias)

        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.lin_1.weight)
        nn.init.xavier_uniform_(self.lin_2.weight)

    def forward(self, x, edge_index):
        # Compute normalization
        from_, to_ = edge_index
        deg = degree(to_, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]

        # Start propagating messages
        out = self.propagate(edge_index, x=(x, x), norm=norm)

        # Perform update after aggregation
        out += self.lin_1(x)
        out = F.dropout(out, self.dropout, self.training)
        return F.leaky_relu(out)

    def message(self, x_j, x_i, norm):
        return norm.view(-1, 1) * (self.lin_1(x_j) + self.lin_2(x_j * x_i))


class RecSysGNN(nn.Module):
    def __init__(
            self,
            latent_dim,
            num_layers,
            core_layer="LightGCN",  # 'NGCF' or 'LightGCN'
            dropout=0.1  # Only used in NGCF
    ):
        super(RecSysGNN, self).__init__()

        assert (core_layer == 'NGCF' or core_layer == 'LightGCN'), \
            'Model must be NGCF or LightGCN'
        self.model = core_layer
        # self.embedding = nn.Embedding(num_users + num_items, latent_dim)

        if self.model == 'NGCF':
            self.convs = nn.ModuleList(
                NGCFConv(latent_dim, dropout=dropout) for _ in range(num_layers)
            )
        else:
            self.convs = nn.ModuleList(SAGEConv((-1, -1), latent_dim) for i in range(num_layers))

        self.init_parameters()

    def init_parameters(self):
        pass

    def forward(self, x, edge_index):
        emb0 = x
        embs = [emb0]

        emb = emb0
        for conv in self.convs:
            emb = conv(x=emb, edge_index=edge_index)
            embs.append(emb)

        out = (
            torch.cat(embs, dim=-1) if self.model == 'NGCF'
            else torch.mean(torch.stack(embs[1:], dim=0), dim=0)
        )

        return emb0, out

    def encode_minibatch(self, nodes, pos_items, neg_items, edge_index):
        emb0, out = self(edge_index)
        return (
            out[nodes],
            out[pos_items],
            out[neg_items],
            emb0[nodes],
            emb0[pos_items],
            emb0[neg_items]
        )


class BipartiteRecSys(knowledge_algorithms.deep_learning.pytorch.PyTorchAlgorithm):
    def __init__(self, driver: Neo4jAPI, core_layer="lightGCN", latent_dim=64, num_layer=3, dropout=0.1, **kwargs):
        self.out_embed = None
        self.in_embed = None
        self.driver = driver
        model = RecSysGNN(latent_dim, num_layer, core_layer, dropout)
        super().__init__(model)

    @torch.no_grad()
    def __call__(self, subgraph: torch_geometric.data.HeteroData):
        self.model(subgraph.x_dict, subgraph.edge_index_dict)

    def predict(self, node_or_subgraph, top_k, search_flag=0, mapping=None):
        """
        :param node_or_subgraph: a node of the current graph
            or the subgraph which is compatible with the current graph
        :param top_k: choose the top k scored candidate
        :param search_flag: 0 if search the person, 1 if search the project
        :param mapping: the node-index dict, for example the person_mapping
        :return:
        """
        if isinstance(node_or_subgraph, str):
            # find the subgraph by id
            pass
        assert isinstance(node_or_subgraph, torch_geometric.data.HeteroData)
        _, out_embed = self(node_or_subgraph)
        subgraph_person_emb = out_embed["person"]
        subgraph_project_emb = out_embed["project"]
        score = subgraph_person_emb @ subgraph_project_emb.T
        values, indices = torch.topk(score, top_k, dim=search_flag)

        if mapping is None:
            return values, indices
        else:
            isinstance(mapping, dict)
            mapping = bidict(mapping)
            candidates = {}
            if search_flag == 0:
                # search the suitable people
                for project_id in range(indices.shape[1]):
                    for candidate_id in indices[project_id]:
                        candidates[str(project_id)] = {
                            "person": self.driver.fetch_data(
                                query=f"match (p:Person) with p.name=={mapping[candidate_id]}"
                            ),
                            "score": values[candidate_id, project_id]
                        }
            elif search_flag == 1:
                pass

            return candidates

    def train(self, train_data: torch_geometric.data.HeteroData):
        self.in_embed, self.out_embed = self.model(train_data.x_dict, train_data.edge_index_dict)
        pass

    def metrics(self):
        pass


def similarity_recommend(result=None, configs: Union[str, dict] = None, encoding_model: str = None):
    if isinstance(configs, str):
        with open(configs, "r") as f:
            configs = yaml.safe_load(f)

    assert configs["neo4j"]

    # check cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # connect to neo4j
    db = Neo4jAPI(configs["neo4j"]["uri"], configs["neo4j"]["user"], configs["neo4j"]["password"])

    # load a pretrained-encoder from huggingface
    sentence_encoder = SequenceEncoder(model_name=encoding_model)

    # query the database
    # 1. person
    person_query = """
        match (p:person) 
        return 
            p.name as personName, 
            properties(p) as description
        """
    person_x, person_mapping = db.load_node(
        person_query, index_col='personName',
        encoders={"description": sentence_encoder}
    )

    # 2. projects
    project_query = """
        match (p:project) 
        return 
            p.name as projectName, 
            properties(p) as description
        """
    project_x, project_mapping = db.load_node(
        project_query, index_col='projectName',
        encoders={"description": sentence_encoder}
    )

    # 3. relation
    relation_query = """
        MATCH (head:person)-[r]->(tail:project) 
        RETURN head.name as personName, r.name as rname, tail.name as projectName
        """
    edge_index, edge_label = db.load_edge(
        relation_query,
        src_index_col='personName',
        src_mapping=person_mapping,
        dst_index_col='projectName',
        dst_mapping=project_mapping,
        encoders={"rname": sentence_encoder}
    )




if __name__ == "__main__":
    # load configuration
    with open("configs.yaml", "r") as f:
        configs = yaml.safe_load(f)

    # check cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # load a pretrained-encoder from huggingface
    sentence_encoder = SequenceEncoder()

    # connect to neo4j
    db = Neo4jAPI(configs["neo4j"]["uri"], configs["neo4j"]["user"], configs["neo4j"]["password"])

    # query the database
    # 1. person
    person_query = """
    match (p:Person) 
    return 
        p.name as personName, 
        properties(p) as description
    """
    person_x, person_mapping = db.load_node(
        person_query, index_col='personName',
        encoders={"description": sentence_encoder}
    )

    # 2. projects
    project_query = """
    match (p:Project) 
    return 
        p.name as projectName, 
        properties(p) as description
    """
    project_x, project_mapping = db.load_node(
        project_query, index_col='projectName',
        encoders={"description": sentence_encoder}
    )

    # 3. relation
    relation_query = """
    MATCH (head:Person)-[r]->(tail:Project) 
    RETURN head.name as personName, r.name as rname, tail.name as projectName
    """
    edge_index, edge_label = db.load_edge(
        relation_query,
        src_index_col='personName',
        src_mapping=person_mapping,
        dst_index_col='projectName',
        dst_mapping=project_mapping,
        encoders={"rname": sentence_encoder}
    )

    data = HeteroData()
    # Add user node features for message passing:
    data['person'].x = person_x
    # Add movie node features
    data['project'].x = project_x
    # Add ratings between users and movies
    data['person', 'workedAs', 'project'].edge_index = edge_index
    data['person', 'workedAs', 'project'].edge_label = edge_label
    # data = ToUndirected()(data)
    data.to(device, non_blocking=True)
