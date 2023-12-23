import yaml
import json
import random
import pandas as pd
import torch
import torch.nn.functional as F

from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected, RandomLinkSplit

import torch_geometric.transforms as T
# from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.nn.conv import MessagePassing, GCNConv
from torch_geometric.nn import HeteroConv, RGCNConv, SAGEConv
from torch_geometric.utils import degree

import knowledge_algorithms.deep_learning.pytorch
from knowledge_algorithms.knowledge_graph.neo4j import Neo4jAPI
from knowledge_algorithms.knowledge_graph.processors import SequenceEncoder

from typing import Union, Dict, Any, List, Tuple

from pathlib import Path

from .metrics import compute_auc, compute_precision_recall_f1, hit_at_k


def get_hetero_conv_layer(in_channels, out_channels, data: HeteroData, conv_layer=SAGEConv, aggr="sum"):
    """
    Creates a heterogeneous convolution layer suitable for processing graph data with multiple types of edges. 
    The function dynamically selects the convolution layer based on the provided `conv_layer` argument. 
    It supports both `RGCNConv` and other convolution layers like `SAGEConv`. The function iterates over the 
    different edge types in the provided `HeteroData` object and creates a convolution layer for each edge type.

    :param in_channels: The number of input channels (features).
    :type in_channels: int
    :param out_channels: The number of output channels (features).
    :type out_channels: int
    :param data: The heterogeneous graph data containing multiple types of edges.
    :type data: HeteroData
    :param conv_layer: The type of convolution layer to be used, defaults to SAGEConv.
                       Must be a class that can be instantiated with in_channels, out_channels, and other required arguments.
    :type conv_layer: type, optional
    :param aggr: The type of aggregation to use ('sum', 'mean', 'max', etc.), defaults to 'sum'.
    :type aggr: str, optional
    :return: An instance of `HeteroConv` with a dictionary mapping each edge type to its respective convolution layer.
    :rtype: HeteroConv

    The function is primarily used in graph neural network models that handle heterogeneous graph data, where
    the graph contains multiple types of nodes and edges. The choice of `conv_layer` and `aggr` allows for 
    customization of the convolution operation based on the model's specific requirements.
    """
    if conv_layer == RGCNConv:
        return HeteroConv({
            edge_type: conv_layer(in_channels, out_channels, 1) for edge_type in data.edge_types
        }, aggr=aggr)
    else:
        return HeteroConv({
            edge_type: conv_layer(in_channels, out_channels, num_relations=1, add_self_loops=False) for edge_type in data.edge_types
        }, aggr=aggr)


class HeteroGNN(torch.nn.Module):
    """
    A heterogeneous Graph Neural Network (GNN) module built using PyTorch. This class is designed for handling
    heterogeneous graph data with multiple types of nodes and edges. It constructs a neural network with a specified
    number of convolutional layers, which can be of a user-defined type.

    The network is composed of a sequence of convolutional layers with ReLU activations, except for the final layer.
    It is compatible with node feature dictionaries and edge index dictionaries.

    :param input_channels: The number of input channels (features).
    :type input_channels: int
    :param hidden_channels: The number of hidden channels in the middle layers.
    :type hidden_channels: int
    :param out_channels: The number of output channels (features) produced by the last layer.
    :type out_channels: int
    :param data: The heterogeneous graph data containing multiple types of edges.
    :type data: HeteroData
    :param num_conv_layers: The number of convolutional layers in the network, defaults to 3. Must be at least 2.
    :type num_conv_layers: int, optional
    :param conv_layer: The type of convolution layer to be used in the network, defaults to RGCNConv.
    :type conv_layer: type, optional

    :raises AssertionError: If the number of convolutional layers is less than 2.

    Example:
        >>> data = HeteroData() # Example data with multiple edge types
        >>> input_channels = 128
        >>> hidden_channels = 64
        >>> out_channels = 32
        >>> model = HeteroGNN(input_channels, hidden_channels, out_channels, data)
        >>> x_dict, edge_index_dict = data.x_dict, data.edge_index_dict # Example node features and edge indices
        >>> out = model(x_dict, edge_index_dict) # Perform forward pass
    """
    def __init__(self, input_channels, hidden_channels, out_channels, data: HeteroData, num_conv_layers=3, 
                 conv_layer=SAGEConv):
        super().__init__()

        assert(num_conv_layers >= 2)
        
        self.conv_layer_list = torch.nn.ModuleList()
        self.num_conv_layers = num_conv_layers

        # Define the first convolutional layer
        conv1 = get_hetero_conv_layer(input_channels, hidden_channels, data, conv_layer)
        self.conv_layer_list.append(conv1)
        
        # Define the middle convolutional layers
        for _ in range(num_conv_layers-2):
            self.conv_layer_list.append(get_hetero_conv_layer(hidden_channels, hidden_channels, data))

        # Define the third convolutional layer
        self.conv_layer_list.append(get_hetero_conv_layer(hidden_channels, out_channels, data))

    def forward(self, x_dict, edge_index_dict):
        """
        The forward pass of the HeteroGNN.

        :param x_dict: A dictionary containing the node features for each node type.
        :type x_dict: Dict[str, torch.Tensor]
        :param edge_index_dict: A dictionary containing the edge indices for each edge type.
        :type edge_index_dict: Dict[str, torch.Tensor]
        :return: A dictionary containing the output node features for each node type after passing through the GNN layers.
        :rtype: Dict[str, torch.Tensor]
        """
        for i in range(self.num_conv_layers - 1):
            x_dict = self.conv_layer_list[i](x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        # Final layer
        x_dict = self.conv_layer_list[-1](x_dict, edge_index_dict)

        return x_dict
    

def load_configs(configs: Union[dict, str], check_keys: list = None):
    """Load the configuration.

    :param configs: Either a dict or a yaml configuration file.
    :type configs: Union[dict, str]
    :param check_keys: check if the config contains the keys in the checklist
    :type check_keys: list[str]
    :return: configuration
    :rtype: dict
    """
    if not isinstance(configs, dict):
        with open(configs, "r") as f:
            configs = yaml.safe_load(f)
    
    # check if the configuration contains needed keys
    if check_keys is not None:
        for key in check_keys:
            assert key in configs
    return configs


def save_configs(configs: dict, file_path: str):
    """
    Save the configuration to a YAML file.

    :param configs: The configuration to save.
    :type configs: dict
    :param file_path: The file path where the configuration will be saved.
    :type file_path: str
    """
    with open(file_path, "w") as f:
        yaml.safe_dump(configs, f, default_flow_style=False)


def transform_hetero_data_for_rec_sys(data, random_data_split=True, 
                                      num_val=0.1, num_test=0.0, is_undirected=True, 
                                      neg_sampling_ratio=0.0, edge_types=None, rev_edge_types=None):
    """
    Transforms heterogeneous graph data for a recommendation system, with options to make the graph undirected
    and to split the data randomly.

    :param data: The input graph data.
    :type data: HeteroData
    :param random_data_split: If True, splits the data randomly into training, validation, and test sets, defaults to True.
    :type random_data_split: bool, optional
    :param num_val: The proportion of edges used for validation, defaults to 0.1.
    :type num_val: float, optional
    :param num_test: The proportion of edges used for testing, defaults to 0.0.
    :type num_test: float, optional
    :param is_undirected: Specifies if the split edges should be undirected, defaults to True.
    :type is_undirected: bool, optional
    :param neg_sampling_ratio: The ratio of negative samples for each positive sample, defaults to 0.0.
    :type neg_sampling_ratio: float, optional
    :param edge_types: List of edge types to consider for splitting, defaults to None.
    :type edge_types: list of tuples, optional
    :param rev_edge_types: List of reverse edge types, corresponding to `edge_types`, defaults to None.
    :type rev_edge_types: list of tuples, optional
    :return: Either the transformed data or a tuple of (train_data, val_data, test_data), depending on the flags.
    :rtype: HeteroData or (HeteroData, HeteroData, HeteroData)

    Example:
        >>> data = HeteroData() # Example graph data
        >>> train_data, val_data, test_data = transform_hetero_data_for_rec_sys(
                data, num_val=0.15, num_test=0.05, edge_types=[("user", "likes", "item")])
    """
        
    def hash_edge_types(edge_types):
        if edge_types is None or (len(edge_types) == 0):
            return None
        else:
            return [tuple(edge_type) for edge_type in edge_types]

    if random_data_split:
        transform = RandomLinkSplit(
            num_val=num_val,
            num_test=num_test,
            is_undirected=is_undirected,
            neg_sampling_ratio=neg_sampling_ratio,
            edge_types=hash_edge_types(edge_types) or [("employee", "workedIn", "project")],
            rev_edge_types=hash_edge_types(rev_edge_types) or [("project", "rev_workedIn", "employee")]
        )
        train_data, val_data, test_data = transform(data)
        return train_data, val_data, test_data
    else:
        return data



def generate_negative_samples(pos_edge_index, num_nodes, num_neg_samples):
    """
    Generate negative samples.

    :param pos_edge_index: Tensor of shape [2, num_edges] representing positive edges.
    :param num_nodes: Tuple (num_employees, num_projects) - the number of nodes in each category.
    :param num_neg_samples: The number of negative samples to generate.
    :return: Tensor of negative samples.
    

    This step is crucial for effectively training your model, 
    especially since the quality of the negative samples can significantly influence the model's performance.

    - Positive Samples:
        Positive samples are straightforward since they are the existing edges in your graph. 
        For instance, if you're recommending projects to employees, 
        your positive samples are the `('employee', 'workedIn', 'project')` edges.

    - Negative Samples:
        Generating negative samples is more challenging. 
        You need to create pairs of nodes that do not have an existing edge between them. 
        It's important that these negative samples are realistic; 
        that is, they should be plausible but non-existent edges.

    Here's a simple approach to generate negative samples:
    
        1. Randomly select an 'employee' node.
        2. Randomly select a 'project' or 'position' node.
        3. Check if this pair forms an edge in your graph. If not, it's a valid negative sample.
        4. Repeat this process until you have the desired number of negative samples.
    
    
    *Examples:*
     
    >>> # Example usage
    >>> num_employees = 10000
    >>> num_projects = 4371  # or num_positions
    >>> num_neg_samples = 10000  # This can be adjusted

    >>> # Assuming you have your positive edge index for ('employee', 'workedIn', 'project')
    >>> pos_edge_index = data[('employee', 'workedIn', 'project')].edge_index
    >>> neg_edge_index = generate_negative_samples(pos_edge_index, (num_employees, num_projects), num_neg_samples)
    """
    neg_samples = []
    while len(neg_samples) < num_neg_samples:
        # Randomly select an 'employee' and a 'project/position'
        employee = random.randint(0, num_nodes[0] - 1)
        project = random.randint(0, num_nodes[1] - 1)

        # Check if this is a negative sample
        if not torch.any((pos_edge_index[0] == employee) & (pos_edge_index[1] == project)):
            neg_samples.append([employee, project])

    return torch.tensor(neg_samples).t().to(pos_edge_index.device)


def train(model: torch.nn.Module, train_data: HeteroData, val_data: HeteroData = None,
          target_edge_type: tuple = None, loss_function: Union[torch.nn.Module, str] = "MarginRankingLoss",
          optimizer: Union[torch.optim.Optimizer, str] = "Adam", num_epochs: int = 10,
          num_neg_samples: int = None, loss_function_kwargs: dict = {"margin": 0.5},
          optimizer_kwargs: dict = {}, lr: float = 0.001, k_to_hit: int = 5, device: torch.device = None,
          val_per_epochs: int = 10):
    """
    Train a model on heterogeneous graph data using specified loss function, optimizer, and evaluation metrics.

    :param model: The graph neural network model to be trained.
    :type model: torch.nn.Module
    :param train_data: The training dataset.
    :type train_data: HeteroData
    :param val_data: The validation dataset, defaults to None.
    :type val_data: HeteroData, optional
    :param target_edge_type: The target edge type for training, specified as a tuple (head, relation, tail).
    :type target_edge_type: tuple, optional
    :param loss_function: The loss function or its string identifier, defaults to 'MarginRankingLoss'.
    :type loss_function: Union[torch.nn.Module, str], optional
    :param optimizer: The optimizer or its string identifier, defaults to 'Adam'.
    :type optimizer: Union[torch.optim.Optimizer, str], optional
    :param num_epochs: The number of training epochs, defaults to 3.
    :type num_epochs: int, optional
    :param num_neg_samples: The number of negative samples per positive sample, defaults to None.
    :type num_neg_samples: int, optional
    :param loss_function_kwargs: Additional keyword arguments for the loss function, defaults to {"margin": 0.5}.
    :type loss_function_kwargs: dict, optional
    :param optimizer_kwargs: Additional keyword arguments for the optimizer, defaults to {}.
    :type optimizer_kwargs: dict, optional
    :param lr: Learning rate for the optimizer, defaults to 0.001.
    :type lr: float, optional
    :param k_to_hit: The 'k' value for hit rate calculation, defaults to 5.
    :type k_to_hit: int, optional
    :param device: The device to run the training on (CPU or CUDA), defaults to None.
    :type device: torch.device, optional
    :param val_per_epochs: Validation frequency (number of epochs), defaults to 10.
    :type val_per_epochs: int, optional

    Example:
        >>> model = SomeGNNModel()
        >>> train_data, val_data = load_hetero_data()  # Load your training and validation data
        >>> train(model, train_data, val_data, target_edge_type=("user", "interacts", "item"),
                  num_epochs=10, k_to_hit=10, lr=0.005)
    """
    h, r, t = target_edge_type
    model.train()
    
    # ensure device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = train_data.to(device)
    model = model.to(device)
    print(f"Start training on {device} ...")
    
    # loss function
    if loss_function == "MarginRankingLoss":
        loss_function = torch.nn.MarginRankingLoss(**loss_function_kwargs)
        
    # optimizer
    if optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, **optimizer_kwargs)
    
    if num_neg_samples is None:
        num_neg_samples = train_data[target_edge_type].num_edges

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward pass through GNN
        node_embeddings = model(train_data.x_dict, train_data.edge_index_dict)

        # Assume you have a function to generate positive and negative samples
        pos_samples = train_data[target_edge_type].edge_index
        neg_samples = generate_negative_samples(
            pos_samples, 
            (train_data[h].num_nodes, train_data[t].num_nodes), 
            num_neg_samples=num_neg_samples
        )

        # Compute scores for positive and negative samples
        # Example: Using dot product to compute scores
        pos_scores = (node_embeddings[h][pos_samples[0]] * node_embeddings[t][pos_samples[1]]).sum(dim=1)
        neg_scores = (node_embeddings[h][neg_samples[0]] * node_embeddings[t][neg_samples[1]]).sum(dim=1)

        # Target tensor for MarginRankingLoss
        target = torch.ones(pos_scores.size(), device=device)

        # Compute loss
        loss = loss_function(pos_scores, neg_scores, target)

        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        # Evaluate on validation set
        if (epoch + 1) % val_per_epochs == 0:
            if val_data is not None:
                val_data.to(device)
                model.eval()
                with torch.no_grad():
                    # Compute embeddings for validation set
                    val_embeddings = model(val_data.x_dict, val_data.edge_index_dict)

                    # Generate positive and negative samples for validation
                    val_pos_samples = val_data[target_edge_type].edge_index
                    val_neg_samples = generate_negative_samples(
                        val_pos_samples,
                        (val_data[h].num_nodes, val_data[t].num_nodes),
                        num_neg_samples=val_data[target_edge_type].num_edges
                    )
                    
                    # Compute scores for validation samples
                    val_pos_scores = (val_embeddings[h][val_pos_samples[0]] * val_embeddings[t][val_pos_samples[1]]).sum(dim=1)
                    val_neg_scores = (val_embeddings[h][val_neg_samples[0]] * val_embeddings[t][val_neg_samples[1]]).sum(dim=1)
                    
                    # compute validation loss
                    target = torch.ones(val_pos_scores.size(), device=device)
                    val_loss = loss_function(val_pos_scores, val_neg_scores, target)

                    # Calculate metrics
                    val_auc = compute_auc(val_pos_scores, val_neg_scores)
                    val_precision, val_recall, val_f1 = compute_precision_recall_f1(val_pos_scores, val_neg_scores)
                    val_hit_k = hit_at_k(val_pos_scores, val_neg_scores, k=k_to_hit)  # You can adjust k
                    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.4f} "
                        f"Val AUC: {val_auc:.4f}, Val Precision: {val_precision:.4f}, "
                        f"Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}, Val Hit@{k_to_hit}: {val_hit_k:.4f}")
            else:
                print(f"Epoch [{epoch+1:03d}/{num_epochs}], Loss: {loss.item()}")


def rec_sys_train(configs: Union[dict, str], data:HeteroData=None, device=None, save=True):
    """
    Train a recommendation system model using heterogeneous graph data.

    This function trains a recommendation system model using heterogeneous graph data
    and saves the trained model and data configurations if specified.

    :param Union[dict, str] configs: Either a dictionary containing configuration parameters
        or a string specifying the path to a configuration file (in YAML format).
    :param HeteroData data: The heterogeneous graph data to be used for training.
    :param string device: The device to use for training (e.g., "cpu" or "cuda").
    :param bool save: Whether to save the trained model and data configurations.
    
    :return: A trained recommendation system model and the input heterogeneous graph data.
    :rtype: Tuple[HeteroGNN, HeteroData]

    Example:
    >>> configs = "./my_configs.yaml"
    >>> model, data = rec_sys_train(configs)
    
    Example:
    >>> model, data = rec_sys_train({"neo4j": {"url": "http://localhost:7474", "user": "user", "password": "password"}}, data)
    """
    
    # load configs
    if isinstance(configs, str):
        configs_file = configs
    elif isinstance(configs, dict):
        configs_file = "./configs.yaml"
    configs = load_configs(configs, check_keys=["neo4j"])

    # connect to neo4j database
    db = Neo4jAPI(configs["neo4j"]["url"], configs["neo4j"]["user"], configs["neo4j"]["password"])
    print("Connected to Neo4j Database.")

    # get the Hetero Dataset
    if data is None:
        data: HeteroData = db.load_hetero_graph_dataset(configs["neo4j"]["labels"])
    data = ToUndirected()(data)

    train_data, val_data, test_data = transform_hetero_data_for_rec_sys(data, **configs["dataTransform"])

    model = HeteroGNN(data = data, **configs["graphNeuralNetwork"])
    train(model, train_data, val_data, target_edge_type=eval(configs["targetEdgeType"]), **configs["train"])
    
    if save:
        if not isinstance(save, str):
            save = "./checkpoints"
        save = Path(save)
        save.mkdir() if not save.is_dir() else None
        configs['save'] = {
            'model': str(save_model(model, save)),
            'data': str(save_dataset(data, save))
        }
        save_configs(configs=configs, file_path=configs_file)
        
    
    return model, data


def save_model(model: HeteroGNN, path="./"):
    """
    Save a PyTorch model to a specified file path.

    This function saves a PyTorch model to a specified file path. If the path is a directory,
    the model will be saved as "model.pt" in that directory.

    :param HeteroGNN model: The PyTorch model to be saved.
    :param Union[str, Path] path: The file path where the model will be saved.
        If a directory is provided, the model will be saved as "model.pt" in that directory.
    
    :return: The absolute file path where the model was saved.
    :rtype: Path

    Example:
    >>> model = HeteroGNN(...)
    >>> saved_path = save_model(model, "./models")
    >>> print(saved_path)
    '/path/to/models/model.pt'

    Example:
    >>> model = HeteroGNN(...)
    >>> saved_path = save_model(model, "./model.pth")
    >>> print(saved_path)
    '/path/to/model.pth'
    """
    path = Path(path)
    if path.is_dir():
        path /= "model.pt"
    torch.save(model.to("cpu").state_dict(), path)
    return path.absolute()


def load_model(model: HeteroGNN, path: str):
    """
    Load a PyTorch model from a specified file path and set it to evaluation mode.

    This function loads a PyTorch model from a specified file path and sets it to evaluation mode.
    The model should have the same architecture as the one used during training.

    :param HeteroGNN model: The PyTorch model to be loaded.
    :param str path: The file path from which to load the model.
    
    :return: The loaded PyTorch model set to evaluation mode.
    :rtype: HeteroGNN

    Example:
    >>> loaded_model = load_model(MyModel(), "./models/model.pt")
    >>> loaded_model.eval()
    <HeteroGNN ...>

    Example:
    >>> loaded_model = load_model(MyModel(), "./saved_models/model.pth")
    >>> loaded_model.eval()
    <HeteroGNN ...>
    """
    model.load_state_dict(torch.load(Path(path)))
    model.eval()  # Set the model to evaluation mode
    return model


def save_dataset(dataset: HeteroData, path:str="./"):
    """
    Save a HeteroData object to a specified file path.

    This function saves a HeteroData object to a specified file path. If the path is a directory,
    the dataset will be saved as "dataset.pt" in that directory.

    :param HeteroData dataset: The HeteroData object to be saved.
    :param Union[str, Path] path: The file path where the dataset will be saved.
        If a directory is provided, the dataset will be saved as "dataset.pt" in that directory.
    
    :return: The absolute file path where the dataset was saved.
    :rtype: Path

    Example:
    >>> dataset = MyHeteroData()
    >>> saved_path = save_dataset(dataset, "./datasets")
    >>> print(saved_path)
    '/path/to/datasets/dataset.pt'

    Example:
    >>> dataset = MyHeteroData()
    >>> saved_path = save_dataset(dataset, "./data/datafile.pth")
    >>> print(saved_path)
    '/path/to/data/datafile.pth'
    """
    path = Path(path)
    if path.is_dir():
        path /= "dataset.pt"
    # Convert HeteroData to a dictionary
    dataset_dict = dataset.to_dict()
    # Save the dictionary as a PyTorch file
    torch.save(dataset_dict, path)
    return path.absolute()


def load_dataset(path: Union[str, Path]):
    """
    Load a HeteroData object from a specified file path.

    This function loads a HeteroData object from a specified file path. If the path is a directory,
    the dataset will be loaded from "dataset.pt" in that directory.

    :param Union[str, Path] path: The file path from which to load the dataset.
    
    :return: The loaded HeteroData object.
    :rtype: HeteroData

    Example:
    >>> loaded_dataset = load_dataset("./datasets")
    >>> print(loaded_dataset)
    <HeteroData ...>

    Example:
    >>> loaded_dataset = load_dataset("./data/datafile.pth")
    >>> print(loaded_dataset)
    <HeteroData ...>
    """
    path = Path(path)
    if path.is_dir():
        path /= "dataset.pt"

    # Load the dictionary and convert it back to HeteroData
    dataset_dict = torch.load(path)
    dataset = HeteroData().from_dict(dataset_dict)
    
    return dataset


## for recommendation and similarity computation

def json_to_feature_vector(json_data: dict, sentence_encoder=None):
    """
    Convert JSON data to a feature vector using a sentence encoder.

    This function takes a JSON dictionary as input and converts it into a feature vector
    using a specified sentence encoder model.

    :param dict json_data: The JSON data to be converted to a feature vector.
    :param Union[SequenceEncoder, str] sentence_encoder: An instance of a sentence encoder model,
        or the name/directory path of a pre-trained sentence encoder model to be used for encoding.
    
    :return: The feature vector representation of the JSON data.
    :rtype: torch.Tensor

    Example:
    >>> json_data = {"text": "This is a sample text."}
    >>> encoder = SequenceEncoder("bert-base-uncased")
    >>> feature_vector = json_to_feature_vector(json_data, encoder)
    >>> print(feature_vector)
    tensor([[0.1234, 0.5678, ..., 0.9876]])

    Example:
    >>> json_data = {"text": "Another example text."}
    >>> feature_vector = json_to_feature_vector(json_data, "bert-base-uncased")
    >>> print(feature_vector)
    tensor([[0.5678, 0.4321, ..., 0.8765]])
    """
    
    # Example function to convert JSON to a feature vector
    # This should be aligned with how the original features were generated
    if not isinstance(sentence_encoder, SequenceEncoder):
        # could either pass the model name in huggingface, or directory holding models 
        sentence_encoder = SequenceEncoder(sentence_encoder)
    
    # clear json if the entry holds emply list
    k_del_list = [k for k, v in json_data.items() if v == []]
    for k in k_del_list:
        del json_data[k]
    feature_vector = sentence_encoder.model.encode(str(json_data))
    return feature_vector


def add_new_position_node(hetero_data, json_data, sentence_encoder=None, techstack_mapping:dict=None, device='cpu'):
    """
    Add a new 'position' node with edges to 'techStack' nodes based on JSON data.

    This function adds a new 'position' node to the heterogeneous graph data and
    creates edges to 'techStack' nodes based on the information provided in the JSON data.

    :param hetero_data: The heterogeneous graph data to which the new node and edges will be added.
    :param json_data: JSON data containing information about the new 'position' node
        and its related 'techStack' nodes.
    :param sentence_encoder: An instance of a sentence encoder model or the name/directory path
        of a pre-trained sentence encoder model to be used for encoding descriptions.
    :param techstack_mapping: A dictionary mapping tech stack names to their corresponding indices.
    :param device: The device (e.g., 'cpu' or 'cuda') to be used for tensor operations.
    
    :return: A tuple containing the updated heterogeneous graph data and the index of the new 'position' node.
    :rtype: Tuple[YourHeteroData, int]

    Example:
    >>> json_data = {"职位": "Software Engineer", "技术栈": ["Python", "JavaScript"]}
    >>> hetero_data, position_idx = add_new_position_node(hetero_data, json_data, sentence_encoder, techstack_mapping, 'cuda')
    >>> print(position_idx)
    42
    >>> print(hetero_data)
    <YourHeteroData ...>

    Example:
    >>> json_data = {"职位": "Data Scientist", "技术栈": ["Python", "R"]}
    >>> hetero_data, position_idx = add_new_position_node(hetero_data, json_data, "bert-base-uncased", techstack_mapping, 'cpu')
    >>> print(position_idx)
    43
    >>> print(hetero_data)
    <YourHeteroData ...>
    """
    
    hetero_data.to(device)
    
    # Convert numpy array to torch tensor
    feature_vector = json_to_feature_vector(json_data, sentence_encoder)
    feature_vector = torch.tensor(feature_vector, dtype=torch.float32).to(device)

    # Add the new position node
    new_position_idx = hetero_data['position'].x.size(0)  # Index of the new position node
    hetero_data['position'].x = torch.cat([hetero_data['position'].x, feature_vector.unsqueeze(0)], dim=0)
    new_position_name = json_data['职位']
    hetero_data['position']["mapping"][new_position_name] = new_position_idx

    # Prepare to add new edges for 'position'-'techStack' relationship
    techstack_names = json_data.get('技术栈', [])
    new_edges = []

    if techstack_mapping is None:
        if "mapping" in hetero_data["techStack"]:
            techstack_mapping = hetero_data["techStack"]["mapping"]
        else:
            raise Exception("You need to give the techStack mapping: name --> index !!!")
        
    for tech_name in techstack_names:
        if tech_name in techstack_mapping:
            techstack_idx = techstack_mapping[tech_name]
            new_edges.append((new_position_idx, techstack_idx))

    if new_edges:
        new_edges = torch.tensor(new_edges, dtype=torch.long).t().to(device)
        hetero_data['position', 'needTechstack', 'techStack'].edge_index = torch.cat(
            [hetero_data['position', 'needTechstack', 'techStack'].edge_index, new_edges], dim=1)

    print(f"{len(new_edges)} new edges added.")
    return hetero_data, new_position_idx


def score_and_rank_employees(model, hetero_data, new_position_idx, device='cpu'):
    """
    Score and rank employees based on compatibility with a new 'position' node.

    This function scores and ranks employees based on their compatibility with a new 'position' node
    that was added to the heterogeneous graph. Compatibility is calculated using embeddings obtained
    from a trained heterogeneous graph neural network model.

    :param YourHeteroGNN model: The trained heterogeneous graph neural network model.
    :param YourHeteroData hetero_data: The heterogeneous graph data containing employees and the new 'position'.
    :param int new_position_idx: The index of the new 'position' node in the graph.
    :param str device: The device (e.g., 'cpu' or 'cuda') to be used for tensor operations.
    
    :return: Two tensors containing sorted indices and corresponding scores of employees.
    :rtype: Tuple[torch.Tensor, torch.Tensor]

    Example:
    >>> model = YourHeteroGNN()
    >>> hetero_data = YourHeteroData()
    >>> new_position_idx = 42
    >>> sorted_indices, sorted_scores = score_and_rank_employees(model, hetero_data, new_position_idx, 'cuda')
    >>> print(sorted_indices)
    tensor([10, 5, 20, ...])
    >>> print(sorted_scores)
    tensor([0.9876, 0.8765, 0.7654, ...])

    Example:
    >>> model = YourHeteroGNN()
    >>> hetero_data = YourHeteroData()
    >>> new_position_idx = 43
    >>> sorted_indices, sorted_scores = score_and_rank_employees(model, hetero_data, new_position_idx, 'cpu')
    >>> print(sorted_indices)
    tensor([15, 12, 30, ...])
    >>> print(sorted_scores)
    tensor([0.8765, 0.7654, 0.6543, ...])
    """
    model.eval()
    hetero_data = hetero_data.to(device)
    with torch.no_grad():
        embeddings = model(hetero_data.x_dict, hetero_data.edge_index_dict)

    # Get the embedding of the new position
    position_embedding = embeddings['position'][new_position_idx]

    # Get the embeddings of all employees
    employee_embeddings = embeddings['employee']

    # Calculate compatibility scores (e.g., using dot product)
    scores = torch.matmul(employee_embeddings, position_embedding.T)

    # Rank employees based on scores
    sorted_scores, sorted_indices = torch.sort(scores, descending=True)

    return sorted_indices, sorted_scores


def _reverse_mapping(mapping_dict):
    """
    Reverse the keys and values of a dictionary.

    This function takes a dictionary as input and returns a new dictionary where
    the keys and values are reversed.

    :param dict mapping_dict: The dictionary to be reversed.
    
    :return: A new dictionary with keys and values reversed.
    :rtype: dict

    Example:
    >>> original_dict = {'a': 1, 'b': 2, 'c': 3}
    >>> reversed_dict = _reverse_mapping(original_dict)
    >>> print(reversed_dict)
    {1: 'a', 2: 'b', 3: 'c}
    """
    reversed_dict = {index: name for name, index in mapping_dict.items()}
    return reversed_dict


def recommend(json_request, model, data, sentence_encoder=None, topk=5, verbose=1):
    """
    Recommend top employees for a new position based on JSON request and model.

    This function recommends the top employees for a new position based on the provided JSON request,
    a trained recommendation model, and existing data. It calculates compatibility scores for employees
    and ranks them to provide the top recommendations.

    :param dict json_request: JSON request containing information about the new 'position' node.
    :param model: The trained recommendation model.
    :param data: The heterogeneous graph data containing employee and 'position' nodes.
    :param sentence_encoder: An instance of a sentence encoder model or the name/directory path
        of a pre-trained sentence encoder model to be used for encoding descriptions.
    :param int topk: The number of top recommendations to return (default is 5).
    :param int verbose: Verbosity level (0 for no output, 1 for minimal output).
    
    :return: A list of tuples containing recommended employee names and their corresponding scores.
    :rtype: List[Tuple[str, float]]

    Example:
    >>> json_request = {"职位": "Software Engineer", "技术栈": ["Python", "JavaScript"]}
    >>> recommendations = recommend(json_request, model, data, sentence_encoder, topk=3, verbose=1)
    >>> for employee, score in recommendations:
    ...     print(f"Employee: {employee}, Score: {score}")
    Employee: John Smith, Score: 0.9876
    Employee: Sarah Brown, Score: 0.8765
    Employee: Michael Johnson, Score: 0.7654
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    new_hetero_data, new_position_idx = add_new_position_node(data, json_request, sentence_encoder, device=device)
    
    model.to(device)
    new_hetero_data.to(device)
    sentence_encoder.model.to(device)
    
    ranked_employee_indices, ranked_scores = score_and_rank_employees(model, new_hetero_data, new_position_idx, device)
    
    # Get top k recommended employees
    top_employees = ranked_employee_indices[:topk]
    top_scores = ranked_scores[:topk]
    
    # Display recommendations
    outs = []
    employee_reverse_mapping = _reverse_mapping(new_hetero_data["employee"]["mapping"])
    for idx, score in zip(top_employees.tolist(), top_scores.tolist()):
        if verbose > 0:
            print(f"Employee ID: {idx}, Name: {employee_reverse_mapping[idx]}, Score: {score}")
        outs.append((employee_reverse_mapping[idx], score))
        
    return outs


def similarity_compute(
    input_request: str,
    configs: Union[str, Dict[str, Any]],
    topk: int = 5,
    data: Union[str, Path, None] = None,
    model: Any = None,
    verbose: int = 1
) -> Tuple[pd.DataFrame, List[Tuple[str, float]]]:
    """
    Compute and return employee recommendations based on a similarity input request.

    This function computes employee recommendations based on a similarity input request
    by utilizing a trained recommendation model, heterogeneous graph data, and a sentence encoder.

    :param str input_request: The input request describing the new 'position' node.
    :param Union[str, Dict[str, Any]] configs: Configuration information in the form of a dictionary
        or the path to a configuration file.
    :param int topk: The number of top recommendations to return (default is 5).
    :param Union[str, Path, None] data: The path to a saved HeteroData object or None.
    :param Any model: The trained recommendation model or None.
    :param int verbose: Verbosity level (0 for no output, 1 for minimal output).
    
    :return: A tuple containing a DataFrame with employee information and a list of recommended employees with scores.
    :rtype: Tuple[pd.DataFrame, List[Tuple[str, float]]]

    Example:
    >>> input_request = "We are looking for a Python Developer with experience in web development."
    >>> config_file = "config.yaml"
    >>> recommendations_df, recommendations = similarity_compute(input_request, config_file, topk=3, data=None, model=None, verbose=1)
    >>> print(recommendations_df)
       工号   学历  首次工作时间  年龄   技术栈
    0  John Smith  Master's Degree   2010  35  ['Python', 'JavaScript']
    1  Sarah Brown  Bachelor's Degree   2012  31  ['Python', 'JavaScript']
    2  Michael Johnson  PhD   2009  37  ['Python', 'JavaScript']
    >>> print(recommendations)
    [('John Smith', 0.9876), ('Sarah Brown', 0.8765), ('Michael Johnson', 0.7654)]
    """
    # load configs
    configs = load_configs(configs, check_keys=["neo4j"])

    
    # connect to neo4j
    db = Neo4jAPI(configs["neo4j"]["url"], configs["neo4j"]["user"], configs["neo4j"]["password"])
    
    # load data  
    if isinstance(data, Union[str, Path]):
        data = load_dataset(data)
    if data is None:
        if "save" in configs and isinstance(configs['save'], dict) and "data" in configs['save']:
            print(f"Found HeteroData at {configs['save']['data']}")
            data = load_dataset(configs['save']["data"])
        else:
            print("No dataset found. Fetch data from neo4j database.")
            data = db.load_hetero_graph_dataset(configs["neo4j"]["labels"])
            data = ToUndirected()(data)
            configs["data"] = data.metadata()
    
    # load model
    if model is None:
        try:
            model = load_model(HeteroGNN(data=data, **configs['graphNeuralNetwork']), path=configs['save']['model'])
        except TypeError:
            print("No model could be found. Now we are going to train a model.")
            model, data = rec_sys_train(configs, data)
    
    # load sentence encoder
    sentence_encoder = SequenceEncoder(configs["sentence_encoder"])
    
    # compute and query
    outs = recommend(input_request, model, data, sentence_encoder, topk=topk, verbose=verbose)
    
    # get the output people
    dfs = []
    for name, score in outs:
        query = f"""
            MATCH (e:employee)-[:hasTechStack]-(t:techStack)
            WHERE e.name = "{name}"
            RETURN e.name AS `工号`, 
                e.`最高学历` AS `学历`, 
                e.`首次工作时间` AS `首次工作时间`,
                date().year - date(e.`出生日期`).year AS `年龄`,
                COLLECT(t.name) AS `技术栈`
        """
        df = db.fetch_data(query)
        if df.empty:
            # this employee has no links to techStack
            query = f"""
                MATCH (e:employee)
                WHERE e.name = "{name}"
                RETURN e.name AS `工号`, 
                    e.`最高学历` AS `学历`, 
                    e.`首次工作时间` AS `首次工作时间`,
                    date().year - date(e.`出生日期`).year AS `年龄`
            """
            df = db.fetch_data(query)
        dfs.append(db.fetch_data(query))
    df = pd.concat(dfs, ignore_index=True)
    return df, outs