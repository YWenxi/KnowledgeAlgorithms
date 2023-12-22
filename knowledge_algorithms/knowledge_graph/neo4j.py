from neo4j import GraphDatabase
import pandas as pd
from torch_geometric.data import HeteroData
import torch
import pickle, base64
import numpy as np
from tqdm import tqdm
from binascii import Error

class Neo4jAPI:
    def __init__(self, uri, user, password):
        # initialize the connection to neo4j database
        super().__init__()
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def fetch_data(self, query, params={}):
        with self.driver.session() as session:
            result = session.run(query, params)
            return pd.DataFrame([r.values() for r in result], columns=result.keys())

    def load_node(self, cypher, index_col, encoders=None, **kwargs):
        """_summary_

        :param cypher: input cypher to query nodes
        :type cypher: string
        :param index_col: index_columns in the returned dataframe
        :type index_col: string
        :param encoders: name or directory path for sentence encoder, defaults to None. If none, would use the default.
        :type encoders: string, optional
        :return: x (the decoded (None, 768) tensor), mapping (the name-to-index dict)
        :rtype: torch.Tensor, dict
        """
        # Execute the cypher query and retrieve data from Neo4j
        df = self.fetch_data(cypher)
        df.set_index(index_col, inplace=True)
        
        # Define node mapping
        mapping = {index: i for i, index in enumerate(df.index.unique())}
        
        # Define node features
        x = None
        if encoders is not None:
            xs = [encoder(df[col]) for col, encoder in encoders.items()]
            x = torch.cat(xs, dim=-1)
        else:
            # if each node is pre-encoded with a property `enc` we get this enc and convert it to tensor
            assert "enc" in df.columns
            x = self.decode_col(df)
            x = torch.tensor(x.tolist())

        return x, mapping
    
    @staticmethod
    def decode(enc: str, to_tensor=False, map_none=0.0):
        """Decode the encoded string encoding. The decoded output is either a list of floats or a 1d tensor with dim=768

        :param enc: Encoded string
        :type enc: str
        :param to_tensor: whether output the pytorch tensor, defaults to False
        :type to_tensor: bool, optional
        :param map_none: the none value would be converted to this-value-fulling tensor/list defaults to 0.0
        :type map_none: float, optional
        :return: either a list of floats or a 1d tensor
        :rtype: either a list of floats or a 1d tensor
        """
        if enc is None:
            return [map_none] * 768
        # Base64 decode
        try:
            b64_decoded = base64.b64decode(enc)
            # Unpickle
            tensor = pickle.loads(b64_decoded)
        except Error:
            # there is a node.enc = 'xx'!!!
            tensor = [map_none] * 768
            
        if to_tensor:
            # Convert list to tensor (torch array)
            tensor = torch.tensor(tensor)
        return tensor
    
    def decode_col(self, df: pd.DataFrame, col="enc", map_none=0):
        """Decode the colomn of the dataframe

        :param df: dataframe to be decoded in place
        :type df: pd.DataFrame
        :param col: the column to be decoded, defaults to "enc"
        :type col: str, optional
        :param map_none: the none value would be converted to this value, defaults to 0 (under development)
        :type map_none: int, optional
        :return: decoded tensor
        :rtype: pd.DataFrame
        """
        result_tensor = None
        if col in df.columns:
            result_tensor = df[col].apply(self.decode)
        return result_tensor
    

    def load_edge(self, src_index_col, src_mapping, dst_index_col, dst_mapping,
                  encoders=None, **kwargs):
        
        # get cypher
        cypher = """
        MATCH (head:{h})-[r]->(tail:{t}) 
        RETURN head.name as {h}, r.name as rname, tail.name as {t}
        """.format_map({"h": src_index_col, "t": dst_index_col})
        
        # Execute the cypher query and retrieve data from Neo4j
        df = self.fetch_data(cypher)
        
        # Define edge index
        src = [src_mapping[index] for index in df[src_index_col]]
        dst = [dst_mapping[index] for index in df[dst_index_col]]
        edge_index = torch.tensor([src, dst])
        
        # Define edge features
        edge_attr = None
        if encoders is not None:
            edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
            edge_attr = torch.cat(edge_attrs, dim=-1)
        
        # try to get the relation name if such a relation exists
        try:
            rname = df["rname"][0]
        except KeyError:
            rname = None
        
        return edge_index, edge_attr, rname
    
    def load_hetero_graph_dataset(self, node_labels: list, include_mapping=True) -> HeteroData:
        """ load the graph dataset containing labels in node_labels 
        """
        # init the data object
        data = HeteroData()

        # load the nodes
        node_query_template = """
            match (n:{}) 
            return 
                n.name as {}, 
                n.enc as enc
        """
        
        node_xs = {}
        node_mappings = {}

        for node_label in node_labels:
            print(f"Loading node:{node_label} ...", end="\t")
            query = node_query_template.format(node_label, node_label)
            x, mapping = self.load_node(query, index_col=f"{node_label}")
            node_xs[node_label] = x
            node_mappings[node_label] = mapping
            print(f"{len(mapping):6d} nodes loaded.")
            
            data[node_label].x = x
            if include_mapping:
                data[node_label]["mapping"] = mapping

            
        # load the edges
        ht_tuples = list((h, t) for h in node_labels for t in node_labels if h != t)
        for (h, t) in ht_tuples:
            print(f"Loading relation ({h})-->({t}) ...", end="\t")
            
            relation_query = f"""
            MATCH (head:{h})-[r]->(tail:{t}) 
            RETURN head.name as {h}, r.name as rname, tail.name as {t}
            """
            
            edge_index, edge_label, rname = self.load_edge(
                src_index_col=h,
                src_mapping=node_mappings[h],
                dst_index_col=t,
                dst_mapping=node_mappings[t],
            )
            
            if rname is None:
                print("{:6d} loaded.".format(0))
                continue
            
            data[h, rname, t].edge_index = edge_index  # [2, num_edges]
            print(f"{data[h, rname, t].num_edges:6d} loaded.")
            
        return data