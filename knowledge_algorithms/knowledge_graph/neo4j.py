from neo4j import GraphDatabase
import pandas as pd
import torch
import pickle, base64
import numpy as np


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
            x = self.decode(df)

        return x, mapping
    
    def decode(self, df: pd.DataFrame):
        """
        Decode the encoded tensor from a DataFrame.

        Parameters:
        df (pd.DataFrame): DataFrame with columns ["name", "enc"].

        Returns:
        np.ndarray: Decoded (n_rows, 764) tensor.
        """
        decoded_tensors = []

        for enc in df['enc']:
            # Base64 decode
            b64_decoded = base64.b64decode(enc)

            # Unpickle
            tensor_list = pickle.loads(b64_decoded)

            # Convert list to tensor (torch array)
            tensor = torch.tensor(tensor_list)

            # Append to the list of tensors
            decoded_tensors.append(tensor)

        # Stack tensors to form a (n_rows, 764) array
        result_tensor = torch.vstack(decoded_tensors)
    
        return result_tensor
    

    def load_edge(self, cypher, src_index_col, src_mapping, dst_index_col, dst_mapping,
                  encoders=None, **kwargs):
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

        return edge_index, edge_attr