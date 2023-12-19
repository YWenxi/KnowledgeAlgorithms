import rdflib
import torch
from sentence_transformers import SentenceTransformer
# from owlready2 import get_ontology


class RDFProcessor:
    def __init__(self):
        self.graph = rdflib.Graph()

    def load(self, path):
        self.graph.parse(path, format=rdflib.util.guess_format(path))

    def query(self, query):
        return self.graph.query(query)


# class OWLProcessor:
#     def __init__(self):
#         self.ontology = None
#
#     def load(self, path):
#         self.ontology = get_ontology(path).load()
#
#     def query(self, query):
#         # 实现查询逻辑
#         pass


class SequenceEncoder(object):
    """
    The `SequenceEncoder` encodes raw column strings into embeddings.
    """
    def __init__(self, model_name='uer/sbert-base-chinese-nli', device=None, cache_folder="./"):
        self.device = device
        # encode anything to a (768,) tensor
        self.model = SentenceTransformer(model_name, device=device, cache_folder=cache_folder)

    @torch.no_grad()
    def __call__(self, df):
        x = self.model.encode(df.values, show_progress_bar=True,
                              convert_to_tensor=True, device=self.device)
        return x.cpu()


class IdentityEncoder(object):
    """
    The `IdentityEncoder` takes the raw column values and converts them to PyTorch tensors.
    """

    def __init__(self, dtype=None, is_list=False):
        self.dtype = dtype
        self.is_list = is_list

    def __call__(self, df):
        if self.is_list:
            return torch.stack([torch.tensor(el) for el in df.values])
        return torch.from_numpy(df.values).to(self.dtype)

