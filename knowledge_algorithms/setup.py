from setuptools import setup, find_packages

setup(
    name='knowledge_algorithms',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A package for knowledge graph algorithms',
    packages=find_packages(),
    install_requires=[
        'torch',   # 如果您的包依赖于 PyTorch
        # 'tensorflow',  # 如果您的包依赖于 TensorFlow
        # 'jax',     # 如果您的包依赖于 JAX
        'rdflib',  # 如果您的包依赖于 RDFLib
        'owlready2',  # 如果您的包依赖于 Owlready2
        'neo4j',
        'sentence_transformers',
        'pandas'
        # 其他依赖项
    ],
)
