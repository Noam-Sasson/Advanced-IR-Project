# Advanced-IR-Project

### description
This repository contains the code for the Advanced Information Retrieval project. The project is focused on building a search engine that can retrieve relevant documents based on user queries. The search engine uses cluster based retrieval and clustering techniques to improve the relevance of the retrieved documents.


### how to run the code
the main file to run is the *retrieval_pyserini.py* file, which contains the function *run_final_cluster_based_retreival* that handles the retrieval process with option for hyperparameter tuning. The project also includes the topics and queries files used in our experiments, as well as the results of the experiments in the *saved runs* repository.

### acknowledgements

all topics and qrels were taken from the anserini-tools repo which can be found in the following link: https://github.com/castorini/anserini-tools/tree/1c463184d53d3735c3f0bcee2c3e9509be83973d

we used the pyserini library for the initial bm25 retrieval process, and their pre-indexed msmarco-passage and msmarco-doc datasets.
link to their repo:

https://github.com/castorini/pyserini

### dependencies
You can install all required dependencies with the following command:
```bash
pip install pyserini scikit-learn nltk torch transformers tqdm numpy
```