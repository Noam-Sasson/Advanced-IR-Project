from sklearn.feature_extraction.text import TfidfVectorizer
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.cluster import KMeans
from tqdm import tqdm
import logging
import random
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
import torch
from transformers import BertModel, BertTokenizer, AutoModel, AutoTokenizer
import logging
import numpy as np
from sklearn.metrics import silhouette_score, mean_squared_error
import os
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
# import cupy as cp
# from cuml.feature_extraction.text import TfidfVectorizer

# Set up logging
logging.basicConfig(level=logging.INFO)


class Embedder:
    def __init__(self):
        pass

    def embed(self, chunks)->torch.Tensor:
        pass

class BertEmbedder(Embedder):
    def __init__(self, show_prints=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        if show_prints:
            print(f'BertEmbedder Using device: {self.device}')

    def embed(self, chunks):
        encodings = self.tokenizer.batch_encode_plus(chunks, padding=True, truncation=True, return_tensors='pt', add_special_tokens=True).to(self.device)
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state.mean(dim=1)

        return embeddings.to('cpu')

# class E5Embedder(Embedder):
#     def __init__(self, model_name="intfloat/e5-base", show_prints=True):
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model = AutoModel.from_pretrained(model_name).to(self.device)
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         if show_prints:
#             print(f'E5Embedder Using device: {self.device}')

#     def embed(self, chunks, is_query=True):
#         # E5 requires prefixing "query: " or "passage: " to text
#         prefix = "query: " if is_query else "passage: "
#         chunks = [prefix + chunk for chunk in chunks]

#         encodings = self.tokenizer.batch_encode_plus(
#             chunks, padding=True, truncation=True, return_tensors='pt'
#         ).to(self.device)

#         input_ids = encodings['input_ids']
#         attention_mask = encodings['attention_mask']

#         with torch.no_grad():
#             outputs = self.model(input_ids, attention_mask=attention_mask)
#             embeddings = outputs.last_hidden_state[:, 0]  # Use [CLS] token embedding

#         return embeddings.to('cpu')

# class TFIDFEmbedder(Embedder):
#     def __init__(self, corpus):
#         self.vectorizer = TfidfVectorizer()
#         self.vectorizer.fit(corpus)

#     def embed(self, chunks):
#         embeddings = self.vectorizer.transform(chunks).toarray()
#         return torch.tensor(embeddings)

class E5Embedder(Embedder):
    def __init__(self, model_name="intfloat/e5-base", show_prints=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if show_prints:
            print(f'E5Embedder Using device: {self.device}')

    def embed(self, batch_chunks, is_query=True):
        """Embed a batch of document chunks using E5 model."""
        prefix = "query: " if is_query else "passage: "
        batch_chunks = [[prefix + chunk for chunk in chunks] for chunks in batch_chunks]
        flat_chunks = [chunk for chunks in batch_chunks for chunk in chunks]  # Flatten for batch processing

        encodings = self.tokenizer.batch_encode_plus(
            flat_chunks, padding=True, truncation=True, return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(encodings['input_ids'], attention_mask=encodings['attention_mask'])
            flat_embeddings = outputs.last_hidden_state[:, 0].cpu()  # Use [CLS] token embedding

        # Reshape back to batch structure
        index = 0
        batch_embeddings = []
        for chunks in batch_chunks:
            batch_embeddings.append(flat_embeddings[index:index+len(chunks)])
            index += len(chunks)

        return batch_embeddings  # List of lists of embeddings

class TFIDFEmbedder(Embedder):
    def __init__(self, corpus):
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(corpus)

    def embed(self, batch_chunks):
        """Embed a batch of document chunks using TF-IDF."""
        batch_embeddings = []
        for chunks in batch_chunks:
            embeddings = self.vectorizer.transform(chunks).toarray()
            batch_embeddings.append(torch.tensor(embeddings))  # Convert to tensor
        return batch_embeddings  # List of tensors

class GPU_TFIDFEmbedder:
    def __init__(self, corpus):
        """Use CPU-based TF-IDF, then move to GPU."""
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(corpus)

    def embed(self, batch_chunks):
        """Compute TF-IDF on CPU, then move to GPU."""
        all_chunks = [" ".join(chunks) for chunks in batch_chunks]
        tfidf_embeddings = self.vectorizer.transform(all_chunks).toarray()

        # Move data to GPU manually
        tfidf_tensor = torch.tensor(tfidf_embeddings, dtype=torch.float32).to("cuda")
        return tfidf_tensor

# class GPU_TFIDFEmbedder_V7:
#     def __init__(self, corpus):
#         """Initialize the TF-IDF vectorizer on GPU."""
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.vectorizer = TfidfVectorizer()
#         self.vectorizer.fit(corpus)  # Fit on the full corpus (CPU but optimized)

#     def embed(self, batch_chunks):
#         """Compute TF-IDF embeddings on GPU for a batch of documents."""
#         all_chunks = [" ".join(chunks) for chunks in batch_chunks]  # Join chunks per doc
#         tfidf_embeddings = self.vectorizer.transform(all_chunks)  # Still CPU-based
#         tfidf_embeddings_gpu = cp.asarray(tfidf_embeddings.toarray())  # Move to GPU
        
#         return torch.tensor(tfidf_embeddings_gpu, device=self.device)  # Convert to PyTorch tensor

class KnnCluster:
    def __init__(self, k=5, distance_matrix = None, distance_metric = 'euclidean'):
        self.k = k
        self.distance_metric = distance_metric
        self.distance_matrix = distance_matrix

    def fit(self, docs_centers_list, docs_list):
        self.docs_centers_list = docs_centers_list
        self.docs_list = docs_list

        
        # doc_ids = [doc['docid'] for doc in self.docs_list]
        # centers_ids = [doc['docid'] for doc in self.docs_centers_list]

        # Compute pairwise distances
        if self.distance_matrix is not None:
            distances = self.distance_matrix
        else:
            docs_embeddings = torch.stack([doc['d1_combined_embedding'] for doc in self.docs_list])
            centers_embeddings = torch.stack([doc['d1_combined_embedding'] for doc in self.docs_centers_list])
            distances = torch.cdist(centers_embeddings, docs_embeddings)

        for i, center in enumerate(self.docs_centers_list):
            center['neighbors'] = []
            for j, other_doc in enumerate(self.docs_list):
                if i != j:
                    distance = distances[i, j]
                    center['neighbors'].append((self.docs_list[j], distance))

            center['neighbors'] = sorted(center['neighbors'], key=lambda x: x[1])[:self.k]

        return self.docs_centers_list

from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import numpy as np
from collections import defaultdict

class ClusterBasedRetriever:
    def __init__(self, docs_list_with_neighbors, lambda_ = 0.9):
        self.center_docs_list = docs_list_with_neighbors.copy()
        seen_ids = set()
        docs_list = []
        for center_doc in self.center_docs_list:
            if center_doc['docid'] not in seen_ids:
                docs_list.append(center_doc)
                seen_ids.add(center_doc['docid'])
            
            for doc, dist in center_doc['neighbors']:
                if doc['docid'] not in seen_ids:
                    docs_list.append(doc)
                    seen_ids.add(doc['docid'])

        self.docs_list = docs_list
        self.corpus = [doc['contents'] for doc in self.docs_list]

        # Initialize CountVectorizer with stemming
        stemmer = PorterStemmer()
        def stem_tokenizer(text):
            tokens = word_tokenize(text)
            return [stemmer.stem(token) for token in tokens]
        
        
        # Initialize CountVectorizer
        self.vectorizer = CountVectorizer(stop_words='english', tokenizer=stem_tokenizer)  # You can add stopwords removal with stop_words='english'
        self.vectorizer.fit(self.corpus)  # Fit vectorizer on corpus

        # Compute corpus-level language model
        self.corpus_language_model = self.calculate_corpus_language_model()

        self.docs_lm_dict = dict()

        # Compute document-level language models
        for doc in self.docs_list:
            doc_lm = self.calculate_document_language_model(doc)
            doc_lm = self.jelinek_mercer_smoothing(doc_lm, lambda_)
            self.docs_lm_dict[doc['docid']] = doc_lm

        # compute geometric mean for each cluster
        def _geometric_mean(dicts):
            """Compute the geometric mean of probabilities for the same words across multiple dictionaries."""
            # Extract all unique words
            all_words = set()
            for d in dicts:
                all_words.update(d.keys())

            # Initialize a dictionary to store the geometric mean
            geo_mean_dict = {}

            for word in all_words:
                # Extract the probabilities for the current word across all dictionaries
                probs = [d[word] for d in dicts if word in d]

                # Convert to numpy array
                probs = np.array(probs)

                # Compute the geometric mean
                log_probs = np.log(probs)  # Take log to prevent underflow
                mean_log = np.mean(log_probs)  # Compute mean of logs
                geo_mean_dict[word] = np.exp(mean_log)  # Take exp to get the geometric mean

            return geo_mean_dict
        
        def compute_combined_lm(docs, vectorizer):
            """Compute the combined language model for a list of documents."""
            combined_lm = defaultdict(lambda: 0)
            total_words = 0
            for doc in docs:
                doc_text = [doc['contents']]  # Needs to be a list for vectorizer
                doc_term_matrix = self.vectorizer.transform(doc_text).toarray()[0]  # Convert to vector
                
                # Total words in document
                total_words += np.sum(doc_term_matrix)

                # Compute probabilities with Laplace smoothing
                word_count = {
                    word: (doc_term_matrix[idx])
                    for word, idx in self.vectorizer.vocabulary_.items()
                }

                for word, count in word_count.items():
                    combined_lm[word] += count

            for word in combined_lm.keys():
                combined_lm[word] /= total_words

            # print(combined_lm)
            # # smooth using jelinek-mercer smoothing
            combined_lm = self.jelinek_mercer_smoothing(combined_lm, lambda_)

            # print(combined_lm)

            return combined_lm


        for doc in self.center_docs_list:
            cluster_entities = [self.docs_lm_dict[doc_n['docid']] for doc_n, _ in doc['neighbors']] + [self.docs_lm_dict[doc['docid']]]
            doc['geometric_mean'] = _geometric_mean(cluster_entities)

            # doc['geometric_mean'] = compute_combined_lm([doc_n for doc_n, _ in doc['neighbors']] + [doc], self.vectorizer)
        
    def calculate_corpus_language_model(self):
        """Calculate the corpus-wide unigram language model using CountVectorizer."""
        # Convert corpus to document-term matrix
        doc_term_matrix = self.vectorizer.transform(self.corpus).toarray()
        total_word_counts = np.sum(doc_term_matrix, axis=0)  # Sum word counts across all docs
        
        # Total words in corpus
        total_words = np.sum(total_word_counts)
        
        # Vocabulary size
        vocab_size = len(self.vectorizer.vocabulary_)

        # Compute probabilities with Laplace smoothing
        language_model = {
            word: (total_word_counts[idx]) / (total_words)
            for word, idx in self.vectorizer.vocabulary_.items()
        }
        return language_model

    def calculate_document_language_model(self, doc):
        """Calculate the unigram language model for a given document using CountVectorizer."""
        doc_text = [doc['contents']]  # Needs to be a list for vectorizer
        doc_term_matrix = self.vectorizer.transform(doc_text).toarray()[0]  # Convert to vector
        
        # Total words in document
        total_words = np.sum(doc_term_matrix)
        
        # Vocabulary size (use corpus vocabulary for consistency)
        vocab_size = len(self.vectorizer.vocabulary_)

        # Compute probabilities with Laplace smoothing
        language_model = {
            word: (doc_term_matrix[idx]) / (total_words)
            for word, idx in self.vectorizer.vocabulary_.items()
        }
        return language_model

    def jelinek_mercer_smoothing(self, doc_lm, lambda_=0.7):
        """Jelinek-Mercer smoothing for a given document's language model."""
        smoothed_lm = {
            word: lambda_ * doc_lm[word] + (1 - lambda_) * self.corpus_language_model[word]
            for word in doc_lm
        }
        return smoothed_lm
    
    def calc_kld(self, p, q):
        """Calculate the Kullback-Leibler divergence between two language models."""
        kld = sum(p[word] * np.log2(p[word] / q[word]) for word in p)
        return kld
    
    def retrieve(self, query, m=10, lambda_=0.7, method = 'bag', ret_size = 50): # method can be 'bag' or 'set'
        """
        Do cluster based retrival given a query.
        for query-document similarity kld is used with jelelinek-mercer smoothing, where the 'corpus' is all the documents in all the clusters.
        same goes for cluster-documents similarity, where the geometric mean of a cluster is used as its language model.

        m: number of top m clusters to be used for constracting facest
        lambda_: smoothing parameter for jelinek-mercer smoothing
        method: method to use for constructing facets, can be 'bag' or 'set'
        """
        query_lm = self.calculate_document_language_model({'contents': query})
        query_lm = self.jelinek_mercer_smoothing(query_lm, lambda_)

        self.query_geo_mean_sim = {doc['docid']: (np.exp(-self.calc_kld(query_lm, doc['geometric_mean'])),{doc_n[0]['docid'] for doc_n in doc['neighbors']}) for doc in self.center_docs_list}

        # retrieve using fecets bag select and set select

        top_m_clusters = sorted(self.query_geo_mean_sim.items(), key=lambda x: x[1][0], reverse=True)[:m]

        # for cl in top_m_clusters:
        #     print(f'doc id: {cl[0]} | cluster score:{cl[1][0]}')
        #     print(f'->: {cl[1][1]}')

        self.factor_values = defaultdict(lambda: 0)

        if method == 'bag':
            for center_doc_id, (sim_value, docs_ids) in top_m_clusters:
                for doc_id in docs_ids:
                    if doc_id not in self.factor_values.keys():
                        self.factor_values[doc_id] = 0
                    self.factor_values[doc_id] += 1

        elif method == 'set':
            for center_doc_id, (sim_value, docs_ids) in top_m_clusters:
                for doc_id in docs_ids:
                    self.factor_values[doc_id] = 1
        else:
            raise ValueError(f'Method {method} is not supported')
        
        self.docs_score_list = [(doc, self.factor_values[doc['docid']]*np.exp(-self.calc_kld(query_lm, self.docs_lm_dict[doc['docid']])), doc['original_score']) for doc in self.docs_list]
        self.docs_score_list = sorted(self.docs_score_list, key=lambda x: (x[1], x[2]), reverse=True)

        # for doc, score, original_score in self.docs_score_list:
        #     print(f"Doc ID: {doc['docid']} | Score: {score:.4f} | Original Score: {original_score:.4f}")
        
        return self.docs_score_list[:ret_size]
    
class CustomClusteringModel:
    coplase_types = ['mean', 'max', 'min', 'sum']
    weights_type = ['no_weights', 'inv_sqrt_len']

    def __init__(self, window_size, show_prints = True):
        self.window_size = window_size
        self.docs = []

        self.show_prints = show_prints
        
    def data(self, dataset):
        self.dataset = dataset

    # def fit(self, embedders_dict):
    #     self.embbeders_to_idx = {embedder_name: idx for idx, embedder_name in enumerate(embedders_dict.keys())}
    #     self.mean_non_zero_entries = {embedder_name: 0 for embedder_name in embedders_dict.keys()}
    #     mean_non_zero_entries_calc = {embedder_name: {'count': 0, 'sum': 0} for embedder_name in embedders_dict.keys()}
    #     embedders = list(embedders_dict.values())
    #     def _process_doc(doc):
    #         try:
    #             chunks = self._split_doc_by_window(doc['contents'], self.window_size)
    #             embeddings_ = []

    #             if len(chunks) != 0:  # somehow there are empty documents
    #                 for embedder in embedders:
    #                     embed = embedder.embed(chunks)
    #                     embeddings_.append(embed)

    #             return embeddings_, doc
    #         except Exception as e:
    #             logging.error(f"Error processing document: {e}")
    #             return None, None

    #     max_threads = min(12, os.cpu_count())
    #     with ThreadPoolExecutor(max_workers=max_threads) as executor:
    #         futures = [executor.submit(_process_doc, doc) for doc in self.dataset]
    #         if self.show_prints:
    #             for future in tqdm(as_completed(futures), total=len(futures), desc="Processing documents"):
    #                 embeddings_, doc = future.result()
    #                 doc['embeddings'] = embeddings_
    #                 self.docs.append(doc)
    #         else:
    #             for future in as_completed(futures):
    #                 embeddings_, doc = future.result()
    #                 doc['embeddings'] = embeddings_
    #                 self.docs.append(doc)

    #     docs_embeddings = [doc['embeddings'] for doc in self.docs]
    #     for embeddings in docs_embeddings:
    #         for embedding_, name in zip(embeddings, embedders_dict.keys()):
    #             mean_non_zero_entries_calc[name]['count'] += embedding_.shape[0]
    #             mean_non_zero_entries_calc[name]['sum'] += torch.count_nonzero(embedding_).item()
            
    #     for name, embedder in embedders_dict.items():
    #         self.mean_non_zero_entries[name] = mean_non_zero_entries_calc[name]['sum'] / mean_non_zero_entries_calc[name]['count']
    #         print(f'Mean non zero entries for {name}: {self.mean_non_zero_entries[name]}')

    def fit(self, embedders_dict, batch_size=5, max_threads=10):
        self.embbeders_to_idx = {embedder_name: idx for idx, embedder_name in enumerate(embedders_dict.keys())}
        self.mean_non_zero_entries = {embedder_name: 0 for embedder_name in embedders_dict.keys()}
        mean_non_zero_entries_calc = {embedder_name: {'count': 0, 'sum': 0} for embedder_name in embedders_dict.keys()}
        embedders = list(embedders_dict.values())

        def _process_batch(docs_batch):
            """ Process a batch of documents by splitting and embedding them. """
            try:
                batch_chunks = []
                batch_docs = []
                for doc in docs_batch:
                    chunks = self._split_doc_by_window(doc['contents'], self.window_size)
                    if chunks:
                        batch_chunks.append(chunks)
                        batch_docs.append(doc)

                if not batch_chunks:
                    return None, None
                
                # Embed entire batch at once (each embedder should handle batching)
                embeddings_batch = [embedder.embed(batch_chunks) for embedder in embedders]

                for doc, embeddings in zip(batch_docs, zip(*embeddings_batch)):
                    doc['embeddings'] = list(embeddings)

                # Clear GPU memory
                torch.cuda.empty_cache()

                return batch_docs
            except Exception as e:
                logging.error(f"Error processing batch: {e}")
                return None

        # Use max available threads but don't exceed system limit
        max_threads = max_threads

        self.docs = []
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = [executor.submit(_process_batch, self.dataset[i:i+batch_size]) for i in range(0, len(self.dataset), batch_size)]

            if self.show_prints:
                futures = tqdm(as_completed(futures), total=len(futures), desc="Processing batches")

            for future in futures:
                batch_docs = future.result()
                if batch_docs:
                    self.docs.extend(batch_docs)

        # Compute mean non-zero entries efficiently
        docs_embeddings = [doc['embeddings'] for doc in self.docs]
        for embeddings in docs_embeddings:
            for embedding_, name in zip(embeddings, embedders_dict.keys()):
                mean_non_zero_entries_calc[name]['count'] += embedding_.shape[0]
                mean_non_zero_entries_calc[name]['sum'] += torch.count_nonzero(embedding_).item()
        
        for name in embedders_dict.keys():
            if mean_non_zero_entries_calc[name]['count'] > 0:
                self.mean_non_zero_entries[name] = mean_non_zero_entries_calc[name]['sum'] / mean_non_zero_entries_calc[name]['count']
                if self.show_prints:
                    print(f'Mean non-zero entries for {name}: {self.mean_non_zero_entries[name]}')

    def combine_embeddings_per_doc(self, weights_type='no_weights', embbeders_names=None):
        new_docs = []
        for doc in self.docs:
            embeddings_ = doc['embeddings']
            if len(embeddings_) != 0: # remember that there are empty documents
                
                if embbeders_names is not None:
                    embeddings_ = [embeddings_[self.embbeders_to_idx[name]] for name in embbeders_names]
                # combine embeddings with weights
                if weights_type == 'no_weights':
                    if len(embeddings_) > 1:
                        combined_embedding = torch.cat(embeddings_, dim=1)
                elif weights_type == 'inv_sqrt_len':
                    if len(embeddings_) > 1:
                        # print('Using inv_sqrt_len')
                        combined_embedding = torch.cat([embeddings_[i] * self._inv_sqrt_len_weights(embeddings_[i], name) for i, name in zip(range(len(embeddings_)), embbeders_names)], dim=1)
                    else:
                        combined_embedding = embeddings_[0]#*self._inv_sqrt_len_weights(embeddings_)
                else:
                    raise ValueError(f'Weights type {weights_type} is not supported')
                
                doc['combined_embedding'] = combined_embedding
                new_docs.append(doc)

        self.docs = new_docs
      
    # def cluster_k_means(self, n, rand_state=None, colapse_type = None):
    #     self._create_D1_combined_embeddings(colapse_type)
    #     if rand_state is None:
    #         random_state = random.randint(0, 1000)
    #     else:
    #         random_state = rand_state
    #     print(f'Using random state: {random_state}')
    #     self.kmeans = KMeans(n_clusters=n, random_state=0).fit(self.D1_combined_embeddings)
    #     return self.kmeans

    # def cluster_DBSCAN(self, eps=0.5, min_samples=2, metric='euclidean', colapse_type = None):
    #     if metric == 'precomputed':
    #         dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    #         self.dbscan_labels = dbscan.fit_predict(1-self.cosine_sim_matrix)
    #     else:
    #         self._create_D1_combined_embeddings(colapse_type)
    #         dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    #         self.dbscan_labels = dbscan.fit_predict(self.D1_combined_embeddings)
    #     return self.dbscan_labels

    # def cluster_agglomerative(self, n_clusters=2, metric='euclidean', linkage='ward', colapse_type=None):
    #     if metric == 'precomputed':
    #         agglomerative = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage=linkage)
    #         self.agglomerative_labels = agglomerative.fit_predict(1-self.cosine_sim_matrix)
    #     else:
    #         self._create_D1_combined_embeddings(colapse_type)
    #         agglomerative = AgglomerativeClustering(n_clusters=n_clusters, metric=metric, linkage=linkage)
    #         self.agglomerative_labels = agglomerative.fit_predict(self.D1_combined_embeddings)
    #     return self.agglomerative_labels
    
    def claster_knn(self, k = 5, clusters_num = 50, use_sim_mat = False, metric='euclidean', colapse_type= 'sum'):
        '''
        k: number of neighbors
        clusters_num: number of clusters - top clusters_num documents will be used as centers
        use_sim_mat: if True, will use similarity matrix to compute distances
        metric: metric to use for distance calculation (only in case use_sim_mat is True), can be 'cosine' or 'euclidean'
        colapse_type: type of collapsing embeddings, can be 'mean', 'max', 'min', 'sum'. means different things when use_sim_mat is True and False.
        '''
        
        if not use_sim_mat:
            self._create_D1_combined_embeddings(colapse_type)
            self.center_docs = KnnCluster(k=k, distance_matrix=None).fit(self.docs[:clusters_num], self.docs) # docs already ordered by original retrieval score
        else:
            if metric == 'cosine':
                self.create_sim_matrix(clusters_num, colapse_type)
                # print(self.cosine_sim_matrix.shape)
                self.center_docs = KnnCluster(k=k, distance_matrix=1-self.cosine_sim_matrix).fit(self.docs[:clusters_num], self.docs)
            elif metric == 'euclidean':
                self.create_dist_matrix(clusters_num, colapse_type)
                self.center_docs = KnnCluster(k=k, distance_matrix=self.dist_matrix).fit(self.docs[:clusters_num], self.docs)
            else:
                raise ValueError(f'Metric {metric} is not supported')
            
        return self.center_docs

    def create_sim_matrix(self, clusters_num = None, type='max'):
        if clusters_num is None:
            clusters_num = len(self.docs)

        self.cosine_sim_matrix = torch.zeros(clusters_num, len(self.docs))
        for i, doc_1 in enumerate(self.docs[:clusters_num]):
            for j, doc_2 in enumerate(self.docs[i:clusters_num]):
                if i != j+i:
                    cos_sim_matrix = self._get_cos_sim_mat(doc_1['combined_embedding'], doc_2['combined_embedding'])
                else:
                    cos_sim_matrix = torch.zeros(1, 1)

                if type == 'max':
                    self.cosine_sim_matrix[i, j+i] = cos_sim_matrix.max()
                elif type == 'mean':
                    self.cosine_sim_matrix[i, j+i] = cos_sim_matrix.mean()
                elif type == 'min':
                    self.cosine_sim_matrix[i, j+i] = cos_sim_matrix.min()
                elif type == 'sum':
                    self.dist_matrix[i, j+i] = cos_sim_matrix.sum()
                else:
                    raise ValueError(f'Type {type} is not supported')

                self.cosine_sim_matrix[j+i, i] = self.cosine_sim_matrix[i, j+i]

        for i, doc_1 in enumerate(self.docs[:clusters_num]):
            for j, doc_2 in enumerate(self.docs[clusters_num:]):
                cos_sim_matrix = self._get_cos_sim_mat(doc_1['combined_embedding'], doc_2['combined_embedding'])
                if type == 'max':
                    self.cosine_sim_matrix[i, j+clusters_num] = cos_sim_matrix.max()
                elif type == 'mean':
                    self.cosine_sim_matrix[i, j+clusters_num] = cos_sim_matrix.mean()
                elif type == 'min':
                    self.cosine_sim_matrix[i, j+clusters_num] = cos_sim_matrix.min()
                elif type == 'sum':
                    self.cosine_sim_matrix[i, j+clusters_num] = cos_sim_matrix.sum()
                else:
                    raise ValueError(f'Type {type} is not supported')
                  
    def create_dist_matrix(self, clusters_num = None, type='min'):
        if clusters_num is None:
            clusters_num = len(self.docs)

        self.dist_matrix = torch.zeros(clusters_num, len(self.docs))
        for i, doc_1 in enumerate(self.docs[:clusters_num]):
            for j, doc_2 in enumerate(self.docs[i:clusters_num]):
                if i != j+i:
                    dist = self._get_dist_mat(doc_1['combined_embedding'], doc_2['combined_embedding'])
                else:
                    dist = torch.zeros(1, 1)

                if type == 'max':
                    self.dist_matrix[i, j+i] = dist.max()
                elif type == 'mean':
                    self.dist_matrix[i, j+i] = dist.mean()
                elif type == 'min':
                    self.dist_matrix[i, j+i] = dist.min()
                elif type == 'sum':
                    self.dist_matrix[i, j+i] = dist.sum()
                else:
                    raise ValueError(f'Type {type} is not supported')
                
                self.dist_matrix[j+i, i] = self.dist_matrix[i, j+i]

        for i, doc_1 in enumerate(self.docs[:clusters_num]):
            for j, doc_2 in enumerate(self.docs[clusters_num:]):
                dist = self._get_dist_mat(doc_1['combined_embedding'], doc_2['combined_embedding'])
                if type == 'max':
                    self.dist_matrix[i, j+clusters_num] = dist.max()
                elif type == 'mean':
                    self.dist_matrix[i, j+clusters_num] = dist.mean()
                elif type == 'min':
                    self.dist_matrix[i, j+clusters_num] = dist.min()
                elif type == 'sum':
                    self.dist_matrix[i, j+clusters_num] = dist.sum()
                else:
                    raise ValueError(f'Type {type} is not supported')
        
    def _create_D1_combined_embeddings(self, colapse_type):
         # collapse embeddings
        if colapse_type == 'mean' or colapse_type is None:
            # print(combined_embedding.shape)
            combined_embeddings = [doc['combined_embedding'].mean(dim=0) for doc in self.docs]
            # print(combined_embedding.shape)
        elif colapse_type == 'max':
            combined_embeddings = [doc['combined_embedding'].max(dim=0).values for doc in self.docs]
        elif colapse_type == 'min':
            combined_embeddings = [doc['combined_embedding'].min(dim=0).values for doc in self.docs]
        elif colapse_type == 'sum':
            combined_embeddings = [doc['combined_embedding'].sum(dim=0) for doc in self.docs]
        else:
            raise ValueError(f'Colapse type {colapse_type} is not supported')
        
        for doc, combined_embedding_ in zip(self.docs, combined_embeddings):
            doc['d1_combined_embedding'] = combined_embedding_      
      
    # def _split_doc_by_window(self, doc, window_size):
    #     # Initialize the BERT tokenizer
    #     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    #     # Tokenize the document
    #     tokens = tokenizer.tokenize(doc)
    #     chunks = []
    #     chunk = []
    #     current_chunk_size = 0

    #     for token in tokens:
    #         chunk.append(token)
    #         current_chunk_size += 1

    #         if token == '.' and current_chunk_size >= window_size:
    #             chunks.append(' '.join(chunk))
    #             chunk = []
    #             current_chunk_size = 0

    #     if len(chunk) > 0:
    #         chunks.append(' '.join(chunk))

    #     return chunks

    def _split_doc_by_window(self, doc, window_size):
        """
        Splits a document into chunks of approximately `window_size` words, 
        ensuring that chunks end at sentence boundaries if possible.
        """
        # Split the document into sentences using regex
        sentences = re.split(r'(?<=[.!?])\s+', doc.strip())  # Keep sentence boundaries
        chunks = []
        chunk = []
        current_chunk_size = 0

        for sentence in sentences:
            words = sentence.split()  # Split sentence into words
            sentence_length = len(words)

            # If adding the sentence exceeds window size, store current chunk
            if current_chunk_size + sentence_length > window_size:
                if chunk:  # Store only if chunk is non-empty
                    chunks.append(" ".join(chunk))
                chunk = words  # Start new chunk
                current_chunk_size = sentence_length
            else:
                chunk.extend(words)
                current_chunk_size += sentence_length

        # Append any remaining words as the last chunk
        if chunk:
            chunks.append(" ".join(chunk))

        return chunks
   
    def _get_cos_sim_mat(self, embedding_1, embedding_2):
        cos_sim_matrix = embedding_1 @ embedding_2.T

        norm_1 = torch.norm(embedding_1, dim=1)
        norm_2 = torch.norm(embedding_2, dim=1)

        norm_mult_div = norm_1.reshape(-1, 1) @ norm_2.reshape(-1, 1).T

        cos_sim_matrix = cos_sim_matrix / norm_mult_div

        return cos_sim_matrix

    def _get_dist_mat(self, embedding_1, embedding_2):
        dist_matrix = torch.zeros(len(embedding_1), len(embedding_2))

        for i, row_vec in enumerate(embedding_1):
            for j, roe_vec in enumerate(embedding_2):
                dist = torch.norm((row_vec - roe_vec))**2
                dist_matrix[i, j] = dist

        return dist_matrix
    
    def _inv_sqrt_len_weights(self, embeddings, name):
        # print(len(embeddings[0]))
        return 1/((self.mean_non_zero_entries[name])**0.5)
        # return 1/(len(embeddings[0])**0.5)