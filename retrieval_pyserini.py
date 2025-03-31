import subprocess
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm
import csv
import re
from cluster_based_retrieval_tools import E5Embedder, TFIDFEmbedder, CustomClusteringModel, ClusterBasedRetriever, GPU_TFIDFEmbedder
import nltk
import torch
from transformers import AutoTokenizer

def run_regular_retrieval_process(k = 50, corpus = 'msmarco-passage', more_queries = False):
    # 1. Load queries from 'queries.dev.tsv'
    queries = {}

    if not more_queries:
        queries_file = 'topics.msmarco-passage.dev-subset.txt' if corpus == 'msmarco-passage' else 'topics.msmarco-doc.dev.txt'
        qrels_file = 'qrels.msmarco-passage.dev-subset.txt' if corpus == 'msmarco-passage' else 'qrels.msmarco-doc.dev.txt'
        run_file = "run.msmarco-passage.dev-subset.txt" if corpus == 'msmarco-passage' else 'run.msmarco-doc.dev.txt'
    else:
        queries_file = 'topics.dl19-passage.txt' if corpus == 'msmarco-passage' else 'topics.dl20.subset.txt'
        qrels_file = 'qrels.dl19-passage.txt' if corpus == 'msmarco-passage' else 'qrels.dl20-doc.subset.txt'
        run_file = "run.dl19-passage.txt" if corpus == 'msmarco-passage' else 'run.dl20-doc.subset.txt'

    with open(queries_file) as f:
        for line in f:
            qid, query_text = line.strip().split('\t')  # Split by tab
            queries[qid] = query_text

    # 2. Load qrels from qrels_file
    qrels = {}
    if corpus == 'msmarco-passage' or more_queries:
        with open(qrels_file) as f:
            for line in f:
                qid, _, docid, relevance = line.strip().split(' ')  # Split by tab
                qrels[(qid, docid)] = int(relevance)
    else:
        with open(qrels_file) as f:
            for line in f:
                # print(line)
                qid, _, docid, relevance = line.strip().split('\t')
                qrels[(qid, docid)] = int(relevance)

    # Check loaded queries and qrels (optional)
    print("Loaded queries:", list(queries.items())[:5])
    print("Loaded qrels:", list(qrels.items())[:5])

    # 3. Initialize the LuceneSearcher and set BM25 parameters
    if corpus == 'msmarco-doc':
        searcher = LuceneSearcher.from_prebuilt_index('msmarco-doc')
    else:
        searcher = LuceneSearcher.from_prebuilt_index('msmarco-passage')
    searcher.set_bm25(k1=0.9, b=0.4)

    # 4. Open a file to save the retrieval results
    with open(run_file, "w") as f:
        for qid, query in tqdm(queries.items(), desc="Retrieving"):
            hits = searcher.search(query, k=k)  # Retrieve top 50 hits

            # Batch retrieve documents instead of calling searcher.doc() in a loop
            docids = [hit.docid for hit in hits]  # Collect all doc IDs
            docs = searcher.batch_doc(docids, threads=16)  # Batch retrieval

            for rank, hit in enumerate(hits):
                # doc = searcher.doc(hit.docid)  # Retrieve document content
                # doc_text = doc.contents if doc else "N/A"  # Handle missing docs
                f.write(f"{qid} Q0 {hit.docid} {rank+1} {hit.score} BM25\n")

    print("Retrieval completed. Results saved to run_file.")

    # 5. Evaluate the retrieval results using 'trec_eval'
    # qrels_file = qrels_file  # Path to your qrels file
    # run_file = run_file  # Path to your retrieval results file

    # Run trec_eval with various metrics (e.g., NDCG, MAP, etc.)
    command = [
    'python', '-m', 'pyserini.eval.trec_eval', 
    '-c', 
    '-m', 'ndcg_cut.10', 
    '-m', 'recip_rank', 
    '-m', 'recall.50', 
    '-m', 'map',
    '-m', 'P.5',
    qrels_file, 
    run_file
    ]

    # # Execute the evaluation command
    # subprocess.run(command)
    res_file = f"res_regular_retrival_{corpus}"
    with open(res_file, "w") as f:
        subprocess.run(command, stdout=f, stderr=f, text=True)

def run_experimental_cluster_based_retrirval(corpus = 'msmarco-passage', more_queries = False):
    # 1. Load queries from 'queries.dev.tsv'
    queries = {}

    if not more_queries:
        queries_file = 'topics.msmarco-passage.dev-subset.txt' if corpus == 'msmarco-passage' else 'topics.msmarco-doc.dev.txt'
        qrels_file = 'qrels.msmarco-passage.dev-subset.txt' if corpus == 'msmarco-passage' else 'qrels.msmarco-doc.dev.txt'
        run_file = "run.msmarco-passage.dev-subset.clr-50.m-10.k-5.bag.txt" if corpus == 'msmarco-passage' else 'run.msmarco-doc.dev.clr-50.m-10.k-5.bag.txt'
    else:
        queries_file = 'topics.dl19-passage.txt' if corpus == 'msmarco-passage' else 'topics.dl20.txt'
        qrels_file = 'qrels.dl19-passage.txt' if corpus == 'msmarco-passage' else 'qrels.dl20-doc.txt'
        run_file = "run.dl19-passage.clr-50.m-10" if corpus == 'msmarco-passage' else 'run.dl20-doc.clr-50.m-10'

    with open(queries_file) as f:
        for line in f:
            qid, query_text = line.strip().split('\t')  # Split by tab
            queries[qid] = query_text

    # 2. Load qrels from qrels_file
    qrels = {}
    if corpus == 'msmarco-passage' or more_queries:
        with open(qrels_file) as f:
            for line in f:
                qid, _, docid, relevance = line.strip().split(' ')  # Split by tab
                qrels[(qid, docid)] = int(relevance)
    else:
        with open(qrels_file) as f:
            for line in f:
                qid, _, docid, relevance = line.strip().split('\t')
                qrels[(qid, docid)] = int(relevance)

    # Check loaded queries and qrels (optional)
    print("Loaded queries:", list(queries.items())[:5])
    print("Loaded qrels:", list(qrels.items())[:5])

    # 3. Initialize the LuceneSearcher and set BM25 parameters
    if corpus == 'msmarco-doc':
        searcher = LuceneSearcher.from_prebuilt_index('msmarco-doc')
    else:
        searcher = LuceneSearcher.from_prebuilt_index('msmarco-passage')
    searcher.set_bm25(k1=0.9, b=0.4)
    import cProfile
    import pstats
    import io
    import time
    # 4. Open a file to save the retrieval results
    with open(run_file, "w") as f:
        query_to_take = 3

        for qur_q, (qid, query) in enumerate(queries.items()):
            if qur_q < query_to_take:
                continue
            

            # ----- initial document retrieval ----- #
            current_query = query
            current_qid = qid
            hits = searcher.search(query, k=250)  # Retrieve top 50 hits
            
            start_t = time.time()
            # Batch retrieve documents instead of calling searcher.doc() in a loop
            docids = [hit.docid for hit in hits]  # Collect all doc IDs
            scores = [hit.score for hit in hits]
            docs = searcher.batch_doc(docids, threads=16)  # Batch retrieval

            # print('\n'.join(docs[docids[0]].raw().split('<TEXT>')[1].split('</TEXT>')[0].split('\n')[2:]))

            # ----- cluster based retrieval ----- #

            if corpus == 'msmarco-passage':
                top_50_docs_dicts = [{'docid':docid_, 'contents': docs[docid_].raw().split('"contents" : "')[1].split("}")[0][:-2], 'full_object': docs[docid_], 'original_score': score} for docid_, score in zip(docids, scores)]
            else:
                top_50_docs_dicts = [{'docid':docid_, 'contents': '\n'.join(docs[docid_].raw().split('<TEXT>')[1].split('</TEXT>')[0].split('\n')[2:]), 'full_object': docs[docid_], 'original_score': score} for docid_, score in zip(docids, scores)]
            train_corpus = [doc['contents'] for doc in top_50_docs_dicts]

            e5_embedder = E5Embedder()
            tfidf_embedder = TFIDFEmbedder(train_corpus)
            # tfidf_embedder = GPU_TFIDFEmbedder(train_corpus)

            embedders = {'tf-idf':tfidf_embedder, 'e5' :e5_embedder}
            # embedders = {'e5':e5_embedder}
            embbeders_names = ['tf-idf', 'e5']
            # embbeders_names = ['e5']


            # # Create a profile object
            # pr = cProfile.Profile()
            # pr.enable()

            general_clustering_model = CustomClusteringModel(window_size=500)
            general_clustering_model.data(top_50_docs_dicts)
            start_fit = time.time()
            general_clustering_model.fit(embedders)
            print(f"Fit took {time.time() - start_fit} seconds")
            general_clustering_model.combine_embeddings_per_doc(weights_type='inv_sqrt_len', embbeders_names=embbeders_names)

            def count_tokens(text, model_name="bert-base-uncased"):
                """
                Count the number of tokens in a given text using the specified tokenizer.

                Parameters:
                text (str): The input text to tokenize.
                model_name (str): The name of the model to use for tokenization.

                Returns:
                int: The number of tokens in the text.
                """
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                tokens = tokenizer.tokenize(text)
                return len(tokens)

            # pr.disable()

            # # Create a stream to hold the profiling results
            # s = io.StringIO()
            # sortby = 'cumulative'
            # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            # ps.print_stats()

            # # Print the profiling results
            # print(s.getvalue())
            
            # clusters = general_clustering_model.claster_knn(k = 5, clusters_num = 50, use_sim_mat = False, metric='euclidean', colapse_type= 'sum')
            clusters = general_clustering_model.claster_knn(k = 5, clusters_num = 50, use_sim_mat = True, metric='euclidean', colapse_type= 'min')

            # i = 0
            # for cl in clusters:
            #     print(cl['docid'], cl['original_score'])
            #     for n in cl['neighbors']:
            #         print('->',n[0]['docid'], n[0]['original_score'])
                
            #     i+=1
            #     if i == 5:
            #         break

            cluster_based_retriever = ClusterBasedRetriever(clusters, lambda_= 0.9)
            docs_score_list = cluster_based_retriever.retrieve(current_query, m=10, lambda_=0.9, method = 'set')

            final_t = time.time() - start_t
            print(f"Query {current_qid} took {final_t} seconds")

            print("query:", current_query)
            print("top 10 docs contents:")

            for i ,(doc, score, original_score) in enumerate(docs_score_list):
                print('docid:', doc['docid'], 'score:', score, 'original_score:', original_score)
                # print('content:', doc['contents'])
                print('-----------------------')

                if i == 9:
                    break

            print(count_tokens(docs_score_list[0][0]['contents']))
            print('-----------------------')
            print(count_tokens(docs_score_list[1][0]['contents']))
            print('-----------------------')
            print(count_tokens(docs_score_list[2][0]['contents']))
            print('-----------------------')
            print(count_tokens(docs_score_list[3][0]['contents']))
            print('-----------------------')
            

            # print('-----------------------')
            # print('true doc:')
            # for doc in top_50_docs_dicts:
            #     if doc['docid'] == '7187158':
            #         print('docid:', doc['docid'], 'score:', doc['original_score'])
            #         print('content:', doc['contents'])
            #         break
            # print('-----------------------')

            for rank, (doc, score, original_score) in enumerate(docs_score_list):
                # doc = searcher.doc(hit.docid)  # Retrieve document content
                # doc_text = doc.contents if doc else "N/A"  # Handle missing docs
                f.write(f"{current_qid} Q0 {doc['docid']} {rank+1} {score} BM25\n")

            break
        
    print("Retrieval completed. Results saved to run_file.")

    # 5. Evaluate the retrieval results using 'trec_eval'
    # qrels_file = qrels_file  # Path to your qrels file
    # run_file = run_file  # Path to your retrieval results file

    # Run trec_eval with various metrics (e.g., NDCG, MAP, etc.)
    command = [
    'python', '-m', 'pyserini.eval.trec_eval', 
    '-c', 
    '-m', 'ndcg_cut.10', 
    '-m', 'recip_rank', 
    '-m', 'recall.50', 
    '-m', 'map',
    '-m', 'P.5',
    qrels_file, 
    run_file
    ]

    # Execute the evaluation command
    output_file = f"exp_evaluation_results_{corpus}.txt"

    with open(output_file, "w") as f:
        subprocess.run(command, stdout=f, stderr=f, text=True)
    # print(general_clustering_model.docs[0])

def run_final_cluster_based_retreival(initial_retrival_size = 50, nieghbors_num = 5, clusters_num = 50, use_sim_mat = True, metric = 'euclidean', colapse_type = 'min', top_m_clusters = 10, lambda_ = 0.7, method = 'bag', embbeders_names = ['tf-idf', 'e5'], corpus = 'msmarco-passage', more_queries = False):
    '''
    This function runs the final cluster based retrieval process.
    It uses Our custom clustering model to cluster the top 50 documents retrieved by BM25.
    Then it uses the cluster based retriever we implemented to retrieve the top {clusters_num} documents from the initial 50 documents.
    The function saves the results to a file and then evaluates the results using trec_eval.

    Parameters:
    initial_retrival_size: int
        The number of documents to retrieve using BM25
    nieghbors_num: int
        The number of neighbors to consider when clustering the documents
    clusters_num: int
        The number of clusters to create
    use_sim_mat: bool
        Whether to use the similarity matrix when clustering the documents
    metric: str
        The metric to use when clustering the documents (e.g., 'euclidean', 'cosine')
    colapse_type: str
        The type of colapse to use when clustering the documents (e.g., 'mean', 'max', 'min', 'sum')
    top_m_clusters: int
        The number of clusters to retrieve documents from
    lambda_: float
        The lambda parameter for smoothed language model used for scoring the documents in the clusters
    method: str
        The method to use when retrieving documents from the clusters (e.g., 'bag', 'set')
    embbeders_names: list
        The names of the embedders to use when creating the embeddings for the documents
    corpus: str
        The corpus to be retrieved from (e.g., 'msmarco-passage', 'msmarco-doc-v1')
    '''
    
    # 1. Load queries from 'queries.dev.tsv'
    queries = {}
    if not more_queries:
        queries_file = f'topics.msmarco-passage.dev-subset.txt' if corpus == 'msmarco-passage' else 'topics.msmarco-doc.dev.txt'
        qrels_file = f'qrels.msmarco-passage.dev-subset.txt' if corpus == 'msmarco-passage' else 'qrels.msmarco-doc.dev.txt'
    else:
        queries_file = 'topics.dl19-passage.txt' if corpus == 'msmarco-passage' else 'topics.dl20.subset.txt'
        qrels_file = 'qrels.dl19-passage.txt' if corpus == 'msmarco-passage' else 'qrels.dl20-doc.subset.txt'

    sim_mat_val = 'T' if use_sim_mat else 'F'
    metric_val = 'euc' if metric == 'euclidean' else 'cos'
    more_queries_val = 'T' if more_queries else 'F'
    run_file = f"run.{corpus}.dev-subset.{method}.irs-{initial_retrival_size}.cln-{clusters_num}.k-{nieghbors_num}.m-{top_m_clusters}.sim_mat-{sim_mat_val}.met-{metric_val}.colp-{colapse_type}.l-{lambda_}.emb-{embbeders_names}.mq-{more_queries_val}.win-250.txt"

    with open(queries_file) as f:
        for line in f:
            qid, query_text = line.strip().split('\t')  # Split by tab
            queries[qid] = query_text

    # 2. Load qrels from qrels_file
    qrels = {}
    if corpus == 'msmarco-passage' or more_queries:
        with open(qrels_file) as f:
            for line in f:
                qid, _, docid, relevance = line.strip().split(' ')  # Split by tab
                qrels[(qid, docid)] = int(relevance)
    else:
        with open(qrels_file) as f:
            for line in f:
                qid, _, docid, relevance = line.strip().split('\t')
                qrels[(qid, docid)] = int(relevance)

    # Check loaded queries and qrels (optional)
    # print("Loaded queries:", list(queries.items())[:5])
    # print("Loaded qrels:", list(qrels.items())[:5])

    # 3. Initialize the LuceneSearcher and set BM25 parameters
    if corpus == 'msmarco-doc':
        searcher = LuceneSearcher.from_prebuilt_index('msmarco-doc')
    else:
        searcher = LuceneSearcher.from_prebuilt_index('msmarco-passage')
    searcher.set_bm25(k1=0.9, b=0.4)

    # 4. Open a file to save the retrieval results
    with open(run_file, "w") as f:
        for qid, query in tqdm(queries.items(), desc="Retrieving"):
            # ----- initial document retrieval ----- #
            hits = searcher.search(query, k=initial_retrival_size)  # Retrieve top 50 hits

            # Batch retrieve documents instead of calling searcher.doc() in a loop
            docids = [hit.docid for hit in hits]  # Collect all doc IDs
            scores = [hit.score for hit in hits]
            docs = searcher.batch_doc(docids, threads=16)  # Batch retrieval

            # ----- cluster based retrieval ----- #
            if corpus == 'msmarco-passage':
                top_50_docs_dicts = [{'docid':docid_, 'contents': docs[docid_].raw().split('"contents" : "')[1].split("}")[0][:-2], 'full_object': docs[docid_], 'original_score': score} for docid_, score in zip(docids, scores)]
            else:
                top_50_docs_dicts = [{'docid':docid_, 'contents': '\n'.join(docs[docid_].raw().split('<TEXT>')[1].split('</TEXT>')[0].split('\n')[2:]), 'full_object': docs[docid_], 'original_score': score} for docid_, score in zip(docids, scores)]
            train_corpus = [doc['contents'] for doc in top_50_docs_dicts]

            e5_embedder = E5Embedder(show_prints=False)
            tfidf_embedder = TFIDFEmbedder(train_corpus)

            embedders_dict = {'tf-idf':tfidf_embedder, 'e5' :e5_embedder}
            # embedders = {'tf-idf':tfidf_embedder}
            embedders = {key: embedders_dict[key] for key in embbeders_names}
            # embbeders_names = ['tf-idf']
            general_clustering_model = CustomClusteringModel(window_size=250, show_prints=False)
            general_clustering_model.data(top_50_docs_dicts)

            # Clear GPU memory
            torch.cuda.empty_cache()
            general_clustering_model.fit(embedders, batch_size=1, max_threads=1)
            # Clear GPU memory
            torch.cuda.empty_cache()
            general_clustering_model.combine_embeddings_per_doc(weights_type='inv_sqrt_len', embbeders_names=embbeders_names)
            
            # clusters = general_clustering_model.claster_knn(k = 5, clusters_num = 50, use_sim_mat = False, metric='euclidean', colapse_type= 'sum')
            clusters = general_clustering_model.claster_knn(k = nieghbors_num, clusters_num = clusters_num, use_sim_mat = use_sim_mat, metric = metric, colapse_type = colapse_type)
    
            cluster_based_retriever = ClusterBasedRetriever(clusters, lambda_= 0.7)
            docs_score_list = cluster_based_retriever.retrieve(query, m=top_m_clusters, lambda_=lambda_, method = method)

            
            for rank, (doc, score, original_score) in enumerate(docs_score_list):
                # doc = searcher.doc(hit.docid)  # Retrieve document content
                # doc_text = doc.contents if doc else "N/A"  # Handle missing docs
                f.write(f"{qid} Q0 {doc['docid']} {rank+1} {score} CBR\n")
        
            

    # import os
    # import torch
    # import logging
    # from concurrent.futures import ThreadPoolExecutor, as_completed
    # from tqdm import tqdm

    # def process_query(qid, query, searcher, initial_retrival_size, 
    #                 nieghbors_num, clusters_num, use_sim_mat, metric, colapse_type, 
    #                 top_m_clusters, lambda_, method):
    #     """Process a single query and return formatted results."""
    #     try:
    #         # ----- Initial document retrieval ----- #
    #         hits = searcher.search(query, k=initial_retrival_size)
    #         docids = [hit.docid for hit in hits]
    #         scores = [hit.score for hit in hits]
    #         docs = searcher.batch_doc(docids, threads=16)  # Batch retrieval

    #         # ----- Cluster-based retrieval ----- #
    #         top_50_docs_dicts = [{'docid': docid_, 
    #                             'contents': docs[docid_].raw().split('"contents" : "')[1].split("}")[0][:-2], 
    #                             'full_object': docs[docid_], 
    #                             'original_score': score} for docid_, score in zip(docids, scores)]
            
    #         train_corpus = [doc['contents'] for doc in top_50_docs_dicts]

    #         e5_embedder = E5Embedder(show_prints=False)
    #         tfidf_embedder = TFIDFEmbedder(train_corpus)

    #         # Embeddings
    #         embedders = {'tf-idf': tfidf_embedder, 'e5': e5_embedder}
    #         embbeders_names = ['tf-idf', 'e5']

    #         general_clustering_model = CustomClusteringModel(window_size=float('inf'), show_prints=False)
    #         general_clustering_model.data(top_50_docs_dicts)
    #         general_clustering_model.fit(embedders)
    #         general_clustering_model.combine_embeddings_per_doc(weights_type='inv_sqrt_len', embbeders_names=embbeders_names)

    #         # Clustering
    #         clusters = general_clustering_model.claster_knn(k=nieghbors_num, 
    #                                                         clusters_num=clusters_num, 
    #                                                         use_sim_mat=use_sim_mat, 
    #                                                         metric=metric, 
    #                                                         colapse_type=colapse_type)

    #         # Cluster-based retrieval
    #         cluster_based_retriever = ClusterBasedRetriever(clusters)
    #         docs_score_list = cluster_based_retriever.retrieve(query, m=top_m_clusters, lambda_=lambda_, method=method)

    #         # Format results
    #         return [f"{qid} Q0 {doc['docid']} {rank+1} {score} CBR\n" for rank, (doc, score, _) in enumerate(docs_score_list)]
        
    #     except Exception as e:
    #         logging.error(f"Error processing query {qid}: {e}")
    #         return []
        

    # # Parallel execution of queries
    # max_threads = 12
    # with ThreadPoolExecutor(max_workers=max_threads) as executor, open(run_file, "w") as f:
    #     futures = {executor.submit(process_query, qid, query, searcher, initial_retrival_size, 
    #                             nieghbors_num, clusters_num, use_sim_mat, metric, colapse_type, 
    #                             top_m_clusters, lambda_, method): qid for qid, query in queries.items()}
        
    #     for future in tqdm(as_completed(futures), total=len(futures), desc="Retrieving"):
    #         results = future.result()
    #         f.writelines(results)  # Write results to file in parallel
        
    print("Retrieval completed. Results saved to run_file.")

    # 5. Evaluate the retrieval results using 'trec_eval'
    # qrels_file = qrels_file  # Path to your qrels file
    # run_file = run_file  # Path to your retrieval results file

    # Run trec_eval with various metrics (e.g., NDCG, MAP, etc.)
    command = [
    'python', '-m', 'pyserini.eval.trec_eval', 
    '-c', 
    '-m', 'ndcg_cut.10', 
    '-m', 'recip_rank', 
    '-m', 'recall.50', 
    '-m', 'map',
    '-m', 'P.5',
    qrels_file, 
    run_file
    ]

    # Execute the evaluation command
    output_file = f"res.{corpus}.dev-subset.{method}.irs-{initial_retrival_size}.cln-{clusters_num}.k-{nieghbors_num}.m-{top_m_clusters}.sim_mat-{sim_mat_val}.met-{metric_val}.colp-{colapse_type}.l-{lambda_}.emb-{embbeders_names}.mq-{more_queries_val}.win-250.txt"

    with open(output_file, "w") as f:
        subprocess.run(command, stdout=f, stderr=f, text=True)
    # print(general_clustering_model.docs[0])

if __name__ == '__main__':
    # run_regular_retrieval_process(k=50, corpus='msmarco-doc', more_queries=True)
    # nltk.download('punkt_tab')
    # run_experimental_cluster_based_retrirval(corpus = 'msmarco-doc', more_queries=True)


    emb_names = [ ['e5'], ['tf-idf', 'e5']] #['tf-idf'], ['e5'], ['tf-idf', 'e5']]
    methods = ['bag', 'set']
    neighbors = [2, 5, 10]
    
    for emb_name in emb_names:
        for method in methods:
            for neighbor in neighbors:
                run_final_cluster_based_retreival(initial_retrival_size = 250, embbeders_names = emb_name, lambda_=0.9, nieghbors_num = neighbor, method=method, corpus='msmarco-doc', more_queries=True)
                print(f"Finished {emb_name} {method} {neighbor}")


    # with open("topics.dl20.txt") as f:
    #     new_lines = []
    #     for i, line in enumerate(f):
    #         if i < 50:
    #             new_lines.append(line)
    #         else:
    #             break
        
    #     with open("topics.dl20.subset.txt", "w") as out_f:
    #         out_f.writelines(new_lines)

    # with open("qrels.dl20-doc.txt") as f_qr:
    #     queries_set = set()
    #     with open("topics.dl20.subset.txt") as f_t:
    #         for line in f_t:
    #                 qid = line.strip().split('\t')[0]
    #                 queries_set.add(qid)

    #     print(queries_set)
    #     new_lines = []
    #     for line in f_qr:
    #         qid = line.strip().split(' ')[0]
    #         if qid in queries_set:
    #             new_lines.append(line)

    #     with open("qrels.dl20-doc.subset.txt", "w") as out_f:
    #         out_f.writelines(new_lines)
            

    # qrels_file = 'qrels.dl20-doc.subset.txt'

    # run_file = "run.msmarco-doc.dev-subset.set.irs-1000.cln-50.k-10.m-10.sim_mat-T.met-euc.colp-min.l-0.9.emb-['tf-idf'].mq-T.txt"

    # command = [
    # 'python', '-m', 'pyserini.eval.trec_eval', 
    # '-c', 
    # '-m', 'ndcg_cut.10', 
    # '-m', 'recip_rank', 
    # '-m', 'recall.50', 
    # '-m', 'map',
    # '-m', 'P.5',
    # qrels_file, 
    # run_file
    # ]

    # # Execute the evaluation command
    # output_file = f"res.msmarco-doc.dev-subset.set.irs-1000.cln-50.k-10.m-10.sim_mat-T.met-euc.colp-min.l-0.9.emb-['tf-idf'].mq-T.txt"

    # with open(output_file, "w") as f:
    #     subprocess.run(command, stdout=f, stderr=f, text=True)

            
            



    # run_final_cluster_based_retreival(initial_retrival_size = 1000, embbeders_names = ['tf-idf'], lambda_=0.9, method='bag', nieghbors_num = 5, corpus='msmarco-passage', more_queries=True) # run for (tf-idf, e5) embeddings both and separate
    # import os
    # import multiprocessing

    # with open('run.msmarco-doc.dev.txt', 'r') as f:
    #     lines = f.readlines()
    #     with open('run.msmarco-doc.dev-sample.txt', 'w') as out_f:
    #         for i, line in enumerate(lines):
    #             if i == 400:
    #                 break
    #             out_f.write(line)

    # # Get the number of CPU cores
    # cpu_cores = os.cpu_count()  # Includes logical processors (hyper-threading)
    # physical_cores = multiprocessing.cpu_count()  # Only physical cores on most systems

    # print(f"Total CPU threads available: {cpu_cores}")
    # print(f"Physical CPU cores available: {physical_cores}")


    # import torch
    # print(f"CUDA version: {torch.version.cuda}")
    # print(f"Is CUDA available: {torch.cuda.is_available()}")
    # print(f"CUDA device count: {torch.cuda.device_count()}")
    # print(f"CUDA device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

    # Read and save the first 50 lines from 'run.msmarco-passage.dev-subset.txt'
    # with open('run.msmarco-passage.dev-subset.txt', 'r') as f:
    #     lines = f.readlines()
    #     with open('subset_400_lines.txt', 'w') as out_f:
    #         for i, line in enumerate(lines):
    #             if i == 400:
    #                 break
    #             out_f.write(line)

    # # Read and save the first 50 lines from 'run.msmarco-passage.dev-subset.bag.irs-50.cln-50.k-5.m-10.sim_mat-T.met-euc.colp-min.l-0.7.txt'
    # with open("run.msmarco-passage.dev-subset.set.irs-50.cln-50.k-5.m-10.sim_mat-T.met-euc.colp-min.l-0.7.emb-['tf-idf'].txt", 'r') as f:
    #     lines = f.readlines()
    #     with open('set_subset_400_lines.txt', 'w') as out_f:
    #         for i, line in enumerate(lines):
    #             if i == 400:
    #                 break
    #             out_f.write(line)

