# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import json
import pickle
import time
import glob
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

import src.index
import src.data
import src.normalize_text

os.environ["TOKENIZERS_PARALLELISM"] = "true"

class Retriever:
    def __init__(self, args, model=None, tokenizer=None) :
        self.args = args
        self.model = model
        self.tokenizer = tokenizer

    def embed_queries(self, args, queries):
        embeddings, batch_question = [], []
        with torch.no_grad():
            for k, q in enumerate(queries):
                if args.lowercase:
                    q = q.lower()
                if args.normalize_text:
                    q = src.normalize_text.normalize(q)
                batch_question.append(q)

                if len(batch_question) == args.per_gpu_batch_size or k == len(queries) - 1:

                    encoded_batch = self.tokenizer.batch_encode_plus(
                        batch_question,
                        return_tensors="pt",
                        max_length=args.question_maxlength,
                        padding=True,
                        truncation=True,
                    )
                    encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
                    output = self.model(**encoded_batch)

                    last_hidden = output["last_hidden_state"]
                    attention_mask = encoded_batch['attention_mask']
                    last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)

                    emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
                    embeddings.append(emb.cpu())

        embeddings = torch.cat(embeddings, dim=0)
        print(f"Questions embeddings shape: {embeddings.size()}")

        return embeddings.numpy()
    
    def index_encoded_data(self, index, embedding_files, indexing_batch_size):
        allids = []
        allembeddings = np.array([])
        for i, file_path in enumerate(embedding_files):
            print(f"Loading file {file_path}")
            with open(file_path, "rb") as fin:
                ids, embeddings = pickle.load(fin)

            allembeddings = np.vstack((allembeddings, embeddings)) if allembeddings.size else embeddings
            allids.extend(ids)
            while allembeddings.shape[0] > indexing_batch_size:
                allembeddings, allids = self.add_embeddings(index, allembeddings, allids, indexing_batch_size)

        while allembeddings.shape[0] > 0:
            allembeddings, allids = self.add_embeddings(index, allembeddings, allids, indexing_batch_size)

        print("Data indexing completed.")


    def add_embeddings(self, index, embeddings, ids, indexing_batch_size):
        end_idx = min(indexing_batch_size, embeddings.shape[0])
        ids_toadd = ids[:end_idx]
        embeddings_toadd = embeddings[:end_idx]
        ids = ids[end_idx:]
        embeddings = embeddings[end_idx:]
        index.index_data(ids_toadd, embeddings_toadd)
        return embeddings, ids

    def add_passages(self, passages, top_passages_and_scores):
        docs = [passages[doc_id] for doc_id in top_passages_and_scores[0][0]]
        return docs

    def setup_retriever(self):
        print(f"Loading model from: {self.args.retrieval_model_name_or_path}")
        self.model = AutoModel.from_pretrained(self.args.retrieval_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.retrieval_model_name_or_path)
        self.model.eval()
        self.model = self.model.cuda()
        # if not self.args.no_fp16:
        #     self.model = self.model.half()

        self.index = src.index.Indexer(self.args.retrieval_embedding_size, self.args.retrieval_n_subquantizers, self.args.retrieval_n_bits)

        # index all passages
        input_paths = glob.glob(self.args.passages_embeddings)
        input_paths = sorted(input_paths)
        embeddings_dir = os.path.dirname(input_paths[0])
        index_path = os.path.join(embeddings_dir, "index.faiss")
        if self.args.save_or_load_index and os.path.exists(index_path):
            self.index.deserialize_from(embeddings_dir)
        else:
            print(f"Indexing passages from files {input_paths}")
            start_time_indexing = time.time()
            self.index_encoded_data(self.index, input_paths, self.args.indexing_batch_size)
            print(f"Indexing time: {time.time()-start_time_indexing:.1f} s.")
            if self.args.save_or_load_index:
                self.index.serialize(embeddings_dir)

        # load passages
        print("loading passages")
        self.passages = src.data.load_passages(self.args.passages)
        self.passage_id_map = {x["id"]: x for x in self.passages}
        print("passages have been loaded")

    def search_document(self, query, top_n=10):
        questions_embedding = self.embed_queries(self.args, query)

        # get top k results
        start_time_retrieval = time.time()
        top_ids_and_scores = self.index.search_knn(questions_embedding, self.args.max_k)
        print(f"Search time: {time.time()-start_time_retrieval:.1f} s.")

        return self.add_passages(self.passage_id_map, top_ids_and_scores)[:top_n]
    
    def reset_args(self,args):
        self.args = args