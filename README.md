# rag-base

This repository includes a base framework for building **Retrieval-Augmented Generation (RAG)** systems.

Originally for personal reasearch purpose, but you can use this as a basis for building your own RAG System.

This repository mainly covers the simple 'Retrieve-Read' paradigm.

**Work in Progress**

## Details

### Retriever
The Retriever module is a refined implementation based on [Self-RAG](https://github.com/AkariAsai/self-rag), which in turn builds upon [Contriever](https://github.com/facebookresearch/contriever).

Major change includes the usage of search_document() function. Our revision includes parallel query search, which can greatly reduce the time consumption of the retrieval module.

