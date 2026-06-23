"""
src/data_loader.py
------------------
Shared data-loading utilities for the Week 2 Information Retrieval notebooks.

All three notebooks (TF-IDF vs BM25, LMIR, VSM) operate on the same SciFact
dataset. This module loads that dataset once and cleanly, so each notebook can
simply import and call these functions instead of repeating the loading logic.
"""

import json
import csv
import os
import re


# ── Default file paths (relative to repo root) ─────────────────────────────
_CORPUS_PATH      = "scifact/corpus.jsonl"
_QUERIES_PATH     = "scifact/queries.jsonl"
_QRELS_TRAIN_PATH = "scifact/qrels/train.tsv"
_QRELS_TEST_PATH  = "scifact/qrels/test.tsv"


def load_corpus(path: str = _CORPUS_PATH) -> dict:
    """
    Load the SciFact corpus.

    Returns
    -------
    dict
        { doc_id (str) : doc_dict } where doc_dict has keys
        '_id', 'title', 'text', and optionally 'metadata'.
    """
    corpus = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc["_id"]] = doc
    return corpus


def load_queries(path: str = _QUERIES_PATH) -> dict:
    """
    Load the SciFact queries.

    Returns
    -------
    dict
        { query_id (str) : query_text (str) }
    """
    queries = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q["text"]
    return queries


def load_qrels(file_path: str) -> dict:
    """
    Load a single qrels TSV file (train or test).

    Returns
    -------
    dict
        { query_id (str) : { doc_id (str) : relevance_score (int) } }
    """
    qrels = {}
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found.")
        return qrels

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip header row: query-id, corpus-id, score
        for row in reader:
            query_id, corpus_id, score = row
            if query_id not in qrels:
                qrels[query_id] = {}
            qrels[query_id][corpus_id] = int(score)
    return qrels


def load_all_qrels(
    train_path: str = _QRELS_TRAIN_PATH,
    test_path:  str = _QRELS_TEST_PATH,
) -> tuple:
    """
    Load and merge train + test qrels into a single dictionary.

    Returns
    -------
    tuple
        (qrels_train, qrels_test, qrels_all)
        All three are dicts with the same structure as load_qrels().
    """
    qrels_train = load_qrels(train_path)
    qrels_test  = load_qrels(test_path)
    qrels_all   = {**qrels_train, **qrels_test}
    return qrels_train, qrels_test, qrels_all


def load_dataset(
    corpus_path:      str = _CORPUS_PATH,
    queries_path:     str = _QUERIES_PATH,
    qrels_train_path: str = _QRELS_TRAIN_PATH,
    qrels_test_path:  str = _QRELS_TEST_PATH,
    verbose:          bool = True,
) -> tuple:
    """
    Convenience function: load corpus, queries, and all qrels in one call.

    Returns
    -------
    tuple
        (corpus, queries, qrels_train, qrels_test, qrels_all)
    """
    corpus = load_corpus(corpus_path)
    queries = load_queries(queries_path)
    qrels_train, qrels_test, qrels_all = load_all_qrels(qrels_train_path, qrels_test_path)

    if verbose:
        print("--- Dataset Statistics ---")
        print(f"Total documents in corpus       : {len(corpus)}")
        print(f"Total queries                   : {len(queries)}")
        print(f"Queries with TRAIN answer keys  : {len(qrels_train)}")
        print(f"Queries with TEST  answer keys  : {len(qrels_test)}")
        print(f"Total queries with answer keys  : {len(qrels_all)}")

    return corpus, queries, qrels_train, qrels_test, qrels_all


def preprocess_text(text: str) -> list:
    """
    Lowercase, strip punctuation, and split text into tokens.

    Note: stopwords are intentionally kept because BM25's IDF component
    naturally down-weights common words, and in scientific text even
    function words can carry meaning.

    Parameters
    ----------
    text : str

    Returns
    -------
    list of str
    """
    if not text:
        return []
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()


def tokenize_dataset(corpus: dict, queries: dict) -> tuple:
    """
    Tokenize all corpus documents and all queries.

    Returns
    -------
    tuple
        (tokenized_corpus, tokenized_queries)
        Both are dicts mapping id -> list of tokens.
    """
    tokenized_corpus = {
        doc_id: preprocess_text(doc.get("title", "") + " " + doc.get("text", ""))
        for doc_id, doc in corpus.items()
    }
    tokenized_queries = {
        query_id: preprocess_text(text)
        for query_id, text in queries.items()
    }
    return tokenized_corpus, tokenized_queries
