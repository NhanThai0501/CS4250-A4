#-------------------------------------------------------------------------
# AUTHOR: Nhan Thai
# FILENAME: hw4_inverted_index.py
# SPECIFICATION: Build a similar engine in Python that indexes the documents of question 3 of HW4 in MongoDB to enable search, and rank the documents using
# the vector space model
# FOR: CS 4250 - Assignment #4
# TIME SPENT: 4
#-------------------------------------------------------------------------

import re
from pymongo import MongoClient
from collections import defaultdict
import math

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['cs4250-hw4']
terms_collection = db['terms']
documents_collection = db['documents']

# Step 1: Preprocess text
def preprocess(text):
    """
    Preprocess text: remove punctuation, lowercase, and tokenize.
    Generate unigrams, bigrams, and trigrams.
    """
    text = re.sub(r'[^\w\s]', '', text.lower())  # Remove punctuation and lowercase
    tokens = text.split()
    unigrams = tokens
    bigrams = [' '.join(tokens[i:i+2]) for i in range(len(tokens)-1)]
    trigrams = [' '.join(tokens[i:i+3]) for i in range(len(tokens)-2)]
    return unigrams + bigrams + trigrams

# Step 2: Build inverted index with TF-IDF
def calculate_tfidf(documents):
    """
    Calculate TF-IDF for terms in the documents.
    Return an inverted index and a vocabulary dictionary.
    """
    N = len(documents)  # Total number of documents
    vocabulary = {}  # Map terms to unique IDs
    inverted_index = defaultdict(lambda: {"pos": None, "docs": []})
    doc_term_counts = []  # Store term counts for each document

    for doc_id, content in enumerate(documents, start=1):
        terms = preprocess(content)
        term_counts = defaultdict(int)
        for term in terms:
            term_counts[term] += 1
            if term not in vocabulary:
                vocabulary[term] = len(vocabulary) + 1
        doc_term_counts.append((doc_id, term_counts))

    # Calculate TF, DF, and IDF
    df = defaultdict(int)
    for _, term_counts in doc_term_counts:
        for term in term_counts:
            df[term] += 1

    for doc_id, term_counts in doc_term_counts:
        for term, tf in term_counts.items():
            idf = math.log(N / df[term])
            tfidf = tf * idf
            if not inverted_index[term]["pos"]:
                inverted_index[term]["pos"] = vocabulary[term]
            inverted_index[term]["docs"].append({"doc_id": doc_id, "tfidf": tfidf})

    return inverted_index, vocabulary

# Step 3: Insert into MongoDB
def index_documents(documents):
    """
    Store documents and terms in MongoDB.
    """
    # Clear existing collections
    terms_collection.delete_many({})
    documents_collection.delete_many({})

    # Insert documents into MongoDB
    documents_data = [{"_id": i + 1, "content": doc} for i, doc in enumerate(documents)]
    documents_collection.insert_many(documents_data)

    # Build inverted index
    inverted_index, vocabulary = calculate_tfidf(documents)

    # Insert terms into MongoDB
    terms_data = [
        {"_id": vocabulary[term], "pos": vocabulary[term], "docs": data["docs"]}
        for term, data in inverted_index.items()
    ]
    terms_collection.insert_many(terms_data)
    print("Indexing completed.")
    
    return vocabulary

# Step 4: Rank documents using the vector space model
def rank_documents(queries, documents, vocabulary):
    """
    Rank documents for each query using cosine similarity.
    """
    for query in queries:
        print(f"\nQuery: {query}")
        query_terms = preprocess(query)
        query_vector = {}
        query_magnitude = 0

        # Build query vector based on inverted index
        for term in query_terms:
            if term in vocabulary:
                term_data = terms_collection.find_one({"pos": vocabulary[term]})
                if term_data:
                    idf = math.log(len(documents) / len(term_data["docs"]))
                    query_vector[term] = idf
                    query_magnitude += idf ** 2
        query_magnitude = math.sqrt(query_magnitude)

        # Calculate cosine similarity for matching documents
        doc_scores = defaultdict(float)
        for term, q_weight in query_vector.items():
            term_data = terms_collection.find_one({"pos": vocabulary[term]})
            for doc in term_data["docs"]:
                doc_scores[doc["doc_id"]] += q_weight * doc["tfidf"]

        # Normalize scores by document magnitudes
        sorted_results = []
        for doc_id, score in doc_scores.items():
            doc_data = documents_collection.find_one({"_id": doc_id})
            doc_magnitude = math.sqrt(
                sum(
                    tfidf["tfidf"] ** 2
                    for tfidf in term_data["docs"]
                    if tfidf["doc_id"] == doc_id
                )
            )
            normalized_score = score / (query_magnitude * doc_magnitude)
            sorted_results.append((doc_data["content"], normalized_score))

        # Sort results by descending score
        sorted_results = sorted(sorted_results, key=lambda x: -x[1])
        for content, score in sorted_results:
            print(f'"{content}", {score:.4f}')

# Main Execution
if __name__ == "__main__":
    # Define documents and queries
    documents = [
        "After the medication, headache and nausea were reported by the patient.",
        "The patient reported nausea and dizziness caused by the medication.",
        "Headache and dizziness are common effects of this medication.",
        "The medication caused a headache and nausea, but no dizziness was reported."
    ]
    queries = [
        "nausea and dizziness",
        "effects",
        "nausea was reported",
        "dizziness",
        "the medication"
    ]

    # Index documents
    vocabulary = index_documents(documents)

    # Rank documents for each query
    rank_documents(queries, documents, vocabulary)


import re
from pymongo import MongoClient
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# MongoDB connection setup
client = MongoClient('mongodb://localhost:27017/')
db = client['search_engine']

# Clear existing collections
db.terms.drop()
db.documents.drop()

# Step 1: Preprocess text
def preprocess(doc):
    """
    Preprocess text: remove punctuation, lowercase, and tokenize.
    """
    doc = re.sub(r'[^\w\s]', '', doc.lower())  # Remove punctuation and lowercase
    tokens = doc.split()
    return tokens

# Step 2: Generate unigrams, bigrams, and trigrams
def generate_ngrams(tokens, n):
    """
    Generate n-grams from tokens.
    """
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

# Step 3: Insert documents into MongoDB
def index_documents(documents):
    """
    Store documents and build the inverted index in MongoDB.
    Returns the vocabulary and document TF-IDF matrix.
    """
    # Insert documents into MongoDB
    doc_collection = db.documents
    doc_data = [{"_id": i + 1, "content": doc} for i, doc in enumerate(documents)]
    doc_collection.insert_many(doc_data)

    # Build the inverted index
    term_collection = db.terms
    vocabulary = {}

    for doc_id, content in enumerate(documents, start=1):
        tokens = preprocess(content)
        unigrams = tokens
        bigrams = generate_ngrams(tokens, 2)
        trigrams = generate_ngrams(tokens, 3)
        all_terms = unigrams + bigrams + trigrams

        for term in all_terms:
            if term not in vocabulary:
                vocabulary[term] = len(vocabulary) + 1  # Assign a unique position ID
            term_id = vocabulary[term]

            # Check if the term already exists in the database
            term_entry = term_collection.find_one({"_id": term_id})
            if term_entry:
                term_entry["docs"].append({"doc_id": doc_id, "tfidf": 0})  # Placeholder for TF-IDF
                term_collection.replace_one({"_id": term_id}, term_entry)
            else:
                term_collection.insert_one({
                    "_id": term_id,
                    "term": term,
                    "pos": term_id,
                    "docs": [{"doc_id": doc_id, "tfidf": 0}]
                })

    # Compute TF-IDF for terms in documents
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), lowercase=True, stop_words=None)
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Update TF-IDF values in the inverted index
    for term, index in zip(feature_names, range(tfidf_matrix.shape[1])):
        tfidf_scores = tfidf_matrix[:, index].toarray().flatten()
        term_entry = term_collection.find_one({"term": term})
        if term_entry:
            for doc in term_entry["docs"]:
                doc["tfidf"] = float(tfidf_scores[doc["doc_id"] - 1])
            term_collection.replace_one({"_id": term_entry["_id"]}, term_entry)

    return vocabulary, tfidf_matrix

# Step 4: Rank documents using the vector space model
def rank_documents(queries, vocabulary, documents):
    """
    Rank documents for each query using cosine similarity.
    """
    results = []
    for query in queries:
        query_tokens = preprocess(query)
        query_vector = np.zeros(len(vocabulary))
        for token in query_tokens:
            if token in vocabulary:
                query_vector[vocabulary[token] - 1] = 1

        document_scores = []
        for doc_id, content in enumerate(documents, start=1):
            doc_vector = np.zeros(len(vocabulary))
            for term_entry in db.terms.find():
                for doc in term_entry["docs"]:
                    if doc["doc_id"] == doc_id:
                        doc_vector[term_entry["pos"] - 1] = doc["tfidf"]

            # Compute cosine similarity
            score = cosine_similarity(query_vector, doc_vector)
            if score > 0:
                document_scores.append((content, score))

        # Sort documents by score in descending order
        document_scores = sorted(document_scores, key=lambda x: x[1], reverse=True)
        results.append((query, document_scores))

    return results

# Step 5: Compute cosine similarity
def cosine_similarity(query_vector, doc_vector):
    """
    Compute cosine similarity between two vectors.
    """
    dot_product = np.dot(query_vector, doc_vector)
    magnitude = np.linalg.norm(query_vector) * np.linalg.norm(doc_vector)
    if magnitude == 0:
        return 0.0
    return dot_product / magnitude

# Main Execution
if __name__ == "__main__":
    # Define documents and queries
    documents = [
        "After the medication, headache and nausea were reported by the patient.",
        "The patient reported nausea and dizziness caused by the medication.",
        "Headache and dizziness are common effects of this medication.",
        "The medication caused a headache and nausea, but no dizziness was reported."
    ]
    queries = [
        "nausea and dizziness",
        "effects",
        "nausea was reported",
        "dizziness",
        "the medication"
    ]

    # Index documents and build vocabulary
    vocabulary, tfidf_matrix = index_documents(documents)

    # Rank documents for each query
    results = rank_documents(queries, vocabulary, documents)

    # Output results to console and file
    output = []
    for query, docs in results:
        output.append(f"Query: {query}")
        for doc, score in docs:
            output.append(f"\"{doc}\", {score:.2f}")
        output.append("")

    output_path = "output.txt"
    with open(output_path, "w") as file:
        file.write("\n".join(output))

    print(f"Results saved to {output_path}")
