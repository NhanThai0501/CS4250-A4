from pymongo import MongoClient
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Step 1: MongoDB setup
def setup_mongo():
    client = MongoClient('mongodb://localhost:27017/')
    db = client['search_engine']
    db.terms.drop()  # Clear terms collection
    db.documents.drop()  # Clear documents collection
    return db

# Step 2: Preprocessing function
def preprocess(doc):
    """Lowercase, remove punctuation, and tokenize."""
    doc = re.sub(r'[^\w\s]', '', doc.lower())
    tokens = doc.split()
    return tokens

# Step 3: Generate n-grams
def generate_ngrams(tokens, n):
    """Generate n-grams from tokens."""
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

# Step 4: Insert documents into the database
def insert_documents(db, documents):
    doc_collection = db.documents
    doc_data = [{"_id": i+1, "content": doc} for i, doc in enumerate(documents)]
    doc_collection.insert_many(doc_data)

# Step 5: Build the inverted index
def build_inverted_index(db, documents):
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
                vocabulary[term] = len(vocabulary) + 1
            term_id = vocabulary[term]

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
    return vocabulary

# Step 6: Compute and update TF-IDF values
def compute_tfidf(db, documents, vocabulary):
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), lowercase=True, stop_words=None)
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    term_collection = db.terms

    for term, index in zip(feature_names, range(tfidf_matrix.shape[1])):
        tfidf_scores = tfidf_matrix[:, index].toarray().flatten()
        term_entry = term_collection.find_one({"term": term})
        if term_entry:
            for doc in term_entry["docs"]:
                doc["tfidf"] = float(tfidf_scores[doc["doc_id"] - 1])
            term_collection.replace_one({"_id": term_entry["_id"]}, term_entry)

# Step 7: Process queries and rank documents
def process_queries(db, queries, vocabulary, documents):
    term_collection = db.terms
    results = []

    def cosine_similarity(query_vector, doc_vector):
        dot_product = np.dot(query_vector, doc_vector)
        magnitude = np.linalg.norm(query_vector) * np.linalg.norm(doc_vector)
        return dot_product / magnitude if magnitude != 0 else 0.0

    for query in queries:
        query_tokens = preprocess(query)
        query_vector = np.zeros(len(vocabulary))
        for token in query_tokens:
            if token in vocabulary:
                query_vector[vocabulary[token] - 1] = 1

        document_scores = []
        for doc_id, content in enumerate(documents, start=1):
            doc_vector = np.zeros(len(vocabulary))
            for term_entry in term_collection.find():
                for doc in term_entry["docs"]:
                    if doc["doc_id"] == doc_id:
                        doc_vector[term_entry["pos"] - 1] = doc["tfidf"]

            score = cosine_similarity(query_vector, doc_vector)
            if score > 0:
                document_scores.append((content, score))

        document_scores = sorted(document_scores, key=lambda x: x[1], reverse=True)
        results.append((query, document_scores))

    return results

# Step 8: Save results to a file
def save_results(output_path, results):
    output = []
    for query, docs in results:
        output.append(f"Query: {query}")
        for doc, score in docs:
            output.append(f"\"{doc}\", {score:.2f}")
        output.append("")
    
    with open(output_path, "w") as file:
        file.write("\n".join(output))

# Main function to orchestrate all steps
def main():
    # Define input documents
    documents = [
        "After the medication, headache and nausea were reported by the patient.",
        "The patient reported nausea and dizziness caused by the medication.",
        "Headache and dizziness are common effects of this medication.",
        "The medication caused a headache and nausea, but no dizziness was reported."
    ]

    # Define queries
    queries = [
        "nausea and dizziness",
        "effects",
        "nausea was reported",
        "dizziness",
        "the medication"
    ]

    # MongoDB setup
    db = setup_mongo()

    # Insert documents into the database
    insert_documents(db, documents)

    # Build inverted index
    vocabulary = build_inverted_index(db, documents)

    # Compute TF-IDF and update database
    compute_tfidf(db, documents, vocabulary)

    # Process queries and rank documents
    results = process_queries(db, queries, vocabulary, documents)

    # Save results to output file
    output_path = "output.txt"
    save_results(output_path, results)
    
    # Print output file path
    print(f"Results saved to: {output_path}")

# Run the main function
if __name__ == "__main__":
    main()
