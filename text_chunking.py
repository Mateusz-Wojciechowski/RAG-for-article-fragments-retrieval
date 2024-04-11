import spacy
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

# loading models
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')


def texts_to_vectors(texts):
    return model.encode(texts, convert_to_tensor=False)


def cluster_text(vecs, threshold):
    clustering = AgglomerativeClustering(n_clusters=None, metric='cosine', linkage='average', distance_threshold=threshold)
    labels = clustering.fit_predict(vecs)
    return labels


def preprocess_text(text):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    vectors = texts_to_vectors(sentences)
    return sentences, vectors


def split_text(text, threshold):
    sentences, vecs = preprocess_text(text)
    labels = cluster_text(vecs, 1 - threshold)

    clusters = {i: [] for i in np.unique(labels)}
    for i, label in enumerate(labels):
        clusters[label].append(sentences[i])

    final_texts = [' '.join(clusters[key]) for key in clusters if len(clusters[key]) > 0]
    return final_texts







