import hashlib
from bitarray import bitarray
from abc import ABC, abstractmethod

from document import Document
import re
import numpy as np
from collections import defaultdict
from document import Document
from porter import stem_term
from math import log
import porter
class RetrievalModel(ABC):
    @abstractmethod
    def document_to_representation(
        self, document: Document, stopword_filtering=False, stemming=False
    ):
        """
        Converts a document into its model-specific representation.
        This is an abstract method and not meant to be edited. Implement it in the subclasses!
        :param document: Document object to be represented
        :param stopword_filtering: Controls, whether the document should first be freed of stopwords
        :param stemming: Controls, whether stemming is used on the document's terms
        :return: A representation of the document. Data type and content depend on the implemented model.
        """
        raise NotImplementedError()

    @abstractmethod
    def query_to_representation(self, query: str):
        """
        Determines the representation of a query according to the model's concept.
        :param query: Search query of the user
        :return: Query representation in whatever data type or format is required by the model.
        """
        raise NotImplementedError()

    @abstractmethod
    def match(self, document_representation, query_representation) -> float:
        """
        Matches the query and document presentation according to the model's concept.
        :param document_representation: Data that describes one document
        :param query_representation:  Data that describes a query
        :return: Numerical approximation of the similarity between the query and document representation. Higher is
        "more relevant", lower is "less relevant".
        """
        raise NotImplementedError()


class LinearBooleanModel(RetrievalModel):
    # TODO: Implement all abstract methods and __init__() in this class. (PR02)
    def __init__(self):
        self.documents = []

    def document_to_representation(
        self, document: Document, stopword_filtering=False, stemming=False
    ):
        """
        Converts a document into a list of terms (words).
        :param document: Document object to be represented
        :param stopword_filtering: Controls, whether the document should first be freed of stopwords
        :param stemming: Controls, whether stemming is used on the document's terms
        :return: A list of terms representing the document
        """
        terms = document.terms
        terms = [term.lower() for term in terms]  # Convert all terms to lowercase

        if stopword_filtering:
            terms = [term for term in terms if term not in document.filtered_terms]
        return terms

    def query_to_representation(self, query: str):
        """
        Converts a query into a list of terms (words).
        :param query: Search query of the user
        :return: A list of terms representing the query
        """
        return query.lower().split()

    def match(self, document_representation, query_representation) -> float:
        """
        Matches the query and document presentation based on Boolean search.
        :param document_representation: List of terms that describes one document
        :param query_representation: List of terms that describes a query
        :return: 1.0 if the query term is in the document, 0.0 otherwise
        """
        for query_term in query_representation:
            if query_term in document_representation:
                return 1.0
        return 0.0

    def __str__(self):
        return "Boolean Model (Linear)"


class InvertedListBooleanModel(RetrievalModel):
    def __init__(self):
        self.inverted_index = {}
        self.docs = []
        self.is_ready = False

    def document_to_representation(self, document: Document, stopword_filtering=False, stemming=False):
        terms = set()

        if stopword_filtering:
            terms.update(term.lower() for term in document.filtered_terms)
        else:
            terms.update(term.lower() for term in document.raw_text.split() if term)

        if stemming:
            terms = {porter.stem_term(term) for term in terms}

        return terms

    def query_to_representation(self, query: str, stemming=False):
    # Split the query into terms and operators
        terms = re.split(r'(\W)', query.lower())

        # Remove empty strings and whitespace-only strings
        terms = [term for term in terms if term.strip()]
        
        if stemming:
            terms = [porter.stem_term(term) if term.isalnum() else term for term in terms]

        return terms

    def match(self, document_representation, query_representation):
        return all(term in document_representation for term in query_representation)

    def build_inverted_list(self, documents, stopword_filtering=False, stemming=False):
        self.docs = documents
        self.inverted_index = {}
        for doc_id, document in enumerate(documents):
            terms = self.document_to_representation(document, stopword_filtering, stemming)
            for term in terms:
                if term not in self.inverted_index:
                    self.inverted_index[term] = set()
                self.inverted_index[term].add(doc_id)
        self.is_ready = True

    def __str__(self):
        return 'Boolean Model (Inverted Index)'

class SignatureBasedBooleanModel(RetrievalModel):
    def __init__(self, F=64, D=4):

        self.F = F
        self.D = D
        self.documents = []

    def _hash_function(self, term: str) -> int:
        """
        Hash function to convert a term into an integer.
        """
        return int(hashlib.md5(term.encode('utf-8')).hexdigest(), 16)

    def _create_signature(self, terms):
        """
        Creates a signature bitarray for the given terms.
        """
        signature = bitarray(self.F)
        signature.setall(0)

        for term in terms:
            hash_value = self._hash_function(term)
            for i in range(self.D):
                pos = (hash_value + i) % self.F
                signature[pos] = 1
        return signature

    def document_to_representation(
        self, document: Document, stopword_filtering=False, stemming=False
    ):
        terms = document.terms
        terms = [term.lower() for term in terms]  # Convert all terms to lowercase

        if stopword_filtering:
            terms = [term for term in terms if term not in document.filtered_terms]

        signature = self._create_signature(terms)
        self.documents.append((document, signature))
        return signature

    def query_to_representation(self, query: str):
        terms = query.lower().split()
        signature = self._create_signature(terms)
        return signature

    def match(self, document_representation, query_representation) -> float:
        """
        Matches the query and document presentation based on signature.
        Returns 1.0 if the query signature is a subset of the document signature, 0.0 otherwise.
        """
        if (document_representation & query_representation) == query_representation:
            return 1.0
        return 0.0

    def search(self, query: str, mode='and') -> list:
        """
        Search for documents matching the query.
        Supports 'and' and 'or' modes.
        """
        query_representation = self.query_to_representation(query)
        results = []

        if mode == 'and':
            for document, doc_representation in self.documents:
                if self.match(doc_representation, query_representation):
                    results.append(document)
        elif mode == 'or':
            for document, doc_representation in self.documents:
                if (doc_representation & query_representation).count() > 0:
                    results.append(document)

        return results

    def __str__(self):
        return "Boolean Model (Signatures)"


class VectorSpaceModel(RetrievalModel):
    def __init__(self):
        self.inverted_index = defaultdict(list)
        self.document_vectors = {}
        self.document_lengths = {}
        self.doc_count = 0
        self.collection = []  # To keep track of the document collection

    def document_to_representation(self, document: Document, stopword_filtering=False, stemming=False):
        terms = document.terms
        terms = [term.lower() for term in terms]

        if stopword_filtering:
            terms = [term for term in terms if term not in document.filtered_terms]

        if stemming:
            terms = [stem_term(term) for term in terms]

        term_freq = defaultdict(int)
        for term in terms:
            term_freq[term] += 1

        self.doc_count += 1
        doc_id = self.doc_count
        self.collection.append(document)

        for term, freq in term_freq.items():
            self.inverted_index[term].append((doc_id, freq))

        self.document_vectors[doc_id] = term_freq
        self.document_lengths[doc_id] = np.sqrt(sum(freq**2 for freq in term_freq.values()))
        return term_freq

    def query_to_representation(self, query: str):
        terms = query.lower().split()
        term_freq = defaultdict(int)
        for term in terms:
            term_freq[term] += 1
        return term_freq

    def match(self, document_representation, query_representation) -> float:
        raise NotImplementedError("Match is not used directly in this model.")

    def _compute_tf_idf(self, term_freq, term, N):
        tf = term_freq[term]
        df = len(self.inverted_index[term])
        idf = log(N / df) if df > 0 else 0
        return tf * idf

    def buckley_lewit_search(self, query: str, stemming, stop_word_filtering , output_k):
        query_terms = query.lower().split()

        if stop_word_filtering:
            query_terms = [term for term in query_terms if term not in self.stop_word_list]

        if stemming:
            query_terms = [stem_term(term) for term in query_terms]

        query_vector = self.query_to_representation(query)
        query_weights = {}
        N = self.doc_count

        for term in query_terms:
            if term in self.inverted_index:
                query_weights[term] = self._compute_tf_idf(query_vector, term, N)

        scores = defaultdict(float)
        for term, weight in query_weights.items():
            for doc_id, tf in self.inverted_index[term]:
                if term in self.document_vectors[doc_id]:
                    scores[doc_id] += weight * self._compute_tf_idf(self.document_vectors[doc_id], term, N)

        for doc_id in scores:
            if doc_id in self.document_lengths:
                scores[doc_id] /= self.document_lengths[doc_id]

        ranked_documents = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        results = [(score, self.collection[doc_id - 1]) for doc_id, score in ranked_documents[:output_k]]
        return results

    def __str__(self):
        return "Vector Space Model"


class FuzzySetModel(RetrievalModel):
    # TODO: Implement all abstract methods. (PR04)
    def __init__(self):
        raise NotImplementedError()  # TODO: Remove this line and implement the function.

    def __str__(self):
        return "Fuzzy Set Model"
