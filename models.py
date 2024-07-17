# Contains all retrieval models.

from abc import ABC, abstractmethod

from document import Document
import re
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
    # TODO: Implement all abstract methods. (PR04)
    def __init__(self):
        raise NotImplementedError()  # TODO: Remove this line and implement the function.

    def __str__(self):
        return "Boolean Model (Signatures)"


class VectorSpaceModel(RetrievalModel):
    # TODO: Implement all abstract methods. (PR04)
    def __init__(self):
        raise NotImplementedError()  # TODO: Remove this line and implement the function.

    def __str__(self):
        return "Vector Space Model"


class FuzzySetModel(RetrievalModel):
    # TODO: Implement all abstract methods. (PR04)
    def __init__(self):
        raise NotImplementedError()  # TODO: Remove this line and implement the function.

    def __str__(self):
        return "Fuzzy Set Model"
