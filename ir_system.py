# --------------------------------------------------------------------------------
# Information Retrieval SS2024 - Practical Assignment Template
# --------------------------------------------------------------------------------
# This Python template is provided as a starting point for your assignments PR02-04.
# It serves as a base for a very rudimentary text-based information retrieval system.
#
# Please keep all instructions from the task description in mind.
# Especially, avoid changing the base structure, function or class names or the
# underlying program logic. This is necessary to run automated tests on your code.
#
# Instructions:
# 1. Read through the whole template to understand the expected workflow and outputs.
# 2. Implement the required functions and classes, filling in your code where indicated.
# 3. Test your code to ensure functionality and correct handling of edge cases.
#
# Good luck!


import json
import os

import cleanup
import extraction
import models
import porter
from document import Document
import re

import time

# Important paths:
RAW_DATA_PATH = "raw_data"
DATA_PATH = "data"
COLLECTION_PATH = os.path.join(DATA_PATH, "my_collection.json")
STOPWORD_FILE_PATH = os.path.join(DATA_PATH, "stopwords.json")

# Menu choices:
(
    CHOICE_LIST,
    CHOICE_SEARCH,
    CHOICE_EXTRACT,
    CHOICE_UPDATE_STOP_WORDS,
    CHOICE_SET_MODEL,
    CHOICE_SHOW_DOCUMENT,
    CHOICE_EXIT,
) = (1, 2, 3, 4, 5, 6, 9)
MODEL_BOOL_LIN, MODEL_BOOL_INV, MODEL_BOOL_SIG, MODEL_FUZZY, MODEL_VECTOR = (
    1,
    2,
    3,
    4,
    5,
)
SW_METHOD_LIST, SW_METHOD_CROUCH = 1, 2


class InformationRetrievalSystem(object):
    def __init__(self):
        if not os.path.isdir(DATA_PATH):
            os.makedirs(DATA_PATH)

        # Collection of documents, initially empty.
        try:
            self.collection = extraction.load_collection_from_json(COLLECTION_PATH)
        except FileNotFoundError:
            print("No previous collection was found. Creating empty one.")
            self.collection = []

        # Stopword list, initially empty.
        try:
            with open(STOPWORD_FILE_PATH, "r") as f:
                self.stop_word_list = json.load(f)
        except FileNotFoundError:
            print("No stopword list was found.")
            self.stop_word_list = []

        self.model = None  # Saves the current IR model in use.
        self.output_k = 5  # Controls how many results should be shown for a query.

    def main_menu(self):
        """
        Provides the main loop of the CLI menu that the user interacts with.
        """
        while True:
            print(f"Current retrieval model: {self.model}")
            print(f"Current collection: {len(self.collection)} documents")
            print()
            print("Please choose an option:")
            print(f"{CHOICE_LIST} - List documents")
            print(f"{CHOICE_SEARCH} - Search for term")
            print(f"{CHOICE_EXTRACT} - Build collection")
            print(f"{CHOICE_UPDATE_STOP_WORDS} - Rebuild stopword list")
            print(f"{CHOICE_SET_MODEL} - Set model")
            print(f"{CHOICE_SHOW_DOCUMENT} - Show a specific document")
            print(f"{CHOICE_EXIT} - Exit")
            action_choice = int(input("Enter choice: "))

            if action_choice == CHOICE_LIST:
                # List documents in CLI.
                if self.collection:
                    for document in self.collection:
                        print(document)
                else:
                    print("No documents.")
                print()

            elif action_choice == CHOICE_SEARCH:
                # Read a query string from the CLI and search for it.

                # Determine desired search parameters:
                SEARCH_NORMAL, SEARCH_SW, SEARCH_STEM, SEARCH_SW_STEM = 1, 2, 3, 4
                print("Search options:")
                print(f"{SEARCH_NORMAL} - Standard search (default)")
                print(f"{SEARCH_SW} - Search documents with removed stopwords")
                print(f"{SEARCH_STEM} - Search documents with stemmed terms")
                print(
                    f"{SEARCH_SW_STEM} - Search documents with removed stopwords AND stemmed terms"
                )
                search_mode = int(input("Enter choice: "))
                stop_word_filtering = (search_mode == SEARCH_SW) or (
                    search_mode == SEARCH_SW_STEM
                )
                stemming = (search_mode == SEARCH_STEM) or (
                    search_mode == SEARCH_SW_STEM
                )

                # Actual query processing begins here:
                query = input("Query: ")
                if stemming:
                    query = porter.stem_query_terms(query)
                start_time = time.time()  # Start measuring time

                if isinstance(self.model, models.InvertedListBooleanModel):
                    results = self.inverted_list_search(
                        query, stemming, stop_word_filtering
                    )
                elif isinstance(self.model, models.VectorSpaceModel):
                    results = self.buckley_lewit_search(
                        query, stemming, stop_word_filtering
                    )
                elif isinstance(self.model, models.SignatureBasedBooleanModel):
                    results = self.signature_search(
                        query, stemming, stop_word_filtering
                    )
                else:
                    results = self.basic_query_search(
                        query, stemming, stop_word_filtering
                    )
                end_time = time.time()  # End measuring time

                # Output of results:
                for score, document in results:
                    print(f"{score}: {document}")

                # Output of quality metrics:
                print()
                print(f'precision: {self.calculate_precision(results)}')
                print(f'recall: {self.calculate_recall(results)}')

                processing_time = (end_time - start_time) * 1000  # Convert to milliseconds
                print(f'Query processing time: {processing_time:.2f} ms')
            elif action_choice == CHOICE_EXTRACT:
                # Extract document collection from text file.

                raw_collection_file = os.path.join(RAW_DATA_PATH, "aesopa10.txt")
                self.collection = extraction.extract_collection(raw_collection_file)
                assert isinstance(self.collection, list)
                assert all(isinstance(d, Document) for d in self.collection)

                if input("Should stopwords be filtered? [y/N]: ") == "y":
                    cleanup.filter_collection(self.collection)

                if input("Should stemming be performed? [y/N]: ") == "y":
                    porter.stem_all_documents(self.collection)

                extraction.save_collection_as_json(self.collection, COLLECTION_PATH)
                print("Done.\n")

            elif action_choice == CHOICE_UPDATE_STOP_WORDS:
                # Rebuild the stop word list, using one out of two methods.

                print("Available options:")
                print(f"{SW_METHOD_LIST} - Load stopword list from file")
                print(
                    f"{SW_METHOD_CROUCH} - Generate stopword list using Crouch's method"
                )

                method_choice = int(input("Enter choice: "))
                if method_choice in (SW_METHOD_LIST, SW_METHOD_CROUCH):
                    # Load stop words using the desired method:
                    if method_choice == SW_METHOD_LIST:
                        self.stop_word_list = cleanup.load_stop_word_list(
                            os.path.join(RAW_DATA_PATH, "englishST.txt")
                        )
                        print("Done.\n")
                    elif method_choice == SW_METHOD_CROUCH:
                        self.stop_word_list = (
                            cleanup.create_stop_word_list_by_frequency(self.collection)
                        )
                        print("Done.\n")

                    # Save new stopword list into file:
                    with open(STOPWORD_FILE_PATH, "w") as f:
                        json.dump(self.stop_word_list, f)
                else:
                    print("Invalid choice.")

            elif action_choice == CHOICE_SET_MODEL:
                # Choose and set the retrieval model to use for searches.

                print()
                print("Available models:")
                print(f"{MODEL_BOOL_LIN} - Boolean model with linear search")
                print(f"{MODEL_BOOL_INV} - Boolean model with inverted lists")
                print(f"{MODEL_BOOL_SIG} - Boolean model with signature-based search")
                print(f"{MODEL_FUZZY} - Fuzzy set model")
                print(f"{MODEL_VECTOR} - Vector space model")
                model_choice = int(input("Enter choice: "))
                if model_choice == MODEL_BOOL_LIN:
                    self.model = models.LinearBooleanModel()
                elif model_choice == MODEL_BOOL_INV:
                    self.model = models.InvertedListBooleanModel()
                elif model_choice == MODEL_BOOL_SIG:
                    self.model = models.SignatureBasedBooleanModel()
                elif model_choice == MODEL_FUZZY:
                    self.model = models.FuzzySetModel()
                elif model_choice == MODEL_VECTOR:
                    self.model = models.VectorSpaceModel()
                else:
                    print("Invalid choice.")

            elif action_choice == CHOICE_SHOW_DOCUMENT:
                target_id = int(input("ID of the desired document:"))
                found = False
                for document in self.collection:
                    if document.document_id == target_id:
                        print(document.title)
                        print("-" * len(document.title))
                        print(document.raw_text)
                        found = True

                if not found:
                    print(f"Document #{target_id} not found!")

            elif action_choice == CHOICE_EXIT:
                break
            else:
                print("Invalid choice.")

            print()
            input("Press ENTER to continue...")
            print()

    def basic_query_search(
        self, query: str, stemming: bool, stop_word_filtering: bool
    ) -> list:
        """
        Searches the collection for a query string. This method is "basic" in that it does not use any special algorithm
        to accelerate the search. It simply calculates all representations and matches them, returning a sorted list of
        the k most relevant documents and their scores.
        :param query: Query string
        :param stemming: Controls, whether stemming is used
        :param stop_word_filtering: Controls, whether stop-words are ignored in the search
        :return: List of tuples, where the first element is the relevance score and the second the corresponding
        document
        """
        query_representation = self.model.query_to_representation(query)
        document_representations = [
            self.model.document_to_representation(d, stop_word_filtering, stemming)
            for d in self.collection
        ]
        scores = [
            self.model.match(dr, query_representation)
            for dr in document_representations
        ]
        ranked_collection = sorted(
            zip(scores, self.collection), key=lambda x: x[0], reverse=True
        )
        results = ranked_collection[: self.output_k]
        return results

    def inverted_list_search(
        self, query: str, stemming: bool, stop_word_filtering: bool
    ) -> list:
        """
        Fast Boolean query search for inverted lists.
        :param query: Query string
        :param stemming: Controls, whether stemming is used
        :param stop_word_filtering: Controls, whether stop-words are ignored in the search
        :return: List of tuples, where the first element is the relevance score and the second the corresponding
        document
        """
        if not hasattr(self, 'inverted_index'):
            self.build_inverted_index(stemming, stop_word_filtering)
        
        # Parse and process the query
        results = self.process_boolean_query(query)
        
        # Retrieve the actual documents
        result_documents = [(1, self.collection[doc_id - 1]) for doc_id in results]
        
        return result_documents
    
    def build_inverted_index(self, stemming: bool, stop_word_filtering: bool):
        self.inverted_index = {}
        
        for doc_id, document in enumerate(self.collection, start=1):
            terms = self.model.document_to_representation(document , stop_word_filtering, stemming)
            for term in terms:
                if term not in self.inverted_index:
                    self.inverted_index[term] = set()
                self.inverted_index[term].add(doc_id)
    
    def process_boolean_query(self, query: str) -> set:
        terms = self.parse_query(query)
        result_set = None
        
        for term, operator in terms:
            if operator == "&":
                result_set = result_set & self.inverted_index.get(term, set()) if result_set is not None else self.inverted_index.get(term, set())
            elif operator == "|":
                result_set = result_set | self.inverted_index.get(term, set()) if result_set is not None else self.inverted_index.get(term, set())
            elif operator == "-":
                result_set = result_set - self.inverted_index.get(term, set()) if result_set is not None else set(self.collection.keys()) - self.inverted_index.get(term, set())
            else:
                result_set = self.inverted_index.get(term, set()) if result_set is None else result_set
                
        return result_set if result_set is not None else set()

    def parse_query(self, query: str) -> list:
        terms = re.findall(r'(-?\w+)([&|]*)', query)
        parsed_terms = []
        
        for term, operator in terms:
            if term.startswith('-'):
                parsed_terms.append((term[1:], "-"))
            else:
                parsed_terms.append((term, "&" if not operator else operator))
        
        return parsed_terms

    def buckley_lewit_search(
        self, query: str, stemming: bool, stop_word_filtering: bool
    ) -> list:
        """
        Fast query search for the Vector Space Model using the algorithm by Buckley & Lewit.
        :param query: Query string
        :param stemming: Controls, whether stemming is used
        :param stop_word_filtering: Controls, whether stop-words are ignored in the search
        :return: List of tuples, where the first element is the relevance score and the second the corresponding
        document
        """
        # TODO: Implement this function (PR04)
        raise NotImplementedError("To be implemented in PR04")

    def signature_search(
        self, query: str, stemming: bool, stop_word_filtering: bool
    ) -> list:
        """
        Fast Boolean query search using signatures for quicker processing.
        :param query: Query string
        :param stemming: Controls, whether stemming is used
        :param stop_word_filtering: Controls, whether stop-words are ignored in the search
        :return: List of tuples, where the first element is the relevance score and the second the corresponding
        document
        """
        # TODO: Implement this function (PR04)
        raise NotImplementedError("To be implemented in PR04")
    def calculate_precision(self, result_list: list[tuple]) -> float:
        try:
            with open(os.path.join(RAW_DATA_PATH, "ground_truth.txt"), "r") as f:
                ground_truth = f.read().splitlines()

            relevant_docs = set()
            for line in ground_truth:
                if not line.strip() or line.startswith("#"):
                    # Skip empty lines and comments
                    continue

                parts = line.split(' - ')
                if len(parts) == 2:
                    term, doc_ids = parts
                    relevant_docs.update(map(int, doc_ids.split(', ')))
                else:
                    print(f"Skipping malformed line in ground_truth.txt: {line}")

            retrieved_docs = {doc.document_id for _, doc in result_list}
            true_positives = len(relevant_docs.intersection(retrieved_docs))

            return true_positives / len(retrieved_docs) if retrieved_docs else -1

        except FileNotFoundError:
            return -1
        except Exception as e:
            print(f"An error occurred while calculating precision: {e}")
            return -1

    def calculate_recall(self, result_list: list[tuple]) -> float:
        try:
            with open(os.path.join(RAW_DATA_PATH, "ground_truth.txt"), "r") as f:
                ground_truth = f.read().splitlines()

            relevant_docs = set()
            for line in ground_truth:
                if not line.strip() or line.startswith("#"):
                    # Skip empty lines and comments
                    continue

                parts = line.split(' - ')
                if len(parts) == 2:
                    term, doc_ids = parts
                    relevant_docs.update(map(int, doc_ids.split(', ')))
                else:
                    print(f"Skipping malformed line in ground_truth.txt: {line}")

            retrieved_docs = {doc.document_id for _, doc in result_list}
            true_positives = len(relevant_docs.intersection(retrieved_docs))

            return true_positives / len(relevant_docs) if relevant_docs else -1

        except FileNotFoundError:
            return -1
        except Exception as e:
            print(f"An error occurred while calculating recall: {e}")
            return -1
    

if __name__ == "__main__":
    irs = InformationRetrievalSystem()
    irs.main_menu()
    exit(0)
