# Contains functions that deal with the extraction of documents from a text file (see PR01)

import json

from document import Document


def extract_collection(source_file_path: str) -> list[Document]:
    """
    Loads a text file (aesopa10.txt) and extracts each of the listed fables/stories from the file.
    :param source_file_name: File name of the file that contains the fables
    :return: List of Document objects
    """
    catalog = []  # this dictionary will store the document raw_data.\
    document_id = 0  # Unique document ID counter

    with open(source_file_path, "r", encoding="utf-8") as file:
        # Skip lines until reaching line 308
        skipped_lines = 0
        while skipped_lines < 307:
            file.readline()
            skipped_lines += 1

        title = ""  # Temporary variable to store current title
        currentData = ""  # String to get raw text
        isRawtext = False
        emptyLineCounter = 0
        emptyLineBetweenRawData = 0

        for line in file:
            line = line.strip()  # delete white spaces
            if line == "" and not isRawtext:
                emptyLineCounter += 1
                continue
            if emptyLineCounter == 2:
                isRawtext = True
                if line == "":
                    emptyLineBetweenRawData += 1
                else:
                    currentData += line + " "
                    emptyLineBetweenRawData = 0
            if emptyLineBetweenRawData == 3:
                emptyLineCounter = 3

            # Empty Line
            if emptyLineCounter == 3:

                all_terms = currentData.split()
                doc = Document()  # New Document object
                doc.title = title
                doc.document_id = document_id
                doc.terms = all_terms
                doc.raw_text = currentData.strip()
                catalog.append(doc)

                document_id += 1  # Increments doc id
                title = ""
                emptyLineCounter = 0
                currentData = ""
                emptyLineBetweenRawData = 0
                isRawtext = False

                continue

            if not isRawtext and line:  # Line with title
                title = line

        # Manage the last document if the file does not end with enough blank lines.
        if currentData or title:
            all_terms = currentData.split()

            doc = Document()
            doc.title = title
            doc.document_id = document_id
            doc.terms = all_terms
            doc.raw_text = currentData.strip()

            catalog.append(doc)

    return catalog


def save_collection_as_json(collection: list[Document], file_path: str) -> None:
    """
    Saves the collection to a JSON file.
    :param collection: The collection to store (= a list of Document objects)
    :param file_path: Path of the JSON file
    """

    serializable_collection = []
    for document in collection:
        serializable_collection += [
            {
                "document_id": document.document_id,
                "title": document.title,
                "raw_text": document.raw_text,
                "terms": document.terms,
                "filtered_terms": document.filtered_terms,
                "stemmed_terms": document.stemmed_terms,
            }
        ]

    with open(file_path, "w") as json_file:
        json.dump(serializable_collection, json_file)


def load_collection_from_json(file_path: str) -> list[Document]:
    """
    Loads the collection from a JSON file.
    :param file_path: Path of the JSON file
    :return: list of Document objects
    """
    try:
        with open(file_path, "r") as json_file:
            json_collection = json.load(json_file)

        collection = []
        for doc_dict in json_collection:
            document = Document()
            document.document_id = doc_dict.get("document_id")
            document.title = doc_dict.get("title")
            document.raw_text = doc_dict.get("raw_text")
            document.terms = doc_dict.get("terms")
            document.filtered_terms = doc_dict.get("filtered_terms")
            document.stemmed_terms = doc_dict.get("stemmed_terms")
            collection += [document]

        return collection
    except FileNotFoundError:
        print("No collection was found. Creating empty one.")
        return []
