import json
import sys
import os
import math
from lxml import etree

# TODO document nltk.download('popular') command in python console
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


class VSM:

    INDEX_PATH = 'vsm_inverted_index.json'
    TOP_DOCS_PATH = 'ranked_query_docs.txt'
    MAX_DOCUMENTS = 100
    RESULTS_THRESHOLD = 10

    def __init__(self):
        self.inverted_index = {}
        self.document_lengths = {}
        self.document_scores = {}
        self.document_number = 0
        self.corpus_directory = ""
        self.curr_tokens = []
        self.curr_tokens_with_tf = {}
        self.top_docs = []
        self.ranking = ""
        self.limit = False

    # Get a string representing document/query
    # Tokenize the string using nltk, and convert to lowercase
    # Remove unwanted words (punctuation, numbers, stopwords) and do stemming using Porter Stemmer
    # TODO write in the documentation that we used stemming
    # Return the list of valid tokens
    def extract_tokens(self, data_str):
        stemmer = PorterStemmer()

        data = word_tokenize(data_str)

        data = [word.lower() for word in data]

        def my_filter(word):
            return word.isalpha() and len(word) > 1 and word not in stopwords.words('english')

        data = [word for word in data if my_filter(word)]
        data = [stemmer.stem(word) for word in data]

        self.curr_tokens = data

    # Compute term frequencies for a list of tokens (document/query) saved in curr_tokens
    # Normalize by maximum count
    def compute_term_frequencies(self):

        max_count = 1

        for term in self.curr_tokens:
            if term in self.curr_tokens_with_tf:
                self.curr_tokens_with_tf[term] += 1
                if self.curr_tokens_with_tf[term] > max_count:
                    max_count = self.curr_tokens_with_tf[term]
            else:
                self.curr_tokens_with_tf[term] = 1

        self.curr_tokens_with_tf = {term: tf / max_count for term, tf in self.curr_tokens_with_tf.items()}
        return

    def add_doc_data_to_index(self, data_str, doc_num):

        self.curr_tokens = []
        self.curr_tokens_with_tf = {}

        self.extract_tokens(data_str)
        self.compute_term_frequencies()

        for token, count in self.curr_tokens_with_tf.items():

            if token not in self.inverted_index:
                self.inverted_index[token] = {'df': 0, 'doc_tf': {}}

            self.inverted_index[token]['df'] += 1
            self.inverted_index[token]['doc_tf'][doc_num] = count

        return

    # Iterate on every document, pre-process and build inverted index
    # Return inverted_index and n = number of documents
    def build_inverted_index(self):

        # Traverse on XML files in the given directory
        for filename in os.listdir(self.corpus_directory):

            if not filename.endswith(".xml"):
                continue

            # Use the file path to parse the XML file to etree element
            file_path = os.path.join(self.corpus_directory, filename)
            tree = etree.parse(file_path)
            root = tree.getroot()

            # Use XPATH to build a list of all records in the XML file
            doc_list = root.xpath("/root/RECORD")

            # Compute record_num, list of tokens (after pre-processing) with frequency count
            for doc in doc_list:

                if self.limit and self.document_number >= VSM.MAX_DOCUMENTS:
                    break

                doc_num = int(doc.xpath("./RECORDNUM/text()")[0])

                # Use XPATH to extract all words from TITLE, ABSTRACT, EXTRACT
                data_list = doc.xpath("./TITLE/text()") + doc.xpath("./EXTRACT/text()") + doc.xpath("./ABSTRACT/text()")
                data_str = "".join(data_list)

                self.add_doc_data_to_index(data_str, doc_num)
                self.document_number += 1

        return

    # Compute IDF for every token, and add to the index
    def add_idf_scores_to_index(self):

        n = self.document_number
        for token in self.inverted_index.keys():
            n_t = self.inverted_index[token]['df']
            idf = math.log(n / n_t, 2)
            self.inverted_index[token]['idf'] = idf

        return

    # Compute document length for every document and return dictionary of all lengths
    def compute_document_lengths(self):

        # Compute document vector length (sum of squares of tf-idf)
        for token in self.inverted_index.keys():
            idf = self.inverted_index[token]['idf']
            for doc_num, tf in self.inverted_index[token]['doc_tf'].items():

                if doc_num not in self.document_lengths:
                    self.document_lengths[doc_num] = 0
                self.document_lengths[doc_num] += (tf * idf) ** 2

        # Square root to get the norm of the document
        self.document_lengths = {doc_num: math.sqrt(length) for doc_num, length in self.document_lengths.items()}

        return

    def save_index_and_lengths(self):

        index_and_lengths = {'index': self.inverted_index, 'lengths': self.document_lengths}

        with open(VSM.INDEX_PATH, 'w') as outfile:
            json.dump(index_and_lengths, outfile)

        return

    def load_index_and_lengths(self, index_path):

        with open(index_path, 'r') as index_file:
            index_and_lengths = json.load(index_file)

        self.inverted_index = index_and_lengths['index']
        self.document_lengths = index_and_lengths['lengths']

        return

    def retrieve_top_docs(self):

        query_length = 0

        for token, tf_query in self.curr_tokens_with_tf.items():

            # TODO handle this case - is continue- okay?
            if token not in self.inverted_index:
                print("the word " + token + " from the query is not in the corpus")
                continue
            i = self.inverted_index[token]['idf']
            query_length += (tf_query * i) ** 2
            k = tf_query
            w = k * i

            doc_tf = self.inverted_index[token]['doc_tf']

            for doc_num, tf_doc in doc_tf.items():
                if doc_num not in self.document_scores:
                    self.document_scores[doc_num] = 0
                self.document_scores[doc_num] += w * i * tf_doc

        l = math.sqrt(query_length)

        for doc_num in self.document_scores.keys():
            s = self.document_scores[doc_num]
            y = self.document_lengths[doc_num]
            self.document_scores[doc_num] = s / (l * y)

        self.top_docs = sorted(self.document_scores.items(), key=lambda x: x[1], reverse=True)[:VSM.RESULTS_THRESHOLD]
        return

    def save_top_docs(self):

        with open(VSM.TOP_DOCS_PATH, 'w') as outfile:
            for result in self.top_docs:
                outfile.write(result[0] + '\n')
        return


def main(argv):
    if len(argv) < 3:
        print("Not enough arguments")
        return

    vsm_model = VSM()

    if argv[1] == 'create_index':

        vsm_model.corpus_directory = argv[2]

        vsm_model.build_inverted_index()
        vsm_model.add_idf_scores_to_index()
        vsm_model.compute_document_lengths()
        vsm_model.save_index_and_lengths()

        print('Finished building index')
        return

    elif argv[1] == 'query':

        if len(argv) < 5:
            print("Not enough arguments")
            return

        ranking = argv[2]
        index_path = argv[3]
        question = argv[4]

        vsm_model.ranking = ranking

        vsm_model.load_index_and_lengths(index_path)
        vsm_model.extract_tokens(question)
        vsm_model.compute_term_frequencies()
        vsm_model.retrieve_top_docs()
        vsm_model.save_top_docs()

        print('query done')
        return

    else:
        print("Wrong arguments")
        return


if __name__ == "__main__":
    main(sys.argv)
