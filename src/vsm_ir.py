import sys
import os
import math
from lxml import etree

# TODO document nltk.download('popular') command in python console
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# Tokenize, filter and stem the words of the document
def pre_process(record):

    stemmer = PorterStemmer()

    tokens = []

    rec_num = record.xpath("./RECORDNUM/text()")[0]

    # Use XPATH to extract all words from TITLE, ABSTRACT, EXTRACT
    data = record.xpath("./TITLE/text()") + record.xpath("./EXTRACT/text()") + record.xpath("./ABSTRACT/text()")

    # Use nltk to tokenize data into words
    data = [word_tokenize(text) for text in data]

    # Lowercase
    data = [word.lower() for tokens in data for word in tokens]

    for word in data:
        # Keep only alphabetic words and words with length > 1, Drop stopwords
        if word.isalpha() and len(word) > 1 and word not in stopwords.words('english'):
            # TODO note in the documentation that we used stemming
            # Stem tokens using Porter Stemmer
            tokens.append(stemmer.stem(word))

    return rec_num, tokens


# Compute term frequencies for a specific document
def compute_term_frequencies(terms):

    tf = {}
    max_count = 1

    for term in terms:
        if term in tf:
            tf[term] += 1
            if tf[term] > max_count:
                max_count = tf[term]
        else:
            tf[term] = 1

    for term in tf:
        tf[term] = tf[term] / max_count

    return tf


def create_index(directory_path, limit=True):

    inverted_index = {}
    n = 0

    # Traverse on XML files in the given directory
    for filename in os.listdir(directory_path):

        if not filename.endswith(".xml"):
            continue

        # Use the file path to parse the XML file to etree element
        file_path = os.path.join(directory_path, filename)
        tree = etree.parse(file_path)
        root = tree.getroot()

        # Use XPATH to build a list of all records in the XML file
        doc_list = root.xpath("/root/RECORD")

        # Compute record_num, list of tokens (after pre-processing) with frequency count
        for doc in doc_list:

            doc_num, tokens = pre_process(doc)

            tokens_count = compute_term_frequencies(tokens)

            for token, count in tokens_count.items():

                if token not in inverted_index:
                    inverted_index[token] = {'df': 0, 'doc_tf': {}}

                inverted_index[token]['df'] += 1
                inverted_index[token]['doc_tf'][doc_num] = count

            n += 1
            if limit and n >= 15:
                break

        if limit and n >= 15:
            break

    # Compute IDF for every token
    for token in inverted_index.keys():
        n_t = inverted_index[token]['df']
        idf = math.log(n/n_t, 2)
        inverted_index[token]['idf'] = idf

    print(inverted_index)


def main(argv):

    if len(argv) < 3:
        print("Not enough arguments")
        return

    if argv[1] == 'create_index':

        create_index(argv[2])
        print('Finished building index')

        return

    elif argv[1] == 'query':
        print('query')
        return

    else:
        print("Wrong arguments")
        return


if __name__ == "__main__":
    main(sys.argv)
