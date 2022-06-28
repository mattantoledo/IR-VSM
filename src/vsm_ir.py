import sys
import os
from lxml import etree

# TODO document nltk.download('popular') command in python console
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# Compute bag of words with frequencies for specific document
def compute_bag_count(tokens):

    bag = {}

    for word in tokens:
        if word in bag:
            bag[word] += 1
        else:
            bag[word] = 1

    return bag


def pre_process(record):

    # Creating dictionary from RECORD using XPath
    d = {
        'RECORDNUM': record.xpath("./RECORDNUM/text()")[0],
        'TITLE': record.xpath("./TITLE/text()"),
        'EXTRACT': record.xpath("./EXTRACT/text()"),
        'ABSTRACT': record.xpath("./ABSTRACT/text()")
    }

    # Creating list of all words in the document
    data = d['TITLE'] + d['ABSTRACT'] + d['EXTRACT']
    data = [word_tokenize(text) for text in data]
    data = [word for tokens in data for word in tokens]

    # Lowercase
    data = [word.lower() for word in data]

    # Keep only alphabetic words and words with length > 1
    data = [word for word in data if word.isalpha() and len(word) > 1]

    # Remove stopwords
    data = [word for word in data if word not in stopwords.words('english')]

    # TODO note in the documentation that we used stemming
    # Stem tokens using Porter Stemmer
    stemmer = PorterStemmer()
    data = [stemmer.stem(word) for word in data]

    return d['RECORDNUM'], data, compute_bag_count(data)


def print_inverted_index(inverted_index):
    for k, d in inverted_index.items():
        print(k)
        print(d['df'])
        for t in d['list']:
            print(t)
        print("*************")


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

            doc_num, tokens, tokens_count = pre_process(doc)

            for token, count in tokens_count.items():

                if token not in inverted_index:
                    inverted_index[token] = {'df': 0, 'list': []}

                inverted_index[token]['df'] += 1
                inverted_index[token]['list'].append((doc_num, count))

            n += 1

            if limit and n >= 15:
                print_inverted_index(inverted_index)
                return

    print_inverted_index(inverted_index)


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
