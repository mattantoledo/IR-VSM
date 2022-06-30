import json
import sys
import os
import math
from lxml import etree

# TODO document nltk.download('popular') command in python console
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# Get a string representing document/query
# Tokenize the string using nltk, and convert to lowercase
# Remove unwanted words (punctuation, numbers, stopwords) and do stemming using Porter Stemmer
# TODO write in the documentation that we used stemming
# Return the list of valid tokens
def clean_tokens(data_str):
    stemmer = PorterStemmer()

    data = word_tokenize(data_str)

    data = [word.lower() for word in data]

    def my_filter(word):
        return word.isalpha() and len(word) > 1 and word not in stopwords.words('english')

    data = [word for word in data if my_filter(word)]
    data = [stemmer.stem(word) for word in data]

    return data


# Tokenize, filter and stem the words of the query
def pre_process_query(query):
    stemmer = PorterStemmer()
    tokens = []

    data = word_tokenize(query)
    data = [word.lower() for word in data]
    for word in data:
        # Keep only alphabetic words and words with length > 1, Drop stopwords
        if word.isalpha() and len(word) > 1 and word not in stopwords.words('english'):
            # Stem tokens using Porter Stemmer
            tokens.append(stemmer.stem(word))

    return tokens


# Compute term frequencies for a list of tokens (document/query)
# Normalize by maximum count
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


# Iterate on every document, pre-process and build inverted index
# Return inverted_index and n = number of documents
def build_inverted_index(directory_path, limit, n):

    inverted_index = {}
    i = 0

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

            if limit and i >= n:
                break

            doc_num = int(doc.xpath("./RECORDNUM/text()")[0])

            # Use XPATH to extract all words from TITLE, ABSTRACT, EXTRACT
            data_list = doc.xpath("./TITLE/text()") + doc.xpath("./EXTRACT/text()") + doc.xpath("./ABSTRACT/text()")

            data_str = "".join(data_list)

            tokens = clean_tokens(data_str)

            tokens_count = compute_term_frequencies(tokens)

            for token, count in tokens_count.items():

                if token not in inverted_index:
                    inverted_index[token] = {'df': 0, 'doc_tf': {}}

                inverted_index[token]['df'] += 1
                inverted_index[token]['doc_tf'][doc_num] = count

            i += 1

    print(inverted_index)
    return inverted_index, i


# Compute IDF for every token, and add to the index
def add_idf_to_inverted_index(inverted_index, n):
    for token in inverted_index.keys():
        n_t = inverted_index[token]['df']
        idf = math.log(n / n_t, 2)
        inverted_index[token]['idf'] = idf

    return inverted_index


# Compute document length for every document and return dictionary of all lengths
def compute_document_lengths(inverted_index):
    document_lengths = {}

    # Compute document vector length (sum of squares of tf-idf)
    for token in inverted_index.keys():
        idf = inverted_index[token]['idf']
        for doc_num, tf in inverted_index[token]['doc_tf'].items():

            if doc_num not in document_lengths:
                document_lengths[doc_num] = 0
            document_lengths[doc_num] += (tf * idf) ** 2

    # Square root to get the norm of the document
    for doc_num in document_lengths.keys():
        document_lengths[doc_num] = math.sqrt(document_lengths[doc_num])

    return document_lengths


def create_index(directory_path, limit, n):
    inverted_index, k = build_inverted_index(directory_path, limit, n)

    inverted_index = add_idf_to_inverted_index(inverted_index, k)

    document_lengths = compute_document_lengths(inverted_index)

    index_and_lengths = {'index': inverted_index, 'lengths': document_lengths}

    with open('vsm_inverted_index.json', 'w') as outfile:
        json.dump(index_and_lengths, outfile)


def handle_query(ranking, index_path, question, th):
    with open(index_path, 'r') as index_file:
        index_and_lengths = json.load(index_file)

    inverted_index = index_and_lengths['index']
    document_lengths = index_and_lengths['lengths']

    tokens = pre_process_query(question)

    tokens_count = compute_term_frequencies(tokens)

    document_scores = {}

    query_length = 0

    for token, count in tokens_count.items():

        # TODO handle this case
        if token not in inverted_index:
            print("the word " + token + " from the query is not in the corpus")
        i = inverted_index[token]['idf']
        query_length += (count * i) ** 2
        k = count
        w = k * i

        doc_tf = inverted_index[token]['doc_tf']

        for doc_num, tf in doc_tf.items():
            if doc_num not in document_scores:
                document_scores[doc_num] = 0
            document_scores[doc_num] += w * i * tf

    l = math.sqrt(query_length)

    for doc_num in document_scores.keys():
        s = document_scores[doc_num]
        y = document_lengths[doc_num]
        document_scores[doc_num] = s / (l * y)

    result = sorted(document_scores.items(), key=lambda x: x[1], reverse=True)[:th]

    return [t[0] for t in result]


def test(ranking, index_path):
    tree = etree.parse("./../data/cfc-xml/cfquery.xml")
    root = tree.getroot()

    queries = root.xpath("./QUERY")

    queries_db = {}

    for query in queries:
        num = query.xpath("./QueryNumber/text()")[0]
        question = query.xpath("./QueryText/text()")[0]
        matches_count = int(query.xpath("./Results/text()")[0])
        matches = query.xpath("./Records/Item")

        items = {}

        for match in matches:
            doc_num = match.text
            score = match.attrib['score']
            s = sum([int(v) for v in score])
            score_db = {'text': score, 'sum': s, 'avg': s / len(score)}

            items[doc_num] = score_db

        queries_db[num] = {'question': question, 'matches_count': matches_count, 'matches': items}

    top_matches_db = {}
    top2 = {}

    for query_num in queries_db.keys():
        top_matches = sorted(queries_db[query_num]['matches'].items(), key=lambda x: x[1]['sum'], reverse=True)[:10]
        top_matches_db[query_num] = top_matches
        top2[query_num] = [t[0] for t in top_matches]

    for query_num, top_list in top2.items():
        question = queries_db[query_num]['question']

        my_list = handle_query(ranking, index_path, question, th=10)

        print(top_list)
        print(my_list)
        print("*****")


def main(argv):
    if len(argv) < 3:
        print("Not enough arguments")
        return

    if argv[1] == 'create_index':

        corpus_directory = argv[2]

        create_index(corpus_directory, limit=True, n=10)

        print('Finished building index')
        return

    elif argv[1] == 'query':

        if len(argv) < 5:
            print("Not enough arguments")
            return

        ranking = argv[2]
        index_path = argv[3]
        question = argv[4]

        handle_query(ranking, index_path, question, th=10)

        test(ranking, index_path)

        print('query done')
        return

    else:
        print("Wrong arguments")
        return


if __name__ == "__main__":
    main(sys.argv)
