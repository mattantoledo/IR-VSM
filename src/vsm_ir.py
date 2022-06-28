import sys
import os
from lxml import etree

# TODO document nltk.download('popular') command in python console
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def pre_process(record):

    # Creating dictionary from RECORD using XPath
    d = {
        'RECORDNUM': record.xpath("./RECORDNUM/text()"),
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

    # Remove stopwords
    data = [word for word in data if word not in stopwords.words('english')]

    # Remove punctuation
    punctuation = "!#$%&()*+-.,/:;<=>?@[]^_`{|}~\\"
    data = [word for word in data if word not in punctuation]

    # Remove single character words
    data = [word for word in data if len(word) > 1]

    # TODO note in the documentation that we used stemming
    # Stem tokens using Porter Stemmer
    stemmer = PorterStemmer()
    data = [stemmer.stem(word) for word in data]

    return data


def create_index(directory_path):

    n = 0
    limit = 1

    for filename in os.listdir(directory_path):
        if filename.endswith(".xml"):
            file_path = os.path.join(directory_path, filename)
            tree = etree.parse(file_path)
            root = tree.getroot()

            record_list = root.xpath("/root/RECORD")

            for record in record_list:

                tokens = pre_process(record)

                print("******")
                print(tokens)
                print(len(tokens))

                n += 1

                if limit and n >= 10:
                    print(n)
                    return

    print(n)


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