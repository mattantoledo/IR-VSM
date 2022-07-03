import json
import sys
import os
from lxml import etree

XML_PATH = "C:\\Users\\MattanToledo\\PycharmProjects\\IR-VSM\\data\\cfc-xml"
DOC_DATA_PATH = "C:\\Users\\MattanToledo\\PycharmProjects\\IR-VSM\\data\\doc_data\\data.json"


def build_doc_data():

    d = {}

    # Traverse on XML files in the given directory
    for filename in os.listdir(XML_PATH):

        if not filename.endswith(".xml"):
            continue

        # Use the file path to parse the XML file to etree element
        file_path = os.path.join(XML_PATH, filename)
        tree = etree.parse(file_path)
        root = tree.getroot()

        # Use XPATH to build a list of all records in the XML file
        doc_list = root.xpath("/root/RECORD")

        # Compute record_num, list of tokens (after pre-processing) with frequency count
        for doc in doc_list:

            doc_num = int(doc.xpath("./RECORDNUM/text()")[0])

            # Use XPATH to extract all words from TITLE, ABSTRACT, EXTRACT
            data_list = doc.xpath("./TITLE/text()") + doc.xpath("./EXTRACT/text()") + doc.xpath("./ABSTRACT/text()")
            data_str = "\n\n".join(data_list)

            d[doc_num] = data_str

    return d


def save_docs_data(d):

    with open(DOC_DATA_PATH, 'w') as outfile:
        json.dump(d, outfile)
    return


def main(argv):

    d = build_doc_data()
    save_docs_data(d)

    while(True):
        doc_num = int(input())
        if doc_num == 0:
            break

        data = d[doc_num]
        print(data)

    print("done")
    return


if __name__ == "__main__":
    main(sys.argv)
