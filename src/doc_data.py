import sys
import os
from lxml import etree

XML_PATH = "C:\\Users\\MattanToledo\\PycharmProjects\\IR-VSM\\data\\cfc-xml"


# Given a doc_num returns the title and the abstract of the document
def retrieve_doc_data(doc_num):

    # Traverse on XML files in the given directory
    for filename in os.listdir(XML_PATH):

        if not filename.endswith(".xml"):
            continue

        # Use the file path to parse the XML file to etree element
        file_path = os.path.join(XML_PATH, filename)
        tree = etree.parse(file_path)
        root = tree.getroot()

        # Use XPATH to find the matching document record
        doc_list = root.xpath("/root/RECORD")

        for doc in doc_list:
            curr_doc_num = doc.xpath("./RECORDNUM/text()")[0]
            if int(curr_doc_num) == int(doc_num):
                data_list = doc.xpath("./TITLE/text()") + doc.xpath("./EXTRACT/text()") + doc.xpath("./ABSTRACT/text()")
                data_str = "\n\n".join(data_list)
                return data_str

    return None


# loop to get user input of doc_num and prints the data of the document
# stop when input is 0
def main():

    while True:
        doc_num = input()
        if doc_num == '0':
            break

        data = retrieve_doc_data(doc_num)
        print('\n' + data + '\n')

    print("done")
    return


if __name__ == "__main__":
    main()
