# Information Retrieval using Vector Space Model

This project is focused on implementing information retrieval techniques, including the Vector Space Model (VSM), TF-IDF, and BM25. It aims to retrieve relevant documents based on queries related to cystic fibrosis articles.

## Project Structure


The project is structured as follows:

- `src/vsm_ir.py`: This file contains the implementation of the Vector Space Model and the main file for execution.


## Requirements

The project has the following requirements:

- Python 3.X
- nltk
- lxml

## Usage

To use the project, follow these steps:

1. Install the required dependencies by running the following command:  
`pip install -r requirements.txt`


2. Build the inverted index by running the following command:  
```python main.py create_index <corpus_directory>```  
Replace `<corpus_directory>` with the path to the directory containing the corpus of documents in XML format.


3. Perform a query using either the TF-IDF or BM25 ranking methods by running the following command:  
`python main.py query <ranking> <index_path> "<query>"`  
Replace `<ranking>` with either `tfidf` or `bm25` to specify the ranking method.  
Replace `<index_path>` with the path to the index file generated in step 2.  
Replace `<query>` with the query string surrounded by quotes.  
This will retrieve the top relevant documents based on the given query and ranking method.


4. Evaluate the performance of the VSM. You can run the evaluation for both ranking methods using the following command:  
`python main.py evaluate`  
This will compute average NDCG@k, precision, recall, and F-score for a set of queries specified in the `QUERY_DATA_PATH` variable.
