from vsm_ir import *

QUERY_DATA_PATH = './../data/cfc-xml/cfquery.xml'


# Given a etree element representing a query, return a dictionary {doc_num:score}
def compute_relevant_documents(query):
    
    scores = {}

    matches = query.xpath("./Records/Item")

    for match in matches:
        doc_num = match.text
        score = match.attrib['score']
        scores[doc_num] = sum([int(v) for v in score]) / len(score)

    return scores


# Given a list of scores (relevancy by judges), return the dcg score
def compute_dcg(relevance):

    if not relevance:
        return 0
    dcg_score = relevance[0]

    for i in range(1, len(relevance)):
        dcg_score += relevance[i] / math.log2(i+1)
    return dcg_score


# Given a dictionary of relevant documents, compute the idcg score
def compute_idcg(relevant_documents):
    
    sorted_relevant_documents = sorted(relevant_documents, key=lambda x: relevant_documents[x], reverse=True)

    relevance = []
    for i in range(min(VSM.RESULTS_THRESHOLD, len(sorted_relevant_documents))):
        relevance.append(relevant_documents[sorted_relevant_documents[i]])
    
    return compute_dcg(relevance)


# Compare results of all queries, compute evaluation estimators
def test(ranking):

    vsm_model = VSM()
    vsm_model.load_index_and_lengths(VSM.INDEX_PATH)

    tree = etree.parse(QUERY_DATA_PATH)
    root = tree.getroot()
    queries = root.xpath("./QUERY")

    count = 0
    avg_ndcg = 0
    avg_precision = 0
    avg_recall = 0
    avg_f = 0

    for query in queries:

        count += 1
        relevant_documents = compute_relevant_documents(query)

        question = query.xpath("./QueryText/text()")[0]
        retrieved_documents = vsm_model.retrieve_top_docs(ranking, question)

        relevance = []
        inter_size = 0

        for (doc_num, score) in retrieved_documents:
            rel = 0
            if doc_num in relevant_documents:
                inter_size += 1
                rel += relevant_documents[doc_num]

            relevance.append(rel)

        precision = inter_size / len(retrieved_documents) if retrieved_documents else 0
        recall = inter_size / len(relevant_documents)

        dcg_score = compute_dcg(relevance)
        idcg_score = compute_idcg(relevant_documents)
        ndcg_score = dcg_score / idcg_score

        f = (2 * precision * recall) / (precision + recall) if (precision+recall) > 0 else 0

        avg_ndcg += ndcg_score
        avg_precision += precision
        avg_recall += recall
        avg_f += f

    print(ranking)
    res = "{:.3f}".format(avg_ndcg / count)
    print("Average NDCG@10 = " + res)
    res = "{:.3f}".format(avg_precision / count)
    print("Average Precision = " + res)
    res = "{:.3f}".format(avg_recall / count)
    print("Average Recall = " + res)
    res = "{:.3f}".format(avg_f / count)
    print("Average F = " + res)
    return


def main(argv):

    test('tfidf')
    test('bm25')


if __name__ == "__main__":
    main(sys.argv)
