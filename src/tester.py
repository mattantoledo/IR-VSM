from vsm_ir import *

QUERY_DATA_PATH = './../data/cfc-xml/cfquery.xml'


# Given a etree element representing a query, return a dictionary {doc_num:score}
def compute_true_scores(query):
    scores = {}

    matches = query.xpath("./Records/Item")

    for match in matches:
        doc_num = match.text
        score = match.attrib['score']
        scores[doc_num] = sum([int(v) for v in score]) / len(score)

    return scores


# Given a list of scores (relevancy by judges), return the dcg score
def compute_dcg(relevance):

    dcg_score = relevance[0]

    for i in range(1, len(relevance)):
        dcg_score += relevance[i] / math.log2(i+1)
    return dcg_score


# Compare results of all queries, compute evaluation estimators
def test(ranking):

    vsm_model = VSM()
    vsm_model.load_index_and_lengths(VSM.INDEX_PATH)

    tree = etree.parse(QUERY_DATA_PATH)
    root = tree.getroot()
    queries = root.xpath("./QUERY")

    i = 0
    s = 0
    t = 0
    w = 0
    z = 0

    for query in queries:

        i += 1
        true_scores = compute_true_scores(query)

        question = query.xpath("./QueryText/text()")[0]
        top_docs = vsm_model.retrieve_top_docs(ranking, question)

        relevance = []
        inter = 0

        for (doc_num, score) in top_docs:
            rel = 0
            if doc_num in true_scores:
                inter += 1
                rel += true_scores[doc_num]

            relevance.append(rel)

        precision = inter / len(top_docs)
        recall = inter / len(true_scores)

        dcg_score = compute_dcg(relevance)
        idcg_score = compute_dcg(sorted(relevance, reverse=True))

        ndcg_score = dcg_score / idcg_score if idcg_score > 0 else 0

        f = (2 * precision * recall) / (precision + recall) if (precision+recall) > 0 else 0

        s += ndcg_score
        t += precision
        w += recall
        z += f

    print(ranking)
    res = "{:.3f}".format(s / i)
    print("Average NDCG@10 = " + res)
    res = "{:.3f}".format(t / i)
    print("Average Precision = " + res)
    res = "{:.3f}".format(w / i)
    print("Average Recall = " + res)
    res = "{:.3f}".format(z / i)
    print("Average F = " + res)
    return


def main(argv):

    test('tfidf')
    test('bm25')


if __name__ == "__main__":
    main(sys.argv)
