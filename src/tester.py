from vsm_ir import *


class Evaluator:

    QUERY_DOCS_SCORE_PATH = './../data/cfc-xml/cfquery.xml'
    TRUE_DB_PATH = './../data/query_results/true_query_results.json'
    MY_DB_PATH = './../data/query_results/my_query_results.json'

    def __init__(self):
        self.true_query_results = {}
        self.my_query_results = {}

    def build_true_query_results(self):
        tree = etree.parse(Evaluator.QUERY_DOCS_SCORE_PATH)
        root = tree.getroot()
        queries = root.xpath("./QUERY")

        for query in queries:
            query_num = query.xpath("./QueryNumber/text()")[0]
            question = query.xpath("./QueryText/text()")[0]
            matches = query.xpath("./Records/Item")

            docs_score = {}

            for match in matches:
                doc_num = match.text
                score = match.attrib['score']
                docs_score[doc_num] = sum([int(v) for v in score])

            top_docs = sorted(docs_score.items(), key=lambda t: t[1], reverse=True)
            top_docs = [x[0] for x in top_docs]

            self.true_query_results[query_num] = {'question': question, 'top_docs': top_docs}

        return

    def build_my_query_results(self, ranking):

        vsm_model = VSM()
        vsm_model.ranking = ranking

        vsm_model.load_index_and_lengths(VSM.INDEX_PATH)

        for query_num, d in self.true_query_results.items():
            question = d['question']

            vsm_model.extract_tokens(question)
            vsm_model.compute_term_frequencies()

            if ranking == 'tfidf':
                vsm_model.retrieve_top_docs_tf_idf()
            elif ranking == 'bm25':
                vsm_model.retrieve_top_docs_bm25()

            my_top_docs = [t[0] for t in vsm_model.top_docs]
            self.my_query_results[query_num] = {'question': question, 'top_docs': my_top_docs}

        return

    def save_my_query_results(self):

        with open(Evaluator.MY_DB_PATH, 'w') as outfile:
            json.dump(self.my_query_results, outfile)
        return

    def load_my_query_results(self):
        with open(Evaluator.MY_DB_PATH, 'r') as outfile:
            self.my_query_results = json.load(outfile)
        return

    def save_true_query_results(self):

        with open(Evaluator.TRUE_DB_PATH, 'w') as outfile:
            json.dump(self.true_query_results, outfile)
        return

    def load_true_query_results(self):
        with open(Evaluator.TRUE_DB_PATH, 'r') as outfile:
            self.true_query_results = json.load(outfile)
        return


def main(argv):

    e = Evaluator()

    e.build_true_query_results()
    e.build_my_query_results('bm25')

    e.save_true_query_results()
    e.save_my_query_results()

    return


if __name__ == "__main__":
    main(sys.argv)
