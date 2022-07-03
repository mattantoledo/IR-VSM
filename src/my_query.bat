@ECHO OFF

::query
set "mode=query"
set "ranking=tfidf"
set "index_path=C:\Users\MattanToledo\PycharmProjects\IR-VSM\src\vsm_inverted_index.json"
set "question=What is the pathophysiologic role of circulating antibodies to Pseudomonas aeruginosa in CF patients?"

python vsm_ir.py "%mode%" "%ranking%" "%index_path%" "%question%"