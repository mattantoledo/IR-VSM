@ECHO OFF

::query
set "mode=query"
set "ranking=bm25"
set "index_path=C:\Users\MattanToledo\PycharmProjects\IR-VSM\src\vsm_inverted_index.json"
set "question=How effective is bronchial lavage in CF patients?"

python vsm_ir.py "%mode%" "%ranking%" "%index_path%" "%question%"