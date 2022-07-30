@ECHO OFF

::query
set "mode=query"
set "ranking=bm25"
set "index_path=C:\Users\MattanToledo\PycharmProjects\IR-VSM\src\vsm_inverted_index.json"
set "question=What is the role of Vitamin E in the therapy of patients with CF?"

python vsm_ir.py "%mode%" "%ranking%" "%index_path%" "%question%"