@ECHO OFF

::query
set "mode=query"
set "ranking=tfidf"
set "index_path=C:\Users\MattanToledo\PycharmProjects\IR-VSM\src\vsm_inverted_index.json"
set "question=What is the most effective regimen for the use of pancreatic enzyme supplements in the treatment of CF patients?"

python vsm_ir.py "%mode%" "%ranking%" "%index_path%" "%question%"