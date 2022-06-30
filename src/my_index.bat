@ECHO OFF

::create_index
set "mode=create_index"
set "directory_corpus=C:\Users\MattanToledo\PycharmProjects\IR-VSM\data\cfc-xml"

python vsm_ir.py "%mode%" "%directory_corpus%"

