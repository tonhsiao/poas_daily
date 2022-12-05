cd /d D:/Nicole/python/poas/bigTree
call activate poas-env
REM pause
call python preprocess_data.py
REM pause
call python drop_model_bigtree.py
REM pause
call python getSentimentAnalysis_bigtree.py
REM pause
call python multi_label_predict_bigtree.py
REM pause
exit /B