python SPSE_Training.py train_hownet train_data sememe_all SPSE_embedding
python SPSE_Prediction.py SPSE_embedding sememe_all train_data hownet.txt_test output_SPSE 
python scorer.py output_SPSE hownet.txt_answer
