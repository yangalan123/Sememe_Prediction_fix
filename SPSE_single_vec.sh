@'
python SPSE_Training_single_vec.py hownet.txt train_data sememe_all result_rand_fix_single
python SPSE_Prediction_single_vec.py result_rand_fix_single sememe_all train_data hownet.txt_test
python scorer.py result_rand_fix_single hownet.txt_test_answer
'@
