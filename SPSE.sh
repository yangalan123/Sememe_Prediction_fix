python SPSE_Training_half_PMI.py hownet.txt train_data sememe_all result_SPSE_half_PMI
python SPSE_Prediction.py result_SPSE_half_PMI sememe_all train_data hownet.txt_test output_SPSE_half_PMI
python scorer.py output_SPSE_half_PMI hownet.txt_answer > res1.txt
python SPSE_Training_origin.py hownet.txt train_data sememe_all result_SPSE_origin
python SPSE_Prediction.py result_SPSE_origin sememe_all train_data hownet.txt_test output_SPSE_origin 
python scorer.py output_SPSE_origin hownet.txt_answer > res2.txt
python SPSE_Training_individual.py hownet.txt train_data sememe_all result_SPSE_individual
python SPSE_Prediction.py result_SPSE_individual sememe_all train_data hownet.txt_test output_SPSE_individual
python scorer.py output_SPSE_individual hownet.txt_answer > res3.txt