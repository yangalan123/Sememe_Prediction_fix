#Example for SPWE && SPSE
#Make sure that you have run SPSE.sh && SPWE.sh
python Ensemble_model.py score_SPWE score_SPSE 2.1 hownet.txt_test_revised
python scorer.py output_Ensemble hownet.txt_answer_revised
