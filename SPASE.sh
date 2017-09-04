#for i in {1..12}
#do
python SPASE.py train_data sememe_all hownet.txt_test train_hownet 
python scorer.py output_SPASE hownet.txt_answer >> report
#done
