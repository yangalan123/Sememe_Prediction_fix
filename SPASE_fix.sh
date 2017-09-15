#for i in {1..12}
#do
python SPASE_fix.py train_data sememe_all hownet.txt_test train_hownet 
python scorer.py output_SPASE_fix hownet.txt_answer >> report_fix
#done
