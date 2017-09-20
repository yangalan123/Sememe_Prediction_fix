#for i in {1..12}
#do
python SPASE_2.py train_data sememe_all hownet.txt_test train_hownet >>report3
python scorer.py output_SPASE_2 hownet.txt_answer >> report
#done
