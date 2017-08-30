# SPmodels
The code for **Lexical Sememe Prediction via Word Embeddings and Matrix Factorization**(IJCAI2017)

## Running Requirement
**Memory**: at least 8GB, 16GB or more is recommended.
**Storage**: at least 15GB, 20GB or more is recommended. 
(Although other devices has not been specified, we strong recommend that use devices of high performance)

## How to Run
1. Prepare a file that contains Chinese word embeddings(of Google Word2Vec form).We recommend that the amount of words be at least 200,000 and the dimention be at least 200. It will achieve much better result using a large ( 20GB or more is recommended.) corpus to train your embeddings for running this program.

2. Rename the word embedding file as 'embedding_200.txt' and put it under the directory.

3. Run data_generator.sh, the program will automatically generate evaluation data and other data files required during training.

4. Run SPSE.sh/SPWE.sh/SPASE.sh , the corresponding model will be automatically training and evaluated. 
(As for SPASE model, we recommend that run SPASE.sh for serveral times until the average cost is much less than 1. It will take pretty much time to train SPASE model)

5. Run Ensemble_Model.sh after you have run SPSE.sh and SPWE.sh. (Please check Ensemble_Model.sh, you will get more information about how to run other combinations of models (only support combining 2 models at once)

## Data Set
``hownet.txt`` is an Chinese knowledge base with annotated word-sense-sememe information
 
## Evaluation Set
 After you have run data_generator.sh, you will see 'hownet.txt_test' and 'hownet.txt_answer' file under the directory. These two files make the evaluation set. The size of the evaluation set is 10% of the full size of the part of embedding_200.txt which is anotated in hownet.txt. The evaluation set is generated by random choices.

## Result file
Feel free to get insight of the file which is named after 'output_', these files contains the sememe predictions for  evaluation set. 

You can also use pickle library in python to load the files is named after 'model_', for more information, refer to Ensemble_model.py. 
