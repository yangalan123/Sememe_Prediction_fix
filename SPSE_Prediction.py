import numpy as np;
import pickle;
import sys;
import math;
from math import ceil;
if (len(sys.argv)<6):
    print('Parameters insufficient!');
    exit();
sememe_embedding_filename = sys.argv[1];
sememe_all_filename = sys.argv[2];
word_embedding_filename = sys.argv[3];
question_filename = sys.argv[4];
target_filename = sys.argv[5]
target_filename = target_filename.strip().strip('>').strip();
with open(sememe_embedding_filename,'rb') as sememe_embedding_file:
    with open(sememe_all_filename,'r',encoding='utf-8') as sememe_all:
        with open(word_embedding_filename,'r',encoding='utf-8') as embedding_file: 
                sememe_embeddings = pickle.load(sememe_embedding_file);
                bias_word = pickle.load(sememe_embedding_file);
                bias_sememe = pickle.load(sememe_embedding_file)
                line = embedding_file.readline();
                arr = line.strip().split();
                word_size = int(arr[0]);
                dim_size = int(arr[1]);
                embedding_vec = {};
                word2bias = {};
                #W = [];
                index = 0;
                for line in embedding_file:
                    arr = line.strip().split();
                    float_arr = [];
                    for i in range(1,dim_size+1):
                        float_arr.append(float(arr[i]));
                    regular = math.sqrt(sum([x*x for x in float_arr]));
                    word = arr[0].strip();
                    embedding_vec[word] = [];
                    word2bias[word] = bias_word[index];
                    for i in range(1,dim_size+1):
                        embedding_vec[word].append(float(arr[i])/regular);
                    index += 1;
                        #W.append(float(arr[i])/regular);
                #W = np.array(W).reshape(word_size,dim_size);
                index = 0;
                sememe_count = int(sememe_all.readline());
                sememes = sememe_all.readline().strip().strip('[]').split(',');
                sememes = [x.strip().strip('\'') for x in sememes];
                sem2vec = {};
                sem2bias = {} ;
                for sememe in sememes:
                    tmpvec = sememe_embeddings[index] + sememe_embeddings[index+1];
                    regular = math.sqrt(tmpvec.dot(tmpvec.T));
                    tmpvec /= regular;
                    sem2vec[sememe] = tmpvec;
                    sem2bias[sememe] = bias_sememe[int(index/2)];
                    index += 2;
                with open(question_filename,'r',encoding = 'utf-8') as question_file:
                    with open(target_filename,'w',encoding = 'utf-8') as output:
                        for line in question_file:
                            output.write(line.strip()+'\n');
                            score = [];
                            word = line.strip();
                            vec = np.array(embedding_vec[word]);
                            for sememe in sememes:
                                score.append((sememe,abs(sem2vec[sememe].dot(vec.transpose())+word2bias[word]+sem2bias[sememe])));
                            score.sort(key=lambda x:x[1],reverse=True);
                            result = [x[0] for x in score];
                            output.write(str(result)+'\n');
                            #with open('score_SPSE','ab') as score_file:
                                #pickle.dump(score,score_file);
                            



