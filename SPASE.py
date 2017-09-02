from __future__ import division
import sys;
import math;
import pickle;
import numpy as np;
if (len(sys.argv) < 5):
    print('Parameters insufficient!');
    exit();
embedding_filename = sys.argv[1];
sememe_all_filename = sys.argv[2];
test_filename = sys.argv[3];
hownet_filename = sys.argv[4]
def matrix_factorization(M, wordvec, M_alter, semvec, sememe_size, steps=5, alpha=0.001, beta=0):
    word_size = wordvec.shape[0]
    dim_size = wordvec.shape[1]
    der_M_alter_sum = np.ones((word_size,sememe_size))
    der_semvec_sum = np.ones((sememe_size,dim_size))
    delta = np.zeros(wordvec.shape);
    for step in range(steps):
        der_M_alter = np.zeros((word_size,sememe_size))
        der_semvec = np.zeros((sememe_size,dim_size))
        for i in range(word_size):
            delta[i] = np.dot(M[i]*M_alter[i],semvec) - wordvec[i]
            for j in range(sememe_size):
                if (M[i][j] == 0):
                    continue;
                eij = 2 * delta[i]; 
                #M_alter[i](sememe_size),semvec[j](K)
                der_M_alter[i][j] = np.dot(eij,semvec[j].T)
                der_semvec[j] =  eij * M_alter[i][j];
                der_M_alter_sum[i][j] += der_M_alter[i][j] ** 2
                der_semvec_sum[j] += der_semvec[j] ** 2
                M_alter[i][j] = M_alter[i][j] - np.divide(alpha * der_M_alter[i][j],np.sqrt(der_M_alter_sum[i][j]));
                semvec[j] = semvec[j] - np.divide(alpha * der_semvec[j],np.sqrt(der_semvec_sum[j]));
        e = 0
        print('Process:%f' %(float(step)/steps,))
        #if (step % 5 !=0):continue
        delta = np.dot(M*M_alter,semvec) - wordvec
        e = sum(sum(delta ** 2))
        print('loss:%f' % (e/float(wordvec.size)));
        if e < 0.001:
            break
    return M_alter, semvec
with open(embedding_filename,'r') as embedding_file:

    line = embedding_file.readline();
    arr = line.strip().split();

    word_size = int(arr[0]);
    dim_size = int(arr[1]);
    embedding_vec = {};
    word_list = []
    W = [];
    hownet_dict = {}
    with open(hownet_filename,'r') as hownet:
        buf = hownet.readlines()    
        words = buf[0::2]
        word_size = len(words)
        word2sememes = buf[1::2]
        words = [item.strip() for item in words]
        index = 0
        for item in words:
            hownet_dict[item] = word2sememes[index].strip().split()
            index += 1
        print('Hownet File Reading Complete')
    for line in embedding_file:
        arr = line.strip().split();
        word = arr[0].strip();
        if (word in hownet_dict):
            word_list.append(item)
        float_arr = [];
        for i in range(1,dim_size+1):
            float_arr.append(float(arr[i]));
        regular = math.sqrt(sum([x*x for x in float_arr]));
        embedding_vec[word] = [];
        flag = 1;
        if (word not in hownet_dict):
            flag = 0;
        for i in range(1,dim_size+1):
            embedding_vec[word].append(float(arr[i])/regular);
            if (flag == 1):
                W.append(float(arr[i])/regular);
    W = np.array(W).reshape(word_size,dim_size)
    sememe_size = 0;
    sememes = []
    print('Embedding File Reading Complete')
    with open(sememe_all_filename,'r') as sememe_all:
        sememes_buf = sememe_all.readlines() ;
        sememes = sememes_buf[1].strip().strip('[]').split(' ');
        sememes = [sememe.strip().strip('\'') for sememe in sememes];
        #print(sememes);
        sememe_size = len(sememes);
        #read sememe complete
    print('Sememe File Reading Complete')
    index = 0;
    M = np.zeros((word_size,sememe_size))
    for word in word_list:
        buf = hownet_dict[word]
        for sememe in buf:
            position = sememes.index(sememe)
            M[index][position] = 1;
        index += 1;
    M_alter_init = np.random.rand(word_size,sememe_size);
    S_init = np.random.rand(sememe_size,dim_size);
    try:
        print('Checkpoint loading...')
        with open('Checkpoint_SPASE','rb') as checkpoint:
            M_alter_init = pickle.load(checkpoint)
            S_init=pickle.load(checkpoint)
        print('Checkpoint successfully loaded.')
    except:
        print('Checkpoint loading failed, initailized with random value')
    M_alter, S = matrix_factorization(M,W,M_alter_init,S_init,sememe_size);
    with open('Checkpoint_SPASE','wb') as checkpoint:
        pickle.dump(M_alter,checkpoint)
        pickle.dump(S,checkpoint)
    index = 0;
    while (index < sememe_size):
        regular = S[index].dot(S[index].T);
        S[index] = S[index] / regular;
        index += 1;
    with open('output_SPASE','w') as output:
        with open('model_SPASE','wb') as model_outpout:
            with open(test_filename,'r') as test:
                for line in test:
                    word = line.strip();
                    vec = embedding_vec[word];
                    vec = np.array(vec)
                    score_list = [];
                    index = 0;
                    while (index < sememe_size):
                        score = vec.dot(S[index].T);
                        score_list.append((sememes[index],abs(score)));
                        index += 1;
                    score_list.sort(key=lambda x:x[1],reverse=True);
                    output.write(line.strip()+'\n') ;
                    output.write(" ".join([x[0] for x in score_list])+'\n');
                    pickle.dump(score_list,model_outpout);
