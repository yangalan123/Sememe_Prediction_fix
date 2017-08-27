import sys;
import math;
import pickle;
import numpy as np;
if (len(sys.argv) < 4):
    print('Parameters insufficient!');
    exit();
embedding_filename = sys.argv[1];
sememe_all_filename = sys.argv[2];
test_filename = sys.argv[3];
def matrix_factorization(R, P, Q, K, steps=20, alpha=0.01, beta=0):
    Q = Q.T;
    line = R.shape[0]
    col = R.shape[1]
    der_P_sum = np.ones((line,K))
    der_Q_sum = np.ones((col,K))
    for step in range(steps):
        der_P = np.zeros((line,K))
        der_Q = np.zeros((col,K))
        for i in range(line):
            for j in range(col):
                eij = np.dot(P[i],Q[j].T) -R[i][j]
                #P[i](K),Q[j](K)
                der_P[i] +=  (2 * eij * Q[j] );
                der_Q[j] +=  (2 * eij * P[i] );
                der_P_sum[i] += der_P[i] ** 2
                der_Q_sum[j] += der_Q[j] ** 2
                P[i] = P[i] - np.divide(alpha * der_P[i],np.sqrt(der_P_sum[i]));
                Q[j] = Q[j] - np.divide(alpha * der_Q[j],np.sqrt(der_Q_sum[j]));
        e = 0
        for i in range(line):
            for j in range(col):
                e = e + pow(R[i][j] - np.dot(P[i],Q[j]), 2)
        print('Process:%f,loss:%f' % (step/steps,e/float(R.size)));
        if e < 0.001:
            break
    return P, Q.T
with open(embedding_filename,'r',encoding = 'utf-8') as embedding_file:

    line = embedding_file.readline();
    arr = line.strip().split();

    word_size = int(arr[0]);
    dim_size = int(arr[1]);
    embedding_vec = {};
    W = [];
    for line in embedding_file:
        arr = line.strip().split();
        float_arr = [];
        for i in range(1,dim_size+1):
            float_arr.append(float(arr[i]));
        regular = math.sqrt(sum([x*x for x in float_arr]));
        word = arr[0].strip();
        embedding_vec[word] = [];
        for i in range(1,dim_size+1):
            embedding_vec[word].append(float(arr[i])/regular);
            W.append(float(arr[i])/regular);
    W = np.array(W).reshape(word_size,dim_size)
    N = word_size;
    M = dim_size;
    with open(sememe_all_filename,'r',encoding='utf-8') as sememe_all:
        sememes_buf = sememe_all.readlines() ;
        sememes = sememes_buf[1].strip().strip('[]').split(',');
        sememes = [sememe.strip().strip('\'') for sememe in sememes];
        #print(sememes);
        sememe_size = len(sememes);
        #read sememe complete
    K = sememe_size;
    M_alter_init = np.random.rand(N,K);
    S_init = np.random.rand(K,dim_size);
    with open('checkpoint_SPASE','rb') as checkpoint:
        M_alter_init = pickle.load(checkpoint)
        S_init=pickle.load(checkpoint)
    M_alter, S = matrix_factorization(W,M_alter_init,S_init,K);
    with open('checkpoint_SPASE','wb') as checkpoint:
        pickle.dump(M_alter,checkpoint)
        pickle.dump(S,checkpoint)
    index = 0;
    while (index < K):
        regular = S[index].dot(S[index].T);
        S[index] = S[index] / regular;
        index += 1;
    with open('output_SPASE_Opt','w',encoding='utf-8') as output:
        with open('score_SPASE_Opt','ab') as score_outpout:
            with open(test_filename,'r',encoding='utf-8') as test:
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
                    output.write(str(score_list)+'\n');
                    pickle.dump(score_list,score_outpout);
                        
                        
                        

