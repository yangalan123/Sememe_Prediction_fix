import random;
import pickle;
import math;
import numpy as np;
#from scipy.optimize import minimize;
import sys;
np.set_printoptions(threshold=np.nan);
if (len(sys.argv)<5):
    exit(0);
hownet_filename = sys.argv[1];
embedding_filename = sys.argv[2];
sememe_all_filename = sys.argv[3];
target_filename = sys.argv[4];
para_lambda = 0.5;
max_iter = 200;
with open(hownet_filename,'r',encoding='utf-8') as hownet:
    with open(embedding_filename,'r',encoding='utf-8') as embedding_file:
        with open(sememe_all_filename,'r',encoding='utf-8') as sememe_all:
            with open(target_filename,'rb') as target:
                sememe_embedding = pickle.load(target);
                bias_word = pickle.load(target);
                bias_sememe = pickle.load(target)
            with open(target_filename+'_2','wb') as target:
                sememes_buf = sememe_all.readlines() ;
                sememes = sememes_buf[1].strip().strip('[]').split(',');
                sememes = [sememe.strip().strip('\'') for sememe in sememes];
                #print(sememes);
                sememe_size = len(sememes);
                #read sememe complete
                word2sememe = {}
                while True:
                    word = hownet.readline().strip();
                    sememes_tmp = hownet.readline().strip().split();
                    #print(word);
                    if (word or sememes_tmp):
                        word2sememe[word] = [] ;
                        length = len(sememes_tmp);
                        for i in range(0,length):
                            word2sememe[word].append(sememes_tmp[i]);
                    else: break; 
                #read hownet complete
                print("hownet reading complete")
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

                #read embedding complete
                print('Embedding reading complete');
                with open('PMI.txt','r') as PMI:
                    P = []
                    for line in PMI:
                        arr = line.strip().split();
                        arr = [float(e) for e in arr];
                        P.extend(arr);
                    P = np.array(P).reshape(sememe_size,sememe_size);
                    Judge = np.ones((sememe_size,sememe_size),dtype=np.int);
                    for i in range(0,sememe_size - 1):
                        for j in range(0,sememe_size - 1):
                            if (P[i][j] == 0):
                                rand = random.randint(1,1000);
                                if (rand>5):
                                    Judge[i][j] = 0;
                    M = np.zeros((word_size,sememe_size));
                    M_Judge = np.ones((word_size,sememe_size));
                    se_index = 0;
                    word_index = 0;
                    for word in embedding_vec:
                        for sememe in word2sememe[word]:
                            se_index = sememes.index(sememe);
                            M[word_index][se_index] = 1;
                        word_index += 1;
                    for i in range(0,word_size):
                        for j in range(0,sememe_size):
                            if (M[i][j] == 0):
                                rand = random.randint(1,1000)
                                if (rand>5):
                                    M_Judge[i][j] = 0
                    print("PMI calculating complete");
                    #sememe_embedding = np.random.randn(sememe_size*2,dim_size);
                    #sememe_embedding_dersum = np.ones((sememe_size*2,dim_size)) ;
                    #bias_sememe = np.random.randn(sememe_size,1);
                    #bias_sememe_dersum = np.ones((sememe_size,1)) ;
                    #bias_word = np.random.randn(word_size,1);
                    #bias_word_dersum = np.ones((word_size,1)) ;
                    #print('Initailization complete');
                    def ModelLoss(x):
                        losssum = 0;
                        for i in range(0,sememe_size):
                            sem0 = x[2 * i];
                            sem1 = x[2 * i + 1];
                            for j in range(0,word_size):
                                if (M_Judge[j][i] == 0):
                                    continue;
                                w = W[j].reshape(1,dim_size);
                                delta = w.dot((sem0+sem1).transpose())+bias_sememe[i]+bias_word[j]-M[j][i];
                                losssum += delta * delta;
                        for i in range(0,sememe_size):
                            for j in range(0,sememe_size):
                                if (Judge[i][j]):
                                    delta = x[2 * i].dot(x[2 * j + 1].transpose())-P[i][j];
                                    losssum += para_lambda * delta * delta;
                        return losssum;
                    def derivate(x):
                        der = np.zeros((sememe_size * 2,dim_size));
                        der_for_bias_word = np.zeros((word_size,1));
                        der_for_bias_sememe = np.zeros((sememe_size,1));
                        for i in range(0,sememe_size):
                            sem0 = x[2 * i];
                            sem1 = x[2 * i + 1];
                            line_search = np.zeros((1,dim_size));
                            for j in range(0,word_size):
                                if (M_Judge[j][i] == 0):
                                    continue;
                                w = W[j].reshape(1,dim_size);
                                delta = w.dot((sem0+sem1).transpose())+bias_sememe[i]+bias_word[j]-M[j][i];
                                line_search += delta * 2 * w;
                                der_for_bias_word[j] += 2 * delta;
                                der_for_bias_sememe[i] += 2 * delta;
                            der[2 * i] += line_search.reshape(dim_size,);
                            der[2 * i + 1] += line_search.reshape(dim_size,);
                        for i in range(0,sememe_size):
                            line_search = np.zeros((1,dim_size));
                            for j in range(0,sememe_size):
                                if (Judge[i][j]):
                                    delta = x[2 * i].dot(x[2 * j + 1].transpose())-P[i][j];
                                    line_search += para_lambda * delta * 2 * (x[2 * j + 1].reshape(1,dim_size));
                                    der[2 * j + 1] += para_lambda * delta * 2 * (x[2 * i].reshape(dim_size,));
                            der[2 * i] += line_search.reshape(dim_size,);
                        return der,der_for_bias_word,der_for_bias_sememe;
                    learning_rate = 0.00005;
                    for i in range(1,max_iter):
                        #learning_rate = learning_rate / i;
                        #if (i %  == 0):
                        loss = ModelLoss(sememe_embedding);
                        print("Process:%f,learning_rate:%f,loss:%f" %(i/max_iter,learning_rate,loss));
                        der,der_for_bias_word,der_for_bias_sememe = derivate(sememe_embedding);
                        if (i % 40 == 0):
                            learning_rate = learning_rate/2;
                        #if (i % 10 == 0):
                            #with open('checkpoint_'+str(i//10)+'.txt','w') as checkpoint:
                                #for line in sememe_embedding:
                                    #checkpoint.write(str(line).strip()+'\n');
                                #with open('bias_sememe_checkpoint_'+str(i//10)+'.txt','w') as bias_sememe_checkpoint:
                                    #for line in bias_sememe:
                                        #bias_sememe_checkpoint.write(str(line).strip()+'\n');
                                    #with open('bias_word_checkpoint_'+str(i//10)+'.txt','w') as bias_word_checkpoint:
                                        #for line in bias_word:
                                            #bias_word_checkpoint.write(str(line).strip()+'\n');
                            #print("checkpoint over");
                        sememe_embedding += -learning_rate * der;
                        bias_sememe += -learning_rate * der_for_bias_sememe;
                        bias_word += -learning_rate * der_for_bias_word;
                        #sememe_embedding += -learning_rate * np.divide(der,np.sqrt(sememe_embedding_dersum));
                        #sememe_embedding_dersum += der ** 2;
                        #bias_sememe += -learning_rate * np.divide(der_for_bias_sememe,np.sqrt(bias_sememe_dersum));
                        #bias_sememe_dersum += der_for_bias_sememe ** 2;
                        #bias_word += -learning_rate * np.divide(der_for_bias_word,np.sqrt(bias_word_dersum));
                        #bias_word_dersum += der_for_bias_word ** 2;
                    #target.write(str(sememe_embedding));
#                    for line in sememe_embedding:
#                        target.write(str(line).strip()+'\n');
                    pickle.dump(sememe_embedding,target);
                    pickle.dump(bias_word,target);
                    pickle.dump(bias_sememe,target);

                    #result = minimize(ModelLoss,sememe_embedding,method='BFGS',jac=derivate,options={
                        #'disp':True
                        #}) 
                    #target.write(result.x);
                     
