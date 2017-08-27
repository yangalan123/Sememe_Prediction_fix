import pickle;
import sys;
import math;
import numpy as np;
if (len(sys.argv)<4):
    print("Parameters insufficients");
model1_filename = sys.argv[1];
model2_filename = sys.argv[2];
ratio = float(sys.argv[3]);
with open(model1_filename,'rb') as model1_file:
    with open(model2_filename,'rb') as model2_file:
        model1 = [];
        model2 = [];
        while True:
            try:
                model1.append(pickle.load(model1_file));
                model2.append(pickle.load(model2_file));
            except EOFError:
                break;
        index = 0;
        length = len(model1);
        test_words = [];
        with open('hownet.txt_test','r',encoding='utf-8') as test:
            for line in test:
                test_words.append(test.strip());
        with open('Output_Ensemble','w',encoding='utf-8') as output:
            while (index < length):
                predict0 = dict(model1[index]);
                predict1 = dict(model2[index]);
                predict = [];
                for key in predict0:
                    predict.append((key,abs(ratio/(1+ratio)*(predict0[key])+1/(1+ratio)*predict1[key])));
                predict.sort(key=lambda x:x[1],reverse=True);
                result = [x[0] for x in predict];
                output.write(test_words[index]+'\n'+str(result)+'\n');
                
            
            
