import sys;
if (len(sys.argv)<3):
    print("not enough parameters!");
    exit();
test_filename = sys.argv[1];
answer_filename = sys.argv[2];
scores = [];
with open(test_filename,'r',) as test:
    with open(answer_filename,'r',) as answer:
        while (True):
            test_word = test.readline().strip();
            answer_word = answer.readline().strip();
            if (len(test_word)==0 or len(answer_word)==0):
                break;
            #print(test_word);
            #print(answer_word);
            while (test_word != answer_word):  #some word not exist in embeddings
                answer.readline();
                answer_word = answer.readline().strip();
            test_sememes = test.readline().strip().strip('[]').split(',');
            answer_sememes = answer.readline().strip().strip('[]').split(' ');
            point = 0;
            length = len(test_sememes);
            for i in range(0,length):
                #print(test_sememes[i]);
                test_sememes[i] = test_sememes[i].strip().strip('\'');
                #print(i);
                #print(test_sememes[i]);
            if (len(answer_sememes)==0): 
                continue;
            #print(test_sememes);
            #print(answer_sememes);
            index = 1;
            for item in (answer_sememes):
                try:
                   rank = test_sememes.index(item);
                   #print(rank);
                   point += float(index) / (rank+1);
                   index+=1;
                except:
                   point +=0;
                   index+=1;
                   continue;
            point /= len(answer_sememes);
            #print(point);
            scores.append(point);
print("result:%f" % (sum(scores)/len(scores),));
