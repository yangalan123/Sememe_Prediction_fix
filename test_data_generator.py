import sys;
import random;

if (len(sys.argv)<3):
    print("no enough parameter");
    exit();
hownet_filename = sys.argv[1];
embedding_filename = sys.argv[2];
answer_filename = hownet_filename+"_answer_revised";
test_filename = hownet_filename+"_test_revised";
with open(hownet_filename,'r',encoding='utf-8') as hownet:
    with open(test_filename,'w',encoding='utf-8') as test:
        with open(answer_filename,'w',encoding='utf-8') as answer:
            with open(embedding_filename,'r',encoding='utf-8') as embedding:
                data = hownet.readlines();
                dataBuf = [];
                for line in data:
                    dataBuf.append(line.strip());
                data = dataBuf;
                wordsBuf = embedding.readlines();
                sourcewords = [];
                length = len(wordsBuf);
                for i in range(1,length):
                    line = wordsBuf[i].strip();
                    arr = line.split();
                    sourcewords.append(arr[0]);
                words = data[0::2];
                sememes = data[1::2];
                data = list(zip(words,sememes));
                samples = random.sample(sourcewords,int(len(sourcewords)*0.1));
                samplesBuf = [];
                for word in samples:
                    try:
                        position = words.index(word.strip());
                        sememe = sememes[position];
                        samplesBuf.append((word,sememe));
                    except:
                        print(samples.index(word));
                samples = samplesBuf;
                for word,sememe in samples:
                    test.write(word+'\n');
                    answer.write(word+'\n'+sememe+'\n');
                with open('train_hownet','w',encoding='utf-8') as train:
                    for word,sememe in zip(words,sememes):
                        try:
                            position = samples.index((word,sememe));
                        except:
                            train.write(word+'\n'+sememe+'\n');
                        
