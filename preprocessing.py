import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter

def preprocessing(listoftokens):
    #removing stopwords
    stop_words = set(stopwords.words('english')) 
    words = [w for w in listoftokens if not w in stop_words] 
    #lemetizing
    lemmatizer = WordNetLemmatizer()
    lematizedwords = [lemmatizer.lemmatize(w) for w in words]
    return lematizedwords

if __name__ == '__main__':
	uniquewords = Counter()
	freq = 3
	allreports = []
	with open('/content/drive/My Drive/HPCProject/eclipse_platform.csv') as bugfile:
	    line = bugfile.readline()
	    line = bugfile.readline()
	    count = 0
	    while line:
	        cells = line.split(',')
	        report = cells[5]
	        tokenizer = RegexpTokenizer(r'\w+')
	        text = preprocessing(tokenizer.tokenize(report.strip().lower()))
	        for word in text:
	            if word in uniquewords.keys():
	                uniquewords[word] +=1
	            else:
	                uniquewords[word] = 1

	        allreports.append(text)
	        if count%1000 == 0:
	            print(count)
	        count +=1
	        line = bugfile.readline()
	bugfile.close()
	print(len(uniquewords))
	wordindex = 0
	uniqueindexwords = {}
	for ele in uniquewords.keys():
	    if uniquewords[ele] > freq:
	        uniqueindexwords[ele] = wordindex
	        wordindex +=1
	print(len(uniqueindexwords))

	dictionary = open("/content/drive/My Drive/HPCProject/dictionary.txt", "w")
	for key in uniqueindexwords.keys():
	    dictionary.write(key+'\t'+str(uniqueindexwords[key])+'\n')
	dictionary.close()
	count = 0
	reportfile = open('/content/drive/My Drive/HPCProject/reports.txt','w')
	for report in allreports:
	    minimizedreport = [ele for ele in report if ele in uniqueindexwords.keys()]
	    minimizedreport = minimizedreport[:95]
	    minimizedreport = [uniqueindexwords[ele] for ele in minimizedreport]
	    if len(minimizedreport) == 0:
	      continue
	    content = str(len(minimizedreport)+1)+' '+" ".join([str(ele) for ele in minimizedreport])+'\n';
	    reportfile.write(content)
	    if count%1000 == 0:
	      print(count)
	    count +=1
	    #minimizedreports.append(minimizedreport)
	reportfile.close()