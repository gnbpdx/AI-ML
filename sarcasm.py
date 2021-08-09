import nltk
import sklearn
import sklearn.naive_bayes
import sklearn.model_selection
import json
import functools
import collections
import numpy
#Load data from file
def load_data(filename):
    with open(filename) as f:
        data = []
        readline = None
        while readline != '':
            readline = f.readline()
            if readline != '':
                data.append(json.loads(readline))
    return data

def load_NRC_data(filename):
    with open(filename) as f:
        f.readline() #skip first line
        data = {}
        readline = None
        while readline != '':
            readline = f.readline()
            if readline != '':
                readline = readline.split()
                data[readline[0]] = (readline[1], float(readline[2]))
    return data
def get_features_ngrams(dataset, labels):
    #Seperate training and test data (Note this training data will only be used to generate which ngrams are used as features)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(dataset, labels, train_size=0.2, test_size=0.8)

    #Construct ngrams from data
    unigrams = []
    bigrams = []
    trigrams = []
    for headline in X_train:
        words = nltk.tokenize.word_tokenize(headline)
        for word in words:
            unigrams.append(word)
        for index in range(len(words) - 1):
            bigrams.append((words[index], words[index+1]))
        for index in range(len(words) - 2):
            trigrams.append((words[index], words[index+1], words[index+2]))

    #Count number of each unigram, bigram, and trigram
    unigram_frequencies = collections.Counter(unigrams)
    bigram_frequencies = collections.Counter(bigrams)
    trigram_frequencies = collections.Counter(trigrams)

    #Sort ngrams by frequency
    sorted_unigrams = sorted(unigram_frequencies, reverse=True, key= lambda x: unigram_frequencies[x])
    sorted_bigrams = sorted(bigram_frequencies, reverse=True, key = lambda x: bigram_frequencies[x])
    sorted_trigrams = sorted(trigram_frequencies, reverse=True, key = lambda x: trigram_frequencies[x])

    
    testing_instances = numpy.zeros(shape=(len(X_test), 3000))
    testing_labels = numpy.zeros(len(X_test))

    #Calculate features in test data
    for index in range(len(X_test)):
        testing_labels[index] = y_test[index]
        words = nltk.tokenize.word_tokenize(X_test[index])
        for i in range(1000):
            for word in words:
                if sorted_unigrams[i] == word:
                    testing_instances[index][i] += 1 

        bigrams = []
        trigrams = []
        for index in range(len(words) - 1):
            bigrams.append((words[index], words[index+1]))
        for index in range(len(words) - 2):
            trigrams.append((words[index], words[index+1], words[index+2]))
            
        for i in range(1000):
            for bigram in bigrams:
                if sorted_bigrams[i] == bigram:
                    testing_instances[index][i + 1000] += 1
        for i in range(1000):
            for trigram in trigrams:
                if sorted_trigrams[i] == trigram:
                    testing_instances[index][i + 2000] += 1
    return testing_instances, testing_labels

def get_features_mix(dataset, labels):
    data = load_NRC_data('NRC-Emotion-Intensity-Lexicon-v1.txt')
    emotions = list(set([data[word][0] for word in data]))

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(dataset, labels, train_size=0.2, test_size=0.8)
    dataset, labels = X_test, y_test

    testing_instances = numpy.zeros(shape=(len(dataset), len(emotions) + 3006))
    testing_labels = numpy.zeros(len(dataset))

    #Add sum of intensities of an emotion and the number of emotional words
    for instance in range(len(dataset)):
        testing_labels[instance] = labels[instance]
        words = nltk.tokenize.word_tokenize(dataset[instance])
        for word in words:
            if word in data:
                emotion, intensity = data[word]
                index = emotions.index(emotion)
                testing_instances[instance][3000 + index] += intensity
                testing_instances[instance][3000 + len(emotions)] += 1

    #Add number of words in headline
    for instance in range(len(dataset)):
        testing_instances[instance][len(emotions) + 3001] = len(nltk.tokenize.word_tokenize(dataset[instance]))

    #Add number of nouns, verbs, adjectives, and adverbs
    for instance in range(len(dataset)):
        words = nltk.tokenize.word_tokenize(dataset[instance])
        words = nltk.pos_tag(words)

        for word in words:
            #Add nouns
            if word[1] == 'NN' or word[1] == 'NNS':
                testing_instances[instance][len(emotions) + 3002] += 1

            #Add verbs
            if word[1]  == 'VB' or word[1] == 'VBD' or word[1] == 'VBG' or word[1] == 'VBN' or word[1] == 'VBP' or word[1] == 'VBZ':
                testing_instances[instance][len(emotions) + 3003] += 1

            #Add adjectives
            if word[1] == 'JJR' or word[1] == 'JJS' or word[1] == 'JJ':
                testing_instances[instance][len(emotions) + 3004] += 1

            #Add adverbs
            if word[1] == 'RB' or word[1] == 'RBR' or word[1] == 'RBS':
                testing_instances[instance][len(emotions) + 3005] += 1

    #Construct ngrams from data
    positive_unigrams = []
    negative_unigrams = []
    unigrams = []
    for i in range(len(X_train)):
        words = nltk.tokenize.word_tokenize(X_train[i])
        for word in words:
            unigrams.append(word)
            if labels[i] == 1:
                positive_unigrams.append(word)
            else:
                negative_unigrams.append(word)
    #Count number of each unigram for positive and negative instances
    positive_unigram_counts = collections.Counter(positive_unigrams)
    negative_unigram_counts = collections.Counter(negative_unigrams)
    unigram_counts = collections.Counter(unigrams)
    print (positive_unigram_counts['fsaojioas'])

    total_positive_unigrams = sum([unigram for unigram in positive_unigram_counts])
    total_negative_unigrams = sum([unigram for unigram in negative_unigrams])





    #Sort ngrams by frequency
    sorted_unigrams = sorted(unigram_frequencies, reverse=True, key= lambda x: unigram_frequencies[x])
    sorted_bigrams = sorted(bigram_frequencies, reverse=True, key = lambda x: bigram_frequencies[x])
    sorted_trigrams = sorted(trigram_frequencies, reverse=True, key = lambda x: trigram_frequencies[x])

    #Add n-grams
    for index in range(len(X_test)):
        words = nltk.tokenize.word_tokenize(X_test[index])
        for i in range(1000):
            for word in words:
                if sorted_unigrams[i] == word:
                    testing_instances[index][i] += 1 

        bigrams = []
        trigrams = []
        for index in range(len(words) - 1):
            bigrams.append((words[index], words[index+1]))
        for index in range(len(words) - 2):
            trigrams.append((words[index], words[index+1], words[index+2]))
            
        for i in range(1000):
            for bigram in bigrams:
                if sorted_bigrams[i] == bigram:
                    testing_instances[index][i + 1000] += 1
        for i in range(1000):
            for trigram in trigrams:
                if sorted_trigrams[i] == trigram:
                    testing_instances[index][i + 2000] += 1
    return testing_instances, testing_labels

def classifier(X, y, classifier, classifier_name):

    #Perform 10-fold cross validation on classifier
    scores = sklearn.model_selection.cross_validate(classifier, X, y, cv=10, scoring=('f1_macro', 'f1_micro', 'accuracy'))

    #Print out the Accuracy and F1-scores
    print('Classifier: ', classifier_name)
    print('Average Macro F1 Score: ', sum(scores['test_f1_macro']) / 10)
    print('Average Micro F1 Score: ', sum(scores['test_f1_micro']) / 10)
    print('Average Accuracy: ', sum(scores['test_accuracy']) / 10)
    print()

if __name__ == '__main__':
    #Load data from file
    data = load_data('Sarcasm_Headlines_Dataset.json')

    #Seperate data and labels
    dataset = [datum['headline'] for datum in data]
    labels = [datum['is_sarcastic'] for datum in data]

    #Get percentage of sarcastic examples
    percentage_sarcastic = sum(labels) / len(labels)
    print('Percentage of dataset that is sarcastic: ', percentage_sarcastic)
    print()

    #Get features from ngrams
    #X, y = get_features_ngrams(dataset, labels)

    #Run Naive Bayes
    #classifier(X, y, sklearn.naive_bayes.MultinomialNB(), 'Multinomial Naive Bayes (ngrams)')

    #Get features from mixed features
    X, y = get_features_mix(dataset, labels)

    #Run SVM
    classifier(X, y, sklearn.svm.LinearSVC(), 'Support Vector Machine (Mixed features)')
