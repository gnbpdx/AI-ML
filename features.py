#!/usr/bin/python3
import os
import math
import nltk
import collections
class Authorship_Classifier():
    #Given a book located at filename, this outputs a list of paragraphs
    def convert_text_to_paragraphs(self, filename):
        paragraphs = list()
        with open(filename) as f:
            paragraph = 'a'
            while(paragraph != ''):
                paragraph = ''
                line = 'a'
                while(line != '\n' and line != ''):
                    line = f.readline()
                    paragraph += line
                if paragraph != '' and paragraph != '\n':
                    paragraphs.append(paragraph)
        return paragraphs
    #Converts book to one string
    def convert_text_to_string(self, filename):
        text = ''
        with open(filename) as f:
            line = 'a'
            count = 0
            while(line != ''):
                line = f.readline()
                text += line
                count += 1
        return text
    #Converts multiple books to one string
    def convert_texts_to_string(self, filenames):
        text_string = ''
        for file in filenames:
            text_string += self.convert_text_to_string(file)
        return text_string
    #Computes set of letters present in paragraph
    def letters_in_paragraph(self, paragraph):
        features = set()
        for letter in paragraph:
            if letter.isalpha():
                features.add(letter.lower())
        return features
    #Computes set of words that are in a paragraph
    def words_in_paragraph(self, paragraph):
        #Hyphen, en-dash, em-dash, and hypen-minus
        dashes = ['\u2010', '\u2013', '\u2014', '\u002D']
        punctuation = [',', '.', ';', ':', '!', '\"', '?', '“', '”', '(', ')', '*', '_']
        for dash in dashes:
            paragraph = paragraph.replace(dash, ' ')
        for mark in punctuation:
            paragraph = paragraph.replace(mark, ' ')
        words = paragraph.split() 
        for index in range(len(words)):
            word = words[index]
            if word[len(word) - 1] in punctuation:
                words[index] = word[:len(word) - 1]
        #Make words lowercase
        words = [word.lower() for word in words]
        return set(words)
    #Computes set of stemmed words in a paragraph
    def stemmed_words_in_paragraph(self, paragraph):
        #Porter Stemmer
        stemmer = nltk.stem.PorterStemmer()
        #Hyphen, en-dash, em-dash, and hyphen-minus
        dashes = ['\u2010', '\u2013', '\u2014', '\u002D']
        punctuation = [',', '.', ';', ':', '!', '\"', '?', '“', '”', '(', ')', '*', '_']
        for dash in dashes:
            paragraph = paragraph.replace(dash, ' ')
        for mark in punctuation:
            paragraph = paragraph.replace(mark, ' ')
        words = paragraph.split() 
        for index in range(len(words)):
            word = words[index]
            if word[len(word) - 1] in punctuation:
                words[index] = word[:len(word) - 1]
        #Make words lowercase and stem words
        words = [stemmer.stem(word.lower()) for word in words]
        return set(words)
    #Calculates top 300 features 
    def compute_features(self, statistics, total_features):
        feature_gain = {}
        for feature in total_features:
            probability_shelley_positive, probability_shelley_negative, probability_positive = statistics[feature]
            entropy_shelley = 0
            if probability_shelley_positive > 0 and probability_shelley_positive < 1:
                entropy_shelley = -probability_shelley_positive * math.log2(probability_shelley_positive) - (1-probability_shelley_positive) * math.log2(1-probability_shelley_positive)
            entropy_other = 0
            if probability_shelley_negative > 0 and probability_shelley_negative < 1:
                entropy_other = -probability_shelley_negative * math.log2(probability_shelley_negative) - (1-probability_shelley_negative) * math.log2(1-probability_shelley_negative)
            feature_gain[feature] = -(entropy_shelley * probability_positive + entropy_other * (1 - probability_positive))
        features = sorted(feature_gain.items(), key=lambda x: x[1], reverse=True)
        return features[:300]
    #Prints datasets in csv form
    def print_csv(self, features, shelley_paragraphs, other_paragraphs, shelley_paragraph_names, other_paragraph_names):
        shelley_paragraph_features = {}
        other_paragraph_features = {}
        for paragraph in shelley_paragraphs:
            shelley_paragraph_features[paragraph] = self.get_features_in_paragraph(paragraph)
        for paragraph in other_paragraphs:
            other_paragraph_features[paragraph] = self.get_features_in_paragraph(paragraph)
        index = 0
        for paragraph in shelley_paragraphs:
            print(shelley_paragraph_names[index] + ','  + str(1), end=',')
            for feature_index in range(len(features)):
                feature, _ = features[feature_index]
                if feature in shelley_paragraph_features[paragraph]:
                    print(str(1), end='')
                else:
                    print(str(0), end='')
                if feature_index != len(features) - 1:
                    print(',', end='')
            print()
            index += 1
        index = 0
        for paragraph in other_paragraphs:
            print(other_paragraph_names[index] + ','  + str(0), end=',')
            for feature_index in range(len(features)):
                feature, _ = features[feature_index]
                if feature in other_paragraph_features[paragraph]:
                    print(str(1), end='')
                else:
                    print(str(0), end='')
                if feature_index != len(features) - 1:
                    print(',', end='')
            print()
            index += 1
    #Runs program on txt files
    def main(self):
        #Get files to read from
        files = os.listdir()
        files = list(filter(lambda x: len(x) > 4 and x[-4:] == '.txt', files))
        #Files that correspond to Shelley as the author
        shelley_files = list(filter(lambda x: 'shelley' in x, files))
        #File that don't correspond to Shelley as the author
        other_files = list(filter(lambda x: not 'shelley' in x, files))
        shelley_paragraphs = []
        other_paragraphs = []
        shelley_paragraph_names = []
        other_paragraph_names = []
        #Amass paragraphs from books by Shelley
        for file in shelley_files:
            new_paragraphs = self.convert_text_to_paragraphs(file)
            shelley_paragraphs += new_paragraphs
            for i in range(1, len(new_paragraphs) + 1):
                shelley_paragraph_names.append(file + "." + str(i))
        #Amass paragraphs from books not by Shelley
        for file in other_files:
            new_paragraphs = self.convert_text_to_paragraphs(file)
            other_paragraphs += new_paragraphs
            for i in range(1, len(new_paragraphs) + 1):
                other_paragraph_names.append(file + "." + str(i))

        #Convert all books to a string
        text = self.convert_texts_to_string(files)
        #Get all distinct words from books
        all_features = self.get_all_features(text, shelley_paragraphs, other_paragraphs)
        #Compute probabilites 
        statistics = self.compute_statistics(shelley_paragraphs, other_paragraphs, all_features)
        #Compute chosen features
        features = self.compute_features(statistics, all_features)
        #Print data set with chosen features
        self.print_csv(features, shelley_paragraphs, other_paragraphs, shelley_paragraph_names, other_paragraph_names)

    #Returns probabilites for each feature in a dictionary
    def compute_statistics(self, shelley_paragraphs, other_paragraphs, total_features):
        statistics = {}
        shelley_paragraph_features = {}
        other_paragraph_features = {}
        for paragraph in shelley_paragraphs:
            shelley_paragraph_features[paragraph] = self.get_features_in_paragraph(paragraph)
        for paragraph in other_paragraphs:
            other_paragraph_features[paragraph] = self.get_features_in_paragraph(paragraph)
        for feature in total_features:
            #Calculate probability that the feature is in a shelley paragraph given that the feature appears in a paragraph
            num_shelley_paragraphs = 0
            for paragraph in shelley_paragraphs:
                if feature in shelley_paragraph_features[paragraph]:
                    num_shelley_paragraphs += 1
            num_other_paragraphs = 0
            for paragraph in other_paragraphs:
                if feature in other_paragraph_features[paragraph]:
                    num_other_paragraphs += 1
            if num_shelley_paragraphs == 0 and num_other_paragraphs == 0:
                probability_shelley_positive = 0
            else:
                probability_shelley_positive = num_shelley_paragraphs / (num_shelley_paragraphs + num_other_paragraphs)
            if num_shelley_paragraphs == len(shelley_paragraphs) and num_other_paragraphs == len(other_paragraphs):
                probability_shelley_negative = 0
            else:
                probability_shelley_negative = (len(shelley_paragraphs) - num_shelley_paragraphs) / ((len(shelley_paragraphs) - num_shelley_paragraphs) + len(other_paragraphs) - num_other_paragraphs)
            #Calculate the probability that the feature appears in a paragraph
            probability_positive = (num_shelley_paragraphs + num_other_paragraphs) / (len(shelley_paragraphs) + len(other_paragraphs))
            statistics[feature] = (probability_shelley_positive, probability_shelley_negative, probability_positive)
        return statistics
    #Returns set of all features
    def get_all_features(self, text, shelley_paragraphs, other_paragraphs):
        pass
    #Returns set of features in paragraph
    def get_features_in_paragraph(self, paragraph):
        pass
   

#Standard classifier with features as lowercase words
class Word_Classifier(Authorship_Classifier):
    def get_features_in_paragraph(self, paragraph):
        return self.words_in_paragraph(paragraph)  

    def get_all_features(self, text, shelley_paragraphs, other_paragraphs):
        return self.words_in_paragraph(text)

class Word_Strip_Classifier(Authorship_Classifier):
    def __init__(self, n):
        self.strip_num = n
    def get_features_in_paragraph(self, paragraph):
        return set([word for word in self.words_in_paragraph(paragraph) if len(word) > self.strip_num])
    def get_all_features(self, text, shelley_paragraphs, other_paragraphs):
        return set([word for word in self.words_in_paragraph(text) if len(word) > self.strip_num])
#Classifier with features as stems of lowercase words
class Stemmed_Word_Classifier(Authorship_Classifier):
    def get_features_in_paragraph(self, paragraph):
        return self.stemmed_words_in_paragraph(paragraph)

    def get_all_features(self, text, shelley_paragraphs, other_paragraphs):
        return self.stemmed_words_in_paragraph(text)

#Classifier with features as letter frequency
class Letter_Classifier(Authorship_Classifier):
    def get_features_in_paragraph(self, paragraph):
        return self.letters_in_paragraph(paragraph)
    def get_all_features(self, text, shelley_paragraphs, other_paragraphs):
        alphabet = 'a b c d e f g h i j k l m n o p q r s t u v w x y z'
        letters = alphabet.split()
        return set(letters)

#Classifier with word length, paragraph length, letter frequency of each letter, and stemmed words as features
class Feature_Classifier(Authorship_Classifier):

    def average_word_length(self, paragraph):
        length = 0
        words = self.words_in_paragraph(paragraph)
        for word in words:
            length += len(word)
        length /= len(words)
        return length

    def get_features_in_paragraph(self, paragraph):
        features = set()
        if self.average_word_length(paragraph) <= self.average_length:
            features.add('word length')
        if len(paragraph) <= self.average_paragraph_length:
            features.add('paragraph length')
        return features | self.letters_in_paragraph(paragraph) | self.stemmed_words_in_paragraph(paragraph)
    def get_all_features(self, text, shelley_paragraphs, other_paragraphs):
        self.average_length = 0
        for paragraph in shelley_paragraphs:
            self.average_length += self.average_word_length(paragraph)
        self.average_length /= len(shelley_paragraphs)
        self.average_paragraph_length = 0
        for paragraph in shelley_paragraphs:
            self.average_paragraph_length += len(paragraph)
        self.average_paragraph_length /= len(shelley_paragraphs)
        alphabet = 'a b c d e f g h i j k l m n o p q r s t u v w x y z'
        letters = alphabet.split()
        return set(['word length', 'paragraph length']) | set(letters) | self.stemmed_words_in_paragraph(paragraph)
def main():
    authorship = Word_Classifier()
    authorship.main()
if __name__ == '__main__':
    main()
