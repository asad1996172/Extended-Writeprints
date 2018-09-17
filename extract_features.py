def warn(*args, **kwargs):
    pass


import json
import warnings
import estnltk

warnings.warn = warn
from os.path import splitext
from os.path import join

import itertools
import io
import os
import nltk
import spacy
import numpy as np
from os import walk
from keras.preprocessing import text
nlp = spacy.load('en_core_web_sm')
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
cur_dir_path = os.getcwd() + "/"  # + "writeprints_experiments/"


############## FEATURES COMPUTATION #####################


def getCleanText(inputText):
    # cleanText = text.text_to_word_sequence(inputText,filters='!#$%&()*+,-./:;<=>?@[\\]^_{|}~\t\n"\' ', lower=False, split=" ")
    cleanText = text.text_to_word_sequence(
        inputText, filters='', lower=False, split=" ")

    cleanText = ''.join(str(e) + " " for e in cleanText)
    return cleanText


def getTotalWords(inputText):
    tokens = text.text_to_word_sequence(
        inputText, filters=' !#$%&()*+,-./:;<=>?@[\\]^_{|}~\t\n"', lower=True, split=" ")
    return len(tokens)


def charactersCount(inputText):
    '''
    Calculates character count including spaces
    '''
    inputText = inputText.lower().replace(" ", "")
    charCount = len(str(inputText))
    return charCount


def averageCharacterPerWord(inputText):
    '''
    Calculates average number of characters per word
    '''

    words = text.text_to_word_sequence(
        inputText, filters=' !#$%&()*+,-./:;<=>?@[\\]^_{|}~\t\n"', lower=True, split=" ")
    inputText = inputText.lower().replace(" ", "")
    charCount = len(str(inputText))

    avgCharCount = charCount / len(words)
    return avgCharCount


def frequencyOfLetters(inputText):
    '''
    Calculates the frequency of letters
    '''

    inputText = str(inputText).lower()  # because its case sensitive
    inputText = inputText.lower().replace(" ", "")
    characters = "abcdefghijklmnopqrstuvwxyz"
    charsFrequencyDict = {}
    for c in range(0, len(characters)):
        char = characters[c]
        charsFrequencyDict[char] = 0
        for i in str(inputText):
            if char == i:
                charsFrequencyDict[char] = charsFrequencyDict[char] + 1

    # making a vector out of the frequencies
    vectorOfFrequencies = [0] * len(characters)
    totalCount = sum(list(charsFrequencyDict.values()))
    for c in range(0, len(characters)):
        char = characters[c]
        vectorOfFrequencies[c] = charsFrequencyDict[char] / totalCount

    return vectorOfFrequencies


def mostCommonLetterBigrams(inputText):
    # to do
    bigrams = ["th", "he", "in", "er", "an", "re", "nd", "on", "en",
               "at", "ou", "ed", "ha", "to", "or", "it", "is", "hi", "es", "ng"]
    bigramsD = {}
    for t in bigrams:
        bigramsD[t] = True

    bigramCounts = {}
    for t in bigramsD:
        bigramCounts[t] = 0

    totalCount = 0
    for i in range(0, len(inputText) - 2):
        bigram = str(inputText[i:i + 2]).lower()
        if bigram in bigramsD:
            bigramCounts[bigram] = bigramCounts[bigram] + 1
            totalCount = totalCount + 1

    bigramsFrequency = []
    for t in bigrams:
        bigramsFrequency.append(float(bigramCounts[t] / totalCount))

    return bigramsFrequency


def mostCommonLetterTrigrams(inputText):
    # to do
    trigrams = ["the", "and", "ing", "her", "hat", "his", "tha", "ere", "for",
                "ent", "ion", "ter", "was", "you", "ith", "ver", "all", "wit", "thi", "tio"]
    trigramsD = {}
    for t in trigrams:
        trigramsD[t] = True

    trigramCounts = {}
    for t in trigramsD:
        trigramCounts[t] = 0

    totalCount = 0
    for i in range(0, len(inputText) - 3):
        trigram = str(inputText[i:i + 3]).lower()
        if trigram in trigramsD:
            trigramCounts[trigram] = trigramCounts[trigram] + 1
            totalCount = totalCount + 1

    trigramsFrequency = []
    for t in trigrams:
        trigramsFrequency.append(float(trigramCounts[t] / totalCount))

    return trigramsFrequency


def digitsPercentage(inputText):
    '''
    Calculates the percentage of digits out of total characters
    '''
    inputText = inputText.lower().replace(" ", "")
    charsCount = len(str(inputText))
    digitsCount = list(
        [1 for i in str(inputText) if i.isnumeric() == True]).count(1)
    return digitsCount / charsCount


def charactersPercentage(inputText):
    '''
    Calculates the percentage of characters out of total characters
    '''

    inputText = inputText.lower().replace(" ", "")
    characters = "abcdefghijklmnopqrstuvwxyz"
    allCharsCount = len(str(inputText))
    charsCount = list([1 for i in str(inputText) if i in characters]).count(1)
    return charsCount / allCharsCount


def upperCaseCharactersPercentage(inputText):
    '''
    Calculates the percentage of uppercase characters out of total characters
    '''

    inputText = inputText.replace(" ", "")
    characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    allCharsCount = len(str(inputText))
    charsCount = list([1 for i in str(inputText) if i in characters]).count(1)
    return charsCount / allCharsCount


def frequencyOfDigits(inputText):
    '''
    Calculates the frequency of digits
    '''

    inputText = str(inputText).lower()  # because its case sensitive
    inputText = inputText.lower().replace(" ", "")
    digits = "0123456789"
    digitsFrequencyDict = {}
    for c in range(0, len(digits)):
        digit = digits[c]
        digitsFrequencyDict[digit] = 0
        for i in str(inputText):
            if digit == i:
                digitsFrequencyDict[digit] = digitsFrequencyDict[digit] + 1

    # making a vector out of the frequencies
    vectorOfFrequencies = [0] * len(digits)
    totalCount = sum(list(digitsFrequencyDict.values())) + 1
    for c in range(0, len(digits)):
        digit = digits[c]
        vectorOfFrequencies[c] = digitsFrequencyDict[digit] / totalCount

    return vectorOfFrequencies


def frequencyOfDigitsNumbers(inputText, digitLength):
    '''
    Calculates the frequency of digits
    '''

    inputText = str(inputText).lower()  # because its case sensitive
    words = text.text_to_word_sequence(
        inputText, filters=' !#$%&()*+,-./:;<=>?@[\\]^_{|}~\t\n"', lower=True, split=" ")

    count = 0
    wordCount = len(words)
    for w in words:
        if w.isnumeric() == True and len(w) == digitLength:
            count = count + 1

    return count / wordCount


def frequencyOfWordLength(inputText):
    '''
    Calculate frequency of words of specific lengths upto 15
    '''
    lengths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    wordLengthFrequencies = {}
    for l in lengths:
        wordLengthFrequencies[l] = 0

    words = text.text_to_word_sequence(
        inputText, filters=' !#$%&()*+,-./:;<=>?@[\\]^_{|}~\t\n"', lower=True, split=" ")
    for w in words:
        wordLength = len(w)
        if wordLength in wordLengthFrequencies:
            wordLengthFrequencies[wordLength] = wordLengthFrequencies[wordLength] + 1

    frequencyVector = [0] * (len(lengths))
    totalCount = sum(list(wordLengthFrequencies.values()))
    for w in wordLengthFrequencies:
        frequencyVector[w - 1] = wordLengthFrequencies[w] / totalCount

    return frequencyVector


def frequencyOfSpecialCharacters(inputText):
    '''
    Calculates the frequency of special characters
    '''

    inputText = str(inputText).lower()  # because its case insensitive
    inputText = inputText.lower().replace(" ", "")
    specialCharacters = open(
        cur_dir_path + "special_chars.txt", "r").readlines()
    specialCharacters = [s.strip("\n") for s in specialCharacters]
    specialCharactersFrequencyDict = {}
    for c in range(0, len(specialCharacters)):
        specialChar = specialCharacters[c]
        specialCharactersFrequencyDict[specialChar] = 0
        for i in str(inputText):
            if specialChar == i:
                specialCharactersFrequencyDict[specialChar] = specialCharactersFrequencyDict[specialChar] + 1

    # making a vector out of the frequencies
    vectorOfFrequencies = [0] * len(specialCharacters)
    totalCount = sum(list(specialCharactersFrequencyDict.values())) + 1
    for c in range(0, len(specialCharacters)):
        specialChar = specialCharacters[c]
        vectorOfFrequencies[c] = specialCharactersFrequencyDict[specialChar] / totalCount

    return vectorOfFrequencies


def functionWordsPercentage(inputText):
    functionWords = open(
        cur_dir_path + "functionWord.txt", "r").readlines()
    functionWords = [f.strip("\n") for f in functionWords]
    words = text.text_to_word_sequence(
        inputText, filters=' !#$%&()*+,-./:;<=>?@[\\]^_{|}~\t\n"', lower=True, split=" ")
    functionWordsIntersection = set(words).intersection(set(functionWords))
    return len(functionWordsIntersection) / len(list(words))


def frequencyOfPunctuationCharacters(inputText):
    '''
    Calculates the frequency of special characters
    '''

    inputText = str(inputText).lower()  # because its case insensitive
    inputText = inputText.lower().replace(" ", "")
    specialCharacters = open(
        cur_dir_path + "punctuation.txt", "r").readlines() # put all puchutaion marks in this file
    specialCharacters = [s.strip("\n") for s in specialCharacters]
    specialCharactersFrequencyDict = {}
    for c in range(0, len(specialCharacters)):
        specialChar = specialCharacters[c]
        specialCharactersFrequencyDict[specialChar] = 0
        for i in str(inputText):
            if specialChar == i:
                specialCharactersFrequencyDict[specialChar] = specialCharactersFrequencyDict[specialChar] + 1

    # making a vector out of the frequencies
    vectorOfFrequencies = [0] * len(specialCharacters)
    totalCount = sum(list(specialCharactersFrequencyDict.values())) + 1
    for c in range(0, len(specialCharacters)):
        specialChar = specialCharacters[c]
        vectorOfFrequencies[c] = specialCharactersFrequencyDict[specialChar] / totalCount

    return vectorOfFrequencies


def misSpellingsPercentage(inputText):
    misspelledWords = open(
        cur_dir_path + "misspellings.txt", "r").readlines() # add misspleing words
    misspelledWords = [f.strip("\n") for f in misspelledWords]
    words = text.text_to_word_sequence(
        inputText, filters=' !#$%&()*+,-./:;<=>?@[\\]^_{|}~\t\n"', lower=True, split=" ")
    misspelledWordsIntersection = set(words).intersection(set(misspelledWords))
    return len(misspelledWordsIntersection) / len(list(words))


def legomena(inputText):
    freq = nltk.FreqDist(word for word in inputText.split())
    hapax = [key for key, val in freq.items() if val == 1]
    dis = [key for key, val in freq.items() if val == 2]
    try:
        return (len(hapax) / len(dis)) / len(inputText.split())
    except:
        return 0


def posTagFrequency(inputText):
    pos_tags = []
    doc = nlp(str(inputText))
    for token in doc:
        pos_tags.append(str(token.pos_))
    # print(pos_tags)
    tagset = ['ADJ', 'ADP', 'ADV', 'CONJ', 'DET',
              'NOUN', 'NUM', 'PRT', 'PRON', 'VERB', '.', 'X']
    tags = [tag for tag in pos_tags]

    return list(tuple(tags.count(tag) / len(tags) for tag in tagset))


def posTagBigramFrequency(inputText):
    pos_tags = []
    doc = nlp(str(inputText))
    for token in doc:
        pos_tags.append(str(token.pos_))
    # print(pos_tags)
    # print(doc)
    tagset = ['ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRT', 'PRON', 'VERB', '.', 'X']
    tagset = list(itertools.combinations(tagset, 2))
    tagset_freq = [0] * len(tagset)
    # print(tagset)
    for i in range(1, len(pos_tags)):
        if (pos_tags[i - 1], pos_tags[i]) in tagset:
            tagset_freq[tagset.index((pos_tags[i - 1], pos_tags[i]))] += 1
            # print (pos_tags[i-1],pos_tags[i])
    # tags = [tag for tag in pos_tags]
    return [(val / (len(pos_tags) - 1)) for val in tagset_freq]

def posTagTrigramFrequency(inputText):
    pos_tags = []
    doc = nlp(str(inputText))
    for token in doc:
        pos_tags.append(str(token.pos_))
    # print(pos_tags)
    # print(doc)
    tagset = ['ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRT', 'PRON', 'VERB', '.', 'X']
    tagset = list(itertools.combinations(tagset, 3))
    tagset_freq = [0] * len(tagset)
    # print(tagset)
    for i in range(2, len(pos_tags)):
        if (pos_tags[i - 2],pos_tags[i - 1], pos_tags[i]) in tagset:
            tagset_freq[tagset.index((pos_tags[i - 2],pos_tags[i - 1], pos_tags[i]))] += 1
            # print (pos_tags[i-1],pos_tags[i])
    # tags = [tag for tag in pos_tags]
    return [(val / (len(pos_tags) - 2)) for val in tagset_freq]


def frequenciesOf1_20LetterWords(inputText):
    buckets = [0] * 20
    words = text.text_to_word_sequence(inputText, filters=' !#$%&()*+,-./:;<=>?@[\\]^_{|}~\t\n"', lower=True, split=" ")
    for word in words:
        if len(word) <= 20:
            buckets[len(word) - 1] += 1
    return [(val / (len(words))) for val in buckets]


def getParahgraphsDetails(inputText):
    document = estnltk.Text(inputText)
    total_parahs = len(document.paragraph_texts)
    lengths = []
    for parah in document.paragraph_texts:
        words = text.text_to_word_sequence(parah, filters=' !#$%&()*+,-./:;<=>?@[\\]^_{|}~\t\n"', lower=True, split=" ")
        lengths.append(len(words))
    lengths = np.array(lengths)
    average_length = lengths.mean()
    return [total_parahs, average_length]

def getUnigramFeatures(inputText):
    inputText = inputText.lower()
    freq_list = [0]*len(unigrams)
    words = nltk.word_tokenize(inputText)
    for word in words:
        if word in unigrams:
            freq_list[unigrams.index(word)]+=1
    return [(val / (len(words))) for val in freq_list]

def getBigramFeatures(inputText):
    inputText = inputText.lower()
    freq_list = [0]*len(bigrams)
    words = nltk.word_tokenize(inputText)
    for i in range(1,len(words)):
        if (words[i-1],words[i]) in bigrams:
            freq_list[unigrams.index((words[i-1],words[i]))]+=1
    return [(val / (len(words)-1)) for val in freq_list]

def getTrigramFeatures(inputText):
    inputText = inputText.lower()
    freq_list = [0]*len(trigrams)
    words = nltk.word_tokenize(inputText)
    for i in range(1,len(words)):
        if (words[i-2],words[i-1],words[i]) in trigrams:
            freq_list[unigrams.index((words[i-2],words[i-1],words[i]))]+=1
    return [(val / (len(words)-2)) for val in freq_list]



foodir ="amt"
barlist = list()

for root, dirs, files in walk(foodir):
  for f in files:
    if splitext(f)[1].lower() == ".txt":
      barlist.append(join(root, f))

wholeText = ""
for textfile in barlist:
    inputText = io.open(textfile, "r", errors="ignore").readlines()
    inputText = ''.join(str(e) + "\n" for e in inputText)
    wholeText+=inputText
wholeText = wholeText.lower()
global unigrams
global bigrams
global trigrams

unigrams = list(set(nltk.word_tokenize(wholeText)))
bigrams = list(set(list(nltk.bigrams(wholeText.split()))))
trigrams = list(set(list(nltk.trigrams(wholeText.split()))))

# print(len(unigrams))
# print((bigrams))
# print(len(trigrams))
texta = "Hello people of world. How the are you all\nHekki again.\n\nNew Parah graphs started"

