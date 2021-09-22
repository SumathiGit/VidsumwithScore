import matplotlib.pyplot as plt
import torch
import numpy as np 
import argparse
import pickle 
import os
from torchvision import transforms 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN

from PIL import Image
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize,word_tokenize
from gensim.models import Word2Vec
import re


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
wordlemmatizer = nltk.stem.WordNetLemmatizer()

class Capt():

    def __init__(self) -> None:
        pass

    def load_image(self, image_path, transform=None):
        for i in image_path:
            self.image = Image.open(image_path).convert('RGB')
            self.image = self.image.resize([224, 224], Image.LANCZOS)
            # print("image_path -->" , image_path)
            # print("image" , image)
            if transform is not None:
                image = transform(self.image).unsqueeze(0)
            
            return image


    def lemmatize_words(self, words): # used
            lemmatized_words = []
            for word in words:
                tempList = []
                for word2 in word:
                    tempList.append(wordlemmatizer.lemmatize(word2))
                lemmatized_words.append(tempList)
            return lemmatized_words

    def uniqueWord(self, w): # used 
        w2=[]
        for word in w:
            tempList=[]
            for word2 in word:
                if tempList.count(word2)<1:
                        tempList.append(word2)
            w2.append(tempList)
        return w2

    def remove_special_characters(self,text): # used
        regex = r'[^a-zA-Z0-9\s]'
        self.text = re.sub(regex,'',text)
        return text

    def removeStopWord(self, word_text):  # used
        filtered_sentence = [] 
        stop_words = set(stopwords.words('english'))   
        for w in word_text:
            tempList=[]
            for x in w:
                if x.lower() not in stop_words: 
                    tempList.append(x)
            filtered_sentence.append(tempList)
        return filtered_sentence   

    def meanOfWord(self, model, sentence): # used
    #     posValue=nltk.pos_tag(sentence)
        posList=['CD']
        nounList=['NN','NNP','NNS','NNPS']
        value=[]
        count=0
        noun=0
        for word in sentence:
            a=model.wv.similar_by_word(word)
            temp=[]
            for w in a:
                temp.append(w[1])
            posValue=nltk.pos_tag([word])
    #         print(posValue)
            wordScore=np.mean(temp)
            if posValue[0][1] in posList:
                count=count+1
            elif posValue[0][1] in nounList:
                noun=noun + .25
            value.append(wordScore)
        return np.mean(value)+count+noun

    

    def checkNum(self, s):
        l= ['1','2','3','4','5','6','7','8','9','0']
        check =False

        for i in s:
            if i in l:
                check = True
                break
        if check == True:
            return 1
        else:
            return 0

    # captions = []
    def caption(self, args1):
        # Image preprocessing
        transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.485, 0.456, 0.406), 
                                (0.229, 0.224, 0.225))])
        
        # Load vocabulary wrapper
        with open(args1.vocab_path, 'rb') as f:
            vocab = pickle.load(f)

        # Build models
        encoder = EncoderCNN(args1.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
        decoder = DecoderRNN(args1.embed_size, args1.hidden_size, len(vocab), args1.num_layers)
        encoder = encoder.to(device)
        decoder = decoder.to(device)

        # Load the trained model parameters
        encoder.load_state_dict(torch.load(args1.encoder_path))
        decoder.load_state_dict(torch.load(args1.decoder_path))

        # Prepare an image
        directory = 'data'
        img_path_list = []
        img_filename = []
        for filename in os.listdir(directory):
            # print(filename)
            img_filename.append(filename)
            f = os.path.join(directory, filename)
            # checking if it is a file
            if os.path.isfile(f):
                print(f)
                # print(type)
                img_path_list.append(f)
                # img_path_list.sort()
                img_path_list.sort(key=lambda f: int(re.sub('\D', '', f)))
        print(img_path_list)
        args1.image = img_path_list
        # print(args.image, "**") #list of Frames path
        # print(args.image[0])
        # print(type(args.image[0]))

        # captions = []
        for i,j in enumerate(args1.image):
            image = self.load_image(args1.image[i], transform)
            image_tensor = image.to(device)
            
            # Generate an caption from the image
            feature = encoder(image_tensor)
            sampled_ids = decoder.sample(feature)
            sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
            
            # Convert word_ids to words
            sampled_caption = []
            for word_id in sampled_ids:
                word = vocab.idx2word[word_id]
                sampled_caption.append(word)
                if word == '<end>':
                    break
            mysentence = ' '.join(sampled_caption)
            # string1 = 
            # captions.append(mysentence)
            # Print out the image and the generated caption
            print("*****************************************")
            print (mysentence)
            str1 = ['<start>','<end>', "''"]
            for x in str1:
                mysentence= mysentence.replace(x, '')
            print(mysentence)
            # captions.append(mysentence)
            print("*****************************************")
            print(j)



            Stopwords = set(stopwords.words('english'))
            wordlemmatizer = WordNetLemmatizer()
            text=mysentence
            sentences = sent_tokenize(text) # 1: sent tokenize
            text_noSpecial_character = self.remove_special_characters(str(text)) # 2: remove special character:
            word_text = [[text_noSpecial_character for text_noSpecial_character in sentences.split()] for sentences in sentences] # 3: word token
            stop_text= self.removeStopWord(word_text) # 4: remove stop words
            unique_text= self.uniqueWord(stop_text)   # 5: remove duplicate words
            lemma_text = self.lemmatize_words(unique_text) # 6: lemmatization

            model = Word2Vec(lemma_text, min_count=1,sg=1)

            score=[]
            for index, sentence in enumerate(lemma_text):
                i = lemma_text.index(sentence)
                meanScore= self.meanOfWord(model,sentence)
            #         print(str(labels[index])+ ":"+ str(sentence)+ str(meanScore) )
                # temp = [i,meanScore]
                temp = meanScore
                score.append(temp)
            #     print(meanOfWord(model,sentence))
            print(score)



            




parser = argparse.ArgumentParser()
parser.add_argument('-j', '--image', type=str,default="./data", help='input image for generating caption')
parser.add_argument('--encoder_path', type=str, default='models/encoder-5-3000.pkl', help='path for trained encoder')
parser.add_argument('--decoder_path', type=str, default='models/decoder-5-3000.pkl', help='path for trained decoder')
parser.add_argument('--vocab_path', type=str, default='vocabdata/vocab.pkl', help='path for vocabulary wrapper')

# Model parameters (should be same as paramters in train.py)
parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
args1 = parser.parse_args()


capscore = Capt()
capscore.caption(args1)
