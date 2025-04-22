import nltk
#nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet 
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk import FreqDist
from nltk.corpus import stopwords
import string
from nltk import word_tokenize

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J":wordnet.ADJ,"N":wordnet.NOUN,"V":wordnet.VERB,"R":wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def quitarstopwords_eng(texto):
    ingles = stopwords.words("english")
    texto_limpio = [w.lower() for w in texto if w.lower() not in ingles and w not in string.punctuation
                    and w not in ["'s", '|', '--', "''", "``"] ]
    return texto_limpio

def lematizar(texto):
    texto_lema = [lemmatizer.lemmatize(w,get_wordnet_pos(w))for w in texto]
    return texto_lema
#iniciar lematizador
lemmatizer = WordNetLemmatizer()
corpus=PlaintextCorpusReader(".",'corpussylenguajes.txt') #obtiene texto de archivo
texto_tokenizado = word_tokenize(corpus.raw())#tokenizado del texto
frecuencia = FreqDist(lematizar(quitarstopwords_eng(texto_tokenizado))) #quitando stopwords y lematizando
frecuencia.plot(20, show=True, show="hola")#distribuye frecuencia