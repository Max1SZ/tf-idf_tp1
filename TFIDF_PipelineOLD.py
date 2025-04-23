import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet, stopwords
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk import FreqDist
import string
from nltk import word_tokenize
import sklearn 
nltk.download('punkt_tab') #por alguna razon sin descargar el punkt_tab no funcionaba
nltk.download('averaged_perceptron_tagger_eng')

from sklearn.feature_extraction.text import TfidfVectorizer
from tabulate import tabulate #Descarga esta libreria en el cmd/bash asi luce lindo el print

#corpus=PlaintextCorpusReader(".",'corpussylenguajes.txt') #obtiene texto de archivo
corpus  =[
"Python is an interpreted and high-level language, while CPlus is a compiled and low-level language .-",
"JavaScript runs in web browsers, while Python is used in various applications, including data science and artificial intelligence.",
"JavaScript is dynamically and weakly typed, while Rust is statically typed and ensures greater data security .-",
"Python and JavaScript are interpreted languages, while Java, CPlus, and Rust require compilation before execution.",
"JavaScript is widely used in web development, while Go is ideal for servers and cloud applications.",
"Python is slower than CPlus and Rust due to its interpreted nature."
"JavaScript has a strong ecosystem with Node.js for backend development, while Python is widely used in data science .-",
"JavaScript does not require compilation, while CPlus and Rust require code compilation before execution .-",
"Python and JavaScript have large communities and an extensive number of available libraries.",
"Python is ideal for beginners, while Rust and CPlus are more suitable for experienced programmers."
]
#corpus=("CorpusLenguajes.txt") #obtiene texto de archivo
# Tokenizar (separa en palabras y pasamos a minusculas)
texto = ''.join(corpus)
tokens = word_tokenize(corpus.lower())
#wordnet
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J":wordnet.ADJ,"N":wordnet.NOUN,"V":wordnet.VERB,"R":wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

#def quitarstopwords_eng(texto):
   # ingles = stopwords.words("english")
    #texto_limpio = [w.lower() for w in texto if w.lower() not in ingles and w not in string.punctuation
    #                and w not in ["'s", '|', '--','-','.' "''", "``","lematizar","(",")","quitarStopwords_eng","word_tokenize"] ]
    #return texto_limpio


# Elimino stopwords y signos de puntuacion.
stop_words = set(stopwords.words('english')) # 
tokens_filtrados = [palabra for palabra in tokens if palabra in tokens if palabra.isalpha() and palabra not in stop_words]


'''def lematizar(texto):
    texto_lema = [lemmatizer.lemmatize(w,get_wordnet_pos(w))for w in texto]
    return texto_lema'''
#iniciar lematizador

lemmatizer = WordNetLemmatizer()
tokens_lematizados = [lemmatizer.lemmatize(palabra) for palabra in tokens_filtrados]

#texto_tokenizado = word_tokenize(corpus.raw())#tokenizado del texto

# Se prepara el corpus limpio
corpus_preparado =[] #Guarda el texto procesado

for texto in corpus:
    tokens = word_tokenize(texto.lower())  # Tokenizar y pasar a minúsculas
    tokens_filtrados = [word for word in tokens if word.isalpha() and word not in stop_words]  #  Filtrar
    tokens_lematizados = [lemmatizer.lemmatize(word) for word in tokens_filtrados]  #  Lematizar
    frase_limpia = " ".join(tokens_lematizados)  # Une en una oración
    corpus_preparado.append(frase_limpia)  # Guarda el resultado
    


#generar matriz
vectorizer = TfidfVectorizer() #Transforma el texto en numeros segun la frecuencia e importancia de cada palabra.
tfidf_matrix = vectorizer.fit_transform(corpus_preparado) #Analiza el texto y calcula los valores.

#Resultados:
print("_"*70)
print("corpus_preparado:", corpus_preparado)

# MOSTRAR LA MATRIZ
print("_"*70)
print("\nMatriz TF-IDF:\n")
print(tfidf_matrix.toarray()) #Muestra los resultados como matriz(filas=oraciones, columnas=palabras)
print("_"*70)

#Muestra el vocabulario generado
print("\nVocabulario generado:\n")
print(vectorizer.get_feature_names_out())#Muestra una lista de todas las palabras(vocabulario) que analizo.
print("_"*70)

#Lematizacion + stopwords aplicadas en el texto tokenizado
print("_"*70)
print("texto lematizado:")
print("_"*70)
print(tabulate(tokens_lematizados))
#frecuencia = FreqDist(tokens_filtrados) #quitando stopwords y lematizando
#frecuencia.plot(20, show=True)#distribuye frecuencia
