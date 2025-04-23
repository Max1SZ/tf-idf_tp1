# TF-IDF(Term Frequency -Inverse Document Frequency):
#CANALIZACION (pipeline) 
#Realizar una "canalizacion" o "pipeline" para realizar el siguiente CORPUS CORPUS.Lenguaje.text
#1-Aplicar stopwords
#2-lematizacion
#3-TF-IDF
#4-Mostrar el corpus PREPARADO
#5-Mostrar la MATRIZ  TF-IDF generada
#6-Mostar el vocabulario generado

# Analizar el mismo y redactar un informe con las conclusiones obtenidas
# 1-obtener las jeraquias de 6 palabras mas usadas en todos el Corpus.
# 2-la palabra menos ustilizada.
# 3-las palabras mas repetidas en la misma oracion.
# 4-imprimir el grafico de Distribucion de frecuencia.

# Librerías necesarias
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Corpus a analizar
corpus = [
    "Python is an interpreted and high-level language, while CPlus is a compiled and low-level language. JavaScript runs in web browsers, while Python is used in various applications, including data science and artificial intelligence. JavaScript is dynamically and weakly typed, while Rust is statically typed and ensures greater data security .-",
    "Python and JavaScript are interpreted languages, while Java, CPlus, and Rust require compilation before execution.",
    "JavaScript is widely used in web development, while Go is ideal for servers and cloud applications.",
    "Python is slower than CPlus and Rust due to its interpreted nature."
    "JavaScript has a strong ecosystem with Node.js for backend development, while Python is widely used in data science .-",
    "JavaScript does not require compilation, while CPlus and Rust require code compilation before execution .-",
    "Python and JavaScript have large communities and an extensive number of available libraries.",
    "Python is ideal for beginners, while Rust and CPlus are more suitable for experienced programmers."
]

# Preparación
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Acumular tokens de todo el corpus
tokens_todos = []
tokens_filtrados_todos = []
tokens_lematizados_todos = []

for texto in corpus:
    tokens = word_tokenize(texto.lower())
    tokens_todos.extend(tokens)

    tokens_filtrados = [t for t in tokens if t.isalpha() and t not in stop_words]
    tokens_filtrados_todos.extend(tokens_filtrados)

    lemas = [lemmatizer.lemmatize(t) for t in tokens_filtrados]
    tokens_lematizados_todos.extend(lemas)

# Contar repeticiones en oración seleccionada
oracion = corpus[1]
tokens_oracion = word_tokenize(oracion.lower())
tokens_oracion_filtrados = [t for t in tokens_oracion if t.isalpha() and t not in stop_words]
lemas_oracion = [lemmatizer.lemmatize(t) for t in tokens_oracion_filtrados]

repeticiones = {}
for palabra in lemas_oracion:
    repeticiones[palabra] = repeticiones.get(palabra, 0) + 1

# Contar frecuencia general
frecuencia = {}
for palabra in tokens_lematizados_todos:
    frecuencia[palabra] = frecuencia.get(palabra, 0) + 1

frecuencia_ordenada = sorted(frecuencia.items(), key=lambda x: x[1], reverse=True)
palabras = [p[0] for p in frecuencia_ordenada[:10]]
valores = [p[1] for p in frecuencia_ordenada[:10]]

# Preparar corpus lematizado y limpio
corpus_preparado = []
for texto in corpus:
    tokens = word_tokenize(texto.lower())
    tokens_filtrados = [t for t in tokens if t.isalpha() and t not in stop_words]
    lemas = [lemmatizer.lemmatize(t) for t in tokens_filtrados]
    corpus_preparado.append(" ".join(lemas))

# Generar matriz TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus_preparado)

# Salidas
print("_" * 70)
print("corpus_preparado:")
print(corpus_preparado)

print("_" * 70)
print("\nMatriz TF-IDF:\n")
print(tfidf_matrix.toarray())

print("_" * 70)
print("\nVocabulario generado:\n")
print(vectorizer.get_feature_names_out())

print("_" * 70)
print("texto tokenizado:", tokens_todos)

print("_" * 70)
print("tokens filtrados:", tokens_filtrados_todos)

print("_" * 70)
print("tokens lematizados:", tokens_lematizados_todos)

print("_" * 70)
print("Top 6 palabras más frecuentes:")
for palabra, cantidad in frecuencia_ordenada[:6]:
    print(f"{palabra}: {cantidad}")

print("_" * 70)
print("\npalabra menos utilizadas:")
print(frecuencia_ordenada[-1])

print("\nPalabras repetidas en la oración seleccionada:")
for palabra, cantidad in repeticiones.items():
    print(f"{palabra}: {cantidad}")

# Gráfico de barras
plt.figure(figsize=(10, 5))
plt.bar(palabras, valores, color='skyblue')
plt.title("Top 10 Palabras Más Frecuentes")
plt.xlabel("Palabras")
plt.ylabel("Frecuencia")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
