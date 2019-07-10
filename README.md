# Named_Entity_Recognition

In this jupyter notebook, I have applied Named Entity Recognition on MIT Movie data. Please find the data here(https://groups.csail.mit.edu/sls/downloads/movie/).
I have trained my nueral network on trivia10k13train.bio	and tested it on trivia10k13test.bio

I have been recently expsosed to the mysterious world of deep learning. LSTM-RNN (Long Short Term memory Recurrent Neural Networks) is of the fascinating model, I recently used for Named Entity Recognition on this MIT movie dataset. In this notebook, I have tried different word embeddings.
First, lets see what is a word embedding. It is a technique in which words or phrases from the vocabulary are mapped to vectors of real numbers. So, we have three options for chosing our embedding matrix.
a) Randomly initialized embedding matrix: In this method, the embedding matrix will be initialized randomly and the model will learn to differentiate the meaning of words just by looking at the data. It means the neural network will itself learn during the execution. 
 initial_embedding_matrix = np.random.randn(vocabulary_size, embedding_dim) / np.sqrt(embedding_dim)
using this initialization on the dataset, I got following preformance-
b) Pre-trained Word Embedding- There are some corpus of pre trained word embeddings. I have used two different word embeddings-
1) Pre-trained word vectors from Google which were trained on a part of Google News dataset (about 100 billion words). The model contains 300-dimensional vectors for 3 million words and phrases. We will use gensim library to unpack and upload the Google vectors.

Get the vectors from the link below(I am using google colab, you can download it to your local drive as well).
!wget -P /root/input/ -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
Import gensim library which we will use for unpacking and uploading the vectors in Word2Vec format. 
import gensim
from gensim import models
wv_embeddings_google = models.KeyedVectors.load_word2vec_format('/root/input/GoogleNews-vectors-negative300.bin.gz', binary=True, limit=500000)
The performance of this model is -

![alt_text](https://github.com/gmt20/Named_Entity_Recognition/blob/master/Random.png)


2) Glove3 Pre-trained word vectors from Stanford NLP which is created using co-occurrence matrix. If you will visit the website, you will see that you can create your own word embeddings using your data(may be relevant word embeddings). But I am using Glove3 common crawler pretrained word vectors.The model contains 300-dimensional vectors of 840 billion (!!) words and phrases. 


Get the vectors from the link below-
!wget http://nlp.stanford.edu/data/glove.840B.300d.zip
Unzip the file-
!unzip glove.840B.300d.zip
Import glove library for python-
! pip install glove_python
import glove

We will only use first 100000 tokens:

with open('glove.840B.300d.txt', 'rt') as in_file:
 with open('glove.840B.300d.out.txt', 'wt') as out_file:
 print("Reducing GloVE file to first 100k words")
 for i, l in enumerate(in_file.readlines()):
 if i>=100000: break
 out_file.write(l)
 
Create embedding matrix:

word_embedding_glove = glove.Glove.load_stanford('glove.840B.300d.out.txt' )
word_embedding_glove.word_vectors.shape
initial_embedding_matrix = word_embedding_glove.word_vectors
The performance of Glove3 Word Embedding used in Bi-LSTM model -

![alt_text](https://github.com/gmt20/Named_Entity_Recognition/blob/master/golve3.png)



Well , we can see that there is no improvement in using word embedding. May be this post can help."TK"
