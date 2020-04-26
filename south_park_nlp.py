#!/usr/bin/env python
# coding: utf-8

# # Exploring Language in South Park
# This project is about exploring speech data from South Park with NLP. We'll look at how to process NLP data, build and interpret NLP pipelines, calculate and intrepret different word association metrics, and assess different pipeline configurations for text classification. 
# 
# ## Processing Speech Data
# Linguistic data is a particularly complex and challenging type of data to mine and analyze. Like other types of data, text documents require processing of raw data into clean, structured features in order to be modeled. However, processing of linguistic data is uniquely tricky for several reasons--two of which being that text is not numeric and that interpreting speech is difficult and subjective. 
# 
# This is why data processing and feature extraction is an especially involved process for NLP. We'll break down NLP processing with first principles to provide a more interpretable understanding. First, let's load our csv-formatted South Park speech data.

# In[1]:


import pandas as pd
df = pd.read_csv('all_seasons.csv',header = 0,skiprows=0)
print(df)


# We can see that our raw data has 70,896 South Park quotes. Each quote has three accompanying characteristics: the season in which it appeared, the episode in which it appeared, and the character who spoke it. In terms of NLP terminology, each quote is considered a document, and each word in a document is considered a term.
# 
# Processing raw text data with NLP typically consists of **tokenization**, followed by **word reduction**, and then concluded with **vectorization**. These operations are performed sequentially, so they represent a **pipeline**. Tokenization is usually the most straightfoward part of such a pipeline. It deconstructs speech into individual words of interest. **Stop words** are the words which will be filtered out during the tokenization process and thus won't be considered in the model. Some common stop words are conjunctions like "or," and "the," which provide little to no meaningful information--especially when isolated from their surrounding context. Here's an example to illustrate tokenization with nltk and how it varies with different stop words. 

# In[2]:


import nltk

document = 'The lazy fox jumped over the log.'
print("Input document: " + str(document))
print()

stop_words = set(['the'])
tokens = nltk.word_tokenize(document.lower())
valid_tokens = [token for token in tokens if token not in stop_words]
print("Tokenization with stop words = {'the'}: " + str(valid_tokens))

stop_words = set(['the','over'])
tokens = nltk.word_tokenize(document.lower())
valid_tokens = [token for token in tokens if token not in stop_words]
print("Tokenization with stop words = {'the','over'}: " + str(valid_tokens))


# You might notice that one of the tokens is punctuation. Whether to consider punctuation or filter it out is another one of many choices you have for NLP processing. I prefer to include punctuation bcause it can indicate emotional context, as well as writing style if the document was written by the speaker directly. However, if filtering out punctuation is of interest to you, here's a simple method for doing so.

# In[3]:


import string

def remove_punctuation(word):
    for char in word:
        if char in string.punctuation:
            word = word.replace(char,'')
    
    return word

print("Tokens before punctuation filter: " + str(valid_tokens))
unpunctuated_tokens = [remove_punctuation(token) for token in valid_tokens if remove_punctuation(token)]
print("Tokens after punctuation filter: " + str(unpunctuated_tokens))


# Word reduction is the simplification of words to their root words. *Why simplify words in such a way?* One source of motivation is to represent different variations of the same word equivalently. Through doing so, we minimize both the amount of considered words and the complexity of our model. Stemming and lemmatization are two different approaches for word reduction. 
# 
# Lemmatization considers the way in which a word is used (its *morphology*) when reducing it, while stemming is restricted to word transformation through simple operations like removing suffixes and prefixes. Lemmatization is qualitatively superior to stemming because it represents words according to their surrounding contexts, but it typically takes more time. 
# 
# Here are some simple stemmer and lemmatizer classes. They both tokenize a document into word tokens, which are then either stemmed or lemmatized into core words. Once the word reduction has completed, its resulting data will be ready for building features.

# In[4]:


from nltk import WordNetLemmatizer as wnl
from nltk.stem import PorterStemmer

class Stemmer(object):
    def __init__(self):
        self.stemmer = PorterStemmer()
    def __call__(self, doc):
        tokens = nltk.word_tokenize(doc.lower())
        stems = [self.stemmer.stem(token) for token in tokens]
        
        return stems

class Lemmatizer(object):
    def __init__(self):
        self.wnl = wnl()
    def __call__(self, doc):
        tokens = nltk.word_tokenize(doc.lower())
        lemmas = [self.wnl.lemmatize(token) for token in tokens]
        
        return lemmas
                    


# Now we can finish the processing phase with vectorization: the mapping of pre-processed tokens into a numerical representation. Basically, we need to express our tokens in terms of informative numerical features so that we can analyze and model them. The two most common vectorization methods are **count vectorization** and **tfidf**. We'll discuss tfidf shortly, but for now, we'll stick to count vectorization. Count vectorization simply measures the amount of times each token appears in each document. These measurements can be used to build an (m x n) **term document matrix** with some key properties: 
# * It has $m$ rows, where $m$ = # of documents (or quotes).
# * It has $n$ columns, where $n$ = # of tokens.
# * Its $(i,j)$th element is the frequency with which $token_{i}$ appears in $document_{j}$.
# 
# Using nltk and sklearn, we'll process our speech data with a count vectorizer to begin some exploration and analysis of the data. Sklearn vectorizers are capable of all pre-processing tasks: tokenization, stemming and/or lemmatization, and vectorization. 

# In[5]:


import numpy as np
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer, TfidfTransformer
from sklearn.feature_extraction import text

# Download some random nltk data we need.
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

# Create a count vectorizer for tokenizing, lemmatizing and vectorizing our raw quotes. #min_df = minimum token count for the token to be included. 
stop_words = stopwords.words('english')
count_vectorizer = CountVectorizer(min_df=2,tokenizer=Lemmatizer(),stop_words=stop_words)
quotes = df['Line']
document_term_matrix = count_vectorizer.fit_transform(quotes)

# This mapping allows for easy word retrieval from a column in the count matrix. For example, if we want to know
# the word associated with column 3, column_to_word[3] will do the trick. 
column_to_word = count_vectorizer.get_feature_names()
num_columns = len(column_to_word)
word_to_column = {column_to_word[column]:column for column in range(num_columns)}


# Finding least and most popular tokens is a useful way to start exploring linguistic data and to validate that your processing system is working well. Note that for a term document matrix, a column's sum represents a word's total frequency across all documents. So we can find a word's frequency by calculating its column sum.

# In[6]:


# Get the top <num_words> most popular words from a document term matrix and its vectorizer.
def get_top_words(num_words,document_term_matrix,count_vectorizer):
    word_totals = np.sum(document_term_matrix,axis=0)
    descending_columns = np.argsort(-word_totals,axis=1)
    descending_columns = descending_columns.tolist()[0]
    max_columns = descending_columns[0:num_words]
    
    top_counts = [word_totals[0,col] for col in max_columns]
    top_words = [column_to_word[column] for column in max_columns]
    top_words = {"Word":top_words}
    word_counts = pd.DataFrame(top_words)
    word_counts.insert(1,"Count",top_counts)
    
    return word_counts

num_top_words = 10
top_word_counts = get_top_words(num_top_words,document_term_matrix,count_vectorizer)
print(top_word_counts)


# There's an inconsistency problem in our processing pipeline. Inconsistency in this context means that pre-processing stop words (tokenization,lemmatization,stemming...) produces some valid words not contained in stop words. Additionally, the most popular words above fail the intuition test. More specifically, some of the popular tokens, like "n't", are word fragments that have no independent meaning.
# 
# Rather than experiment with different stop words and lemmatization/stemming methods (of which there are many), we'll try a simpler trick. Unless the problem demands for specific, custom pre-processing, returning to default NLP configurations in Python is likely an effective solution. Also, sklearn's countvectorizer is fairly robust and can be a good default option to fall back on.

# In[7]:


# Rebuild document term matrix using default count vectorizer. Then recalculate relevant objects.
count_vectorizer = CountVectorizer(min_df=2,stop_words=stop_words)
document_term_matrix = count_vectorizer.fit_transform(quotes)
column_to_word = count_vectorizer.get_feature_names()
num_columns = len(column_to_word)
word_to_column = {column_to_word[column]:column for column in range(num_columns)}

top_word_counts = get_top_words(num_top_words,document_term_matrix,count_vectorizer)
print(top_word_counts)


# Not only is the pre-processing now consistent, but it looks much better intuitively. Most popular tokens are now valid words that make sense given our knowledge of South Park. We'll now implement a similar algorithm for finding rarest words.

# In[8]:


# Same idea as above, except that we're now finding words whose respective column sums are minimal. 
# Since a word's column sum gives that word's count, these words correspond to the rarest words.
def get_rarest_words(num_words,document_term_matrix,count_vectorizer):
    word_totals = np.sum(document_term_matrix,axis=0)
    ascending_columns = np.argsort(word_totals,axis=1)
    ascending_columns = ascending_columns.tolist()[0]
    min_columns = ascending_columns[0:num_words]
    
    rarest_words = [column_to_word[column] for column in min_columns]
    rarest_counts = [word_totals[0,column] for column in min_columns]
    rarest_words = {"Word":rarest_words}
    word_counts = pd.DataFrame(rarest_words)
    word_counts.insert(1,"Count",rarest_counts)
    
    return word_counts

rarest_word_counts = get_rarest_words(1000,document_term_matrix,count_vectorizer)
print(rarest_word_counts)


# If you're curious about how to thoroughly test and validate these rarest words, that's a great curiosity to have. Unfortunately, this is a feature engineering result for which proving some notion of accuracy is difficult. Instead, we can use our intuition about English and South Park, as well as the most popular word counts, to suggest that the rarest words and their respective counts seem reasonable. Intuition is a helpful and sometimes neccessary tool for exploratory analysis.

# Note that word frequency is a poor metric for measuring word importance and word associations. You can learn about some of these flaws in more depth below.
# * Some of the most universal english words are structural parts of speech (ie, conjunctions and prepositions) that have little to no individual meaning or value. Because their purpose is to provide relational context between words in the same sentence, a lot of the original info these words had (which wasn't much to begin with) is lost when they're isolated. Linguistic meaning is far from linear. Ie, words, sentences, paragraphs,... collectively represent something very different from their individual components. Context usually matters quite a bit. 
# 
# 
# * Consider word frequency in the context of a random variable's distribution. The more likely a random variable is to have certain outcomes, the less informative those outcomes are when they occur. Why? Because they were anticipated more strongly than the less likely outcomes. For ex, if you know that Jack yells "Wolf," with probability 1, how much information do you gain when that happens? Zero. Conversely, if Jack never says "Wolf" but then suddenly shouts it, that is a very surprising event. Surprise, in this context, embodies information. 

# ## Tfidf Vectorization
# Now we'll restructure our processed features with a **tfidf** vectorizer. Tfidf is a NLP metric for measuring information between a term and a document. It scales a term's frequency in a document ("tf") by an inverse document frequency factor ("idf"). Idf is defined as the inverse ratio of a term's associated documents count to the total document count:
# 
# $$\text{idf(term)} = \frac{\text{number of documents}}{\text{number of documents containing term}}.$$
# 
# So the tfidf score is given by 
# 
# $$\text{tfidf(term,document)} = (\text{number of term instances in document})\frac{\text{number of documents}}{\text{number of documents containing term}}.$$
# 
# Idf grows as a term's number of documents declines, thereby increasing that term's tfidf with its associated documents (for unassociated documents, tf=0 so the tf*idf product will be zero). 
# 
# Conversely, idf decays when a term's amount of associated documents increases. Intuition behind this idea is that there is a stronger, more informative association between a word and a document if the word is not likely to be found in other documents. As previously discussed, words that frequently occur in different documents are inherently less informative about the documents in which they appear.   

# In[9]:


vectorizer = TfidfVectorizer(min_df=2,stop_words=stop_words)
quotes = df['Line']
lemmatized_quotes = vectorizer.fit_transform(quotes)
tf = TfidfTransformer()
tf_mat = tf.fit_transform(lemmatized_quotes)
print(tf_mat.shape)


# As part of exploratory analysis, let's build a character frequency distribution to see how well each character is represented in our data. For each character, we'll calculate its quote quantity. 

# In[10]:


from sklearn.preprocessing import LabelEncoder

# Create a feature encoder. Feature encoders are used to map feature indices to their respective features and vice-versa.
# In this application, we're now exploring a different feature: the quote's speaking character. The label encoder
# will encode the character column in 'df' with integer indices--allowing for character information to be represented in our analysis.

encoder = LabelEncoder()
characters = df["Character"]
encoder.fit(characters)
characters_encoded = encoder.transform(characters)
characters = encoder.classes_
encodings = encoder.transform(characters)
decoder = dict(zip(encodings,characters))

# Dictionaries below are intended for accessing a character's list of rows (=character's quotes), and its frequency = # of quotes.
character_to_rows = {char:[] for char in characters}
character_to_total = {char:0 for char in characters}
for row,character_encoded in enumerate(characters_encoded):
    character = decoder[character_encoded]
    character_to_rows[character].append(row)
    character_to_total[character] += 1

# Determine most popular characters through sorting character frequencies. 
popular_characters = sorted(character_to_total,key=character_to_total.get,reverse=True)
popular_totals = [character_to_total[character] for character in popular_characters]
characters_data = {"Character":popular_characters}
df_characters = pd.DataFrame(characters_data)
df_characters.insert(1,"# of Quotes",popular_totals)

print("Most Popular Characters")
print(df_characters)


# Now we'll produce some visuals for both character frequency and average character polarity. Polarity captures the emotional positivity of a document through averaging the polarity of its individual words. Its range of values is [-1,1], where -1 and 1 correspond to negative and positive, respectively. 

# In[11]:


from textblob import TextBlob

def character_polarity(character,character_to_rows,quotes):
    character_rows = character_to_rows[character]
    average_polarity = 0.0
    for row in character_rows:
        average_polarity += TextBlob(quotes[row]).sentiment.polarity
    
    average_polarity = average_polarity / float(len(character_rows))
    
    return average_polarity
        
top_characters = df_characters.loc[df_characters['# of Quotes'] >= 500]
top_characters_polarity = [character_polarity(character,character_to_rows,quotes) for character in top_characters['Character']]
top_characters.insert(2,'Polarity',top_characters_polarity)
top_characters.plot.bar(x='Character',y='# of Quotes')
top_characters.plot.bar(x='Character',y='Polarity',color='red')


# The character frequency bar plot validates data quality in the sense that the most represented characters are those which are the most represented in the show. Character polarity is a bit more interesting--particularly with regards to some noticeable outliers. Cartman is one of the most notoriously evil characters in animated television, yet his average polarity is positive and somewhat normal relative to the other main characters. More generally, average polarity is positive for every character, despite the fact that these characters range from moderately cynical to completely egregious (except for Butters!). This is mild evidence of a slight positive bias in TextBlob's polarity measure. 

# ## Visualizing Linguistic Data
# 
# We'll be mining our data for insightful words, so we should develop a tool for pleasantly visualizing words. One such tool is a **wordcloud**, which displays words in an artistically generated map. Thankfully, there's a great module in Python for this that you can read more about here: https://amueller.github.io/word_cloud/index.html#. We'll use this module to create a method for building a wordcloud from a list of tokens. 

# In[12]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Initialize mask for building wordclouds. Code from an example on wordcloud's homepage was used as a template. 
x, y = np.ogrid[:200, :200]
mask = (x - 100) ** 2 + (y - 100) ** 2 > 100 ** 2
mask = 500* mask.astype(int)
def build_wordcloud(words,title):
    wc = WordCloud(repeat=True, mask=mask)
    wordcloud_text = ' '.join(words)
    wc.generate(wordcloud_text)
    plt.axis("off")
    plt.imshow(wc, interpolation="bilinear")
    plt.title(title)
    plt.show()

example_words = ['check','out','this','wordcloud']
example_title = 'Example'
build_wordcloud(example_words,example_title)


# ## Word Character Relationships 
# When exploring structured data, finding variable associations can provide valuable insight about how different qualities of your data are interdependent. In our case, we'd like to find characteristic words for South Park characters.  A **characteristic word** for a character is one that is strongly associated to that character. Thus, when characteristic words are found in a quote, they indicate that the character is more likely to have spoken the quote.
# 
# What's a good way to measure variable association? There are lots of different ways to approach this. Intuitively, people tend to start exploring this idea under the impression that the more often $X$ occurs with $Y$, the stronger their dependency. But this is not neccessairly the case. To understand why, we'll build wordclouds representing the twenty most frequent words for some key characters. 

# In[13]:


key_characters = ['Cartman', 'Kyle', 'Stan', 'Randy', 'Kenny', 'Chef', 'Wendy',
                  'Butters', 'Sharon', 'Craig', 'Towelie', 'Mr. Garrison', 'Mr. Slave', 
                  'Token', 'Jimbo', 'Ned', 'Timmy', 'Jimmy', 'Mr. Mackey', 'Satan', 
                  'Terrance', 'Phillip', 'Big Gay Al', 'Tweek', 'Grandpa', 'Ike']

num_top_words = 20
# For each key character, sum its rows in the document term matrix to produce its word totals.
# Find the maximal columns in this word totals vector, and convert their indices to words to generate top words.
for character in key_characters:
    character_rows = character_to_rows[character]
    row_sums = np.sum(document_term_matrix[character_rows][:],axis=0)
    sorted_columns = np.argsort(-row_sums,axis=1)
    sorted_columns = sorted_columns.tolist()[0]
    top_words = [column_to_word[column] for column in sorted_columns[0:num_top_words]]
    build_wordcloud(top_words,character)


# Note that there is considerable overlap between the top words of each character. Also, the words have a strong tendency to be generic and neither interesting nor indicative of any particular character. No consideration of a word's prior likelihoods--how likely the word is to be used by any character--is present. This allows for generic words commonly used by many characters to be found, despite the fact that they don't indicate any particular character very strongly. We need a more nuanced approach.

# **Logistic modeling** is a standard supervised learning approach for classification and regression. It's especially valuable in that it has a very clear, probabilistic interpretation. For example, log regression can be used in learning chess to predict who will win given the current board. After training on thousands of games, the logistic output from a board will represent either the probability of white winning (regression), or the player most likely to win (classification). Note that this probabilistic interpretation quality is fairly unique. Other models, such as SVM and decision trees, cannot be interpreted in probabilistic frameworks, and rely on linear algebra for relevance. 
# 
# This concept extends to multinomial problems, where the predicted variable has more than 2 possible outcomes. Since we're concerned with predicting a quote's character from more than two possible characters, a multinomial model is appropriate. Let's train a multinomial logistic model using sklearn to predict speaking character from South Park quotes. Then, we'll use these logistic coefficients to determine characteristic words for some popular characters. We could use sklearn's feature importance directly, but instead we'll develop a quick algorithm to illustrate the underlying linear algebra and logic.
# 
# Heads up: A lot of curse words are about to show up. 

# In[14]:


from sklearn.linear_model import SGDClassifier

# weights_matrix has the coefficients derived from the trained logistic model. Its (i,j)th coefficient 
# represents the weight word_j has on character_i's likelihood. For example, the word 'mr. slave' likely has a large coefficient for Mr. Garrison.
log_classifier = SGDClassifier('log').fit(tf_mat,characters_encoded)
weights_matrix = log_classifier.coef_

# For each key character, first find its row in the weights matrix. Then find the indices of this row at which weight is maximized. These
# indices correspond to that key character's most heavily weighted words. 
for character in key_characters:
    row = encoder.transform([character])
    word_weights = weights_matrix[row,:]
    sorted_columns = np.argsort(-word_weights,axis=1)
    sorted_columns = sorted_columns.tolist()[0]
    top_words = [column_to_word[column] for column in sorted_columns[0:num_top_words]]
    build_wordcloud(top_words,character)
    


# **Lift** is a popular association rule used in data mining. Lift measures the increased likelihood of one variable when you know the outcome of another variable. We can express the lift between a word and a character as
# 
# $$
# \text{lift(word,character)} = \frac{\text{prob(word,character)}}{\text{prob(word)prob(character)}}= \frac{\text{prob(word|character)}}{\text{prob(word)}}.$$
# 
# Note how closely connected lift is to Bayes rule for conditional probability:
# 
# $$\text{prob(character|word)} = \frac{\text{prob(word|character)prob(character)}}{\text{prob(word)}}.$$
# 
# In fact, the only difference between lift is that conditional probability is multiplied by the class likelihood term $ prob(character).$ But for a fixed character, this term will be constant, so ranking a character's words according to their lift is equivalent to ranking them by their conditional probabilities. Characteristic words under this metric are those which maximize the probability that a certain character has spoken them. 

# In[15]:


num_quotes,num_terms = document_term_matrix.shape 
# Vector containing the amount of quotes in which each word appears. Used for calculating prior probabilities for words.
num_word_quotes = np.count_nonzero(document_term_matrix.toarray(),axis=0)
for character in key_characters:
    character_rows = character_to_rows[character]
    num_character_quotes = float(len(character_rows))

    num_shared_quotes = np.count_nonzero(document_term_matrix[character_rows][:].toarray(),axis=0)
    word_likelihoods = num_shared_quotes/num_character_quotes
    word_priors = num_word_quotes/float(num_quotes)
    
    word_lifts = word_likelihoods/word_priors
    sorted_columns = np.argsort(-word_lifts)
    sorted_columns = sorted_columns.tolist()
    top_words = [column_to_word[column] for column in sorted_columns[0:num_top_words]]
    build_wordcloud(top_words,character)


# If you're feeling suspicious about the above word clouds, you're not mistaken. Every character's word cloud contains noticeably many bizarre words that should be considered as noise. When mining data for associations, outliers may emerge. This is often caused by having no constraints in place to filter out noisy associations from small amounts of data. For example, the word "prissy," doesn't even occur 5 times, so any association signal it produces shouldn't be trusted. Implementing a data quantity threshold should reduce this noise pretty well. 

# In[22]:


for character in key_characters:
    character_rows = character_to_rows[character]
    num_character_quotes = float(len(character_rows))

    num_shared_quotes = np.count_nonzero(document_term_matrix[character_rows][:].toarray(),axis=0)
    word_likelihoods = num_shared_quotes/num_character_quotes
    word_priors = num_word_quotes/float(num_quotes)
    
    word_lifts = word_likelihoods/word_priors
    sorted_columns = np.argsort(-word_lifts)
    sorted_columns = sorted_columns.tolist()
    sorted_words = [column_to_word[column] for column in sorted_columns]
    top_words = [word for word in sorted_words[0:1000] if num_word_quotes[word_to_column[word]] >= 15][0:10]
    build_wordcloud(top_words,character)


# Now we'll invert the problem and find indicative characters for a set of target words. Multinomial **Naive Bayes** can be implemented to calculate likelihoods of each target word with respect to all characters. Likelihood represents the probability of a word occuring in a quote given that a certain character is speaking the quote. So for each key word's column, we'll find the rows at which that column's likelihoods are maximized. The characters associated with these rows are the most likely characters to speak the key word.

# In[23]:


from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB().fit(tf_mat,characters_encoded)
likelihoods = classifier.feature_log_prob_
likelihoods = np.exp(likelihoods)
character_priors = np.exp(classifier.class_log_prior_)
feature_characters = encoder.classes_
key_words = ['children','god','timmy','mom','towel','mkay','hamburgers','bastard','eric','authority']
for word in key_words:
    column = word_to_column[word]
    word_likelihoods = likelihoods[:,column]
    sorted_rows = np.argsort(-word_likelihoods)
    sorted_rows = np.squeeze(np.asarray(sorted_rows))
    top_rows = sorted_rows[0:10]
    top_characters = [decoder[row] for row in top_rows]
    build_wordcloud(top_characters,word)


# ## Predictive Modeling
# Finally, we'll classify quotes into their speaking characters by processing them through a full NLP classification pipeline. Quotes will be processed into lemmatized tokens, then vectorized into features and subsequently classified. Model training and testing entails some minor extra steps since feature processing now requires its own training, in addition to model training. We should first train our vectorizer on some training data, then train our classifier on the vectorized transformation of this training data. 
# 
# To get a sense of how different aspects of the pipeline affect classificiation performance, we'll train and test pipelines with different design configurations. Tfidf vs. count vectorization will be compared, as well as three different classification models: logistic, SVM, and multinomial Naive Bayes. 

# In[24]:


from sklearn.model_selection import train_test_split

tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words)
vectorizer = CountVectorizer(stop_words=stop_words)

df_train,df_test = train_test_split(df,test_size=.3)
quotes_train,quotes_test = df_train['Line'],df_test['Line']
characters_train,characters_test = df_train['Character'],df_test['Character']
X_train = vectorizer.fit_transform(quotes_train)
X_test = vectorizer.transform(quotes_test)
Y_train,Y_test = df_train['Character'],df_test['Character']


# In[25]:


count_scores = {}

sgd_classifier = SGDClassifier('log').fit(X_train,Y_train)
score = sgd_classifier.score(X_test,Y_test)
count_scores['log'] = score

svm_classifier = SGDClassifier().fit(X_train,Y_train)
svm_classifier.fit(X_train,Y_train)
score = svm_classifier.score(X_test,Y_test)
count_scores['svm'] = score

NB_classifier = MultinomialNB().fit(X_train,Y_train)
score = NB_classifier.score(X_train,Y_train)
score = NB_classifier.score(X_test,Y_test)
count_scores['bayes'] = score

print("Count vectorization scores: " + str(count_scores))


# In[26]:


tfidf_scores = {}
df_train,df_test = train_test_split(df,test_size=.3)
quotes_train,quotes_test = df_train['Line'],df_test['Line']
characters_train,characters_test = df_train['Character'],df_test['Character']


# In[27]:


tfidf_scores = {}

X_train = tfidf_vectorizer.fit_transform(quotes_train)
X_test = tfidf_vectorizer.transform(quotes_test)
Y_train,Y_test = df_train['Character'],df_test['Character']

sgd_classifier = SGDClassifier('log').fit(X_train,Y_train)
score = sgd_classifier.score(X_test,Y_test)
tfidf_scores['log'] = score

sgd_classifier = SGDClassifier('hinge').fit(X_train,Y_train)
score = sgd_classifier.score(X_test,Y_test)
tfidf_scores['svm'] = score

NB_classifier = MultinomialNB().fit(X_train,Y_train)
score = NB_classifier.score(X_test,Y_test)
tfidf_scores['bayes'] = score

print("Tfidf vectorization scores: " + str(tfidf_scores))


# Classification accuracy is expected to be lower for NLP problems. But consistently getting accuracies around 20% is sufficiently low to warrant a failure and attempt at revision. Observe that variation of accuracy across different vectorizations and models has actually been very low. A valuable insight data scientists often learn through gaining experience is that *features and input data often have a stronger impact on model performance than model choice itself*. 
# 
# In light of this insight, *does our data have any fundamental flaws?* To be more precise, *does the data possess any qualities that could significantly hamper classification?* Well, we're trying to predict speaking characters, and there are hundreds to thousands of South Park characters. It's apparent through revisting some of the previous character analysis that we have about 4000 distinct characters in our processed, vectorized data. This would be OK if each character had a substantial amount of accompanying quotes, but such is not the case. Take a look at the following code snippet. 

# In[28]:


character_frequency_median = np.median(df_characters['# of Quotes'])
print("Median amount of character quotes: " + str(character_frequency_median))


# At least half of all characters have no more than 2 quotes. This is far less input data than needed for even the simplest of machine learning problems and models. Excluding characters with insufficient data, as determined by a minimum frequency threshold, is likely to result in significant improvement. 

# In[29]:


# Filter out data from characters with < 500 quotes before model training and testing. 
min_frequency = 500
model_characters = df_characters.loc[df_characters['# of Quotes'] > min_frequency]
model_df = df.loc[df['Character'].isin(model_characters['Character'].values)]
model_characters = set(model_df['Character'].values)
print("Included characters: " + str(model_characters))

df_train,df_test = train_test_split(model_df,test_size=.3)
quotes_train,quotes_test = df_train['Line'],df_test['Line']
characters_train,characters_test = df_train['Character'],df_test['Character']
X_train = vectorizer.fit_transform(quotes_train)
X_test = vectorizer.transform(quotes_test)
Y_train,Y_test = df_train['Character'],df_test['Character']


# In[30]:


count_scores = {}

sgd_classifier = SGDClassifier('log').fit(X_train,Y_train)
score = sgd_classifier.score(X_test,Y_test)
count_scores['log'] = score

svm_classifier = SGDClassifier().fit(X_train,Y_train)
svm_classifier.fit(X_train,Y_train)
score = svm_classifier.score(X_test,Y_test)
count_scores['svm'] = score

NB_classifier = MultinomialNB().fit(X_train,Y_train)
score = NB_classifier.score(X_test,Y_test)
count_scores['bayes'] = score

print("Count vectorization scores: " + str(count_scores))


# In[31]:


tfidf_scores = {}

X_train = tfidf_vectorizer.fit_transform(quotes_train)
X_test = tfidf_vectorizer.transform(quotes_test)
Y_train,Y_test = df_train['Character'],df_test['Character']

sgd_classifier = SGDClassifier('log').fit(X_train,Y_train)
score = sgd_classifier.score(X_test,Y_test)
tfidf_scores['log'] = score

sgd_classifier = SGDClassifier('hinge').fit(X_train,Y_train)
score = sgd_classifier.score(X_test,Y_test)
tfidf_scores['svm'] = score

NB_classifier = MultinomialNB().fit(X_train,Y_train)
score = NB_classifier.score(X_test,Y_test)
tfidf_scores['bayes'] = score

print("Tfidf vectorization scores: " + str(tfidf_scores))


# Filtering out training and testing data of infrequent classes resulted in almost doubled accuracy. Interestingly, variation in accuracy between tfidf and count vectorization remains very low. Similarly, accuracy variation across the three models is again somewhat trivial. Although the fact that Naive Bayes consistently had the highest error may be partially connected to its independence assumption. A word's surrounding words provide significant value to its meaning. This suggests that class likelihood from one word is dependent on information about other words, which conflicts with the feature independence assumption made by Naive Bayes. 

# In[ ]:




