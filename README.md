# ML-Shallow-Twitter-Analysis

Twitter dataset:
The data set includes tweets about the airline industry as well as user experiences.
The only independent variables in the dataset are the tweets and the tweet id, while the dependent variable is the airline sentiment. Positive, negative, and neutral class labels appear in the dependent target column. 
The dataset includes 11858 tweet samples of type text and their appropriate sentiment. 
It is also evident that there is no null value in the dataset.  
There are 7434 negative tweets in train data whereas neutral and positive are only 2510 and 1914 respectively. 
The model train samples are unbalanced in terms of the predictor variables.


Text data cleaning
The first stage in the data cleaning technique is to remove stop words. The sklearn package "ENGLISH STOP WORDS" is used for this purpose. 
The punctuation is then eliminated using the regex functionality. 
The stemming and lemmatization algorithms from nltk are then utilised to improve the data by refining the noisy words
Text data is transformed into numerical data for machine learning algorithms through processes like word tokenization using Keras Tokenizer and subsequently employing the GLOVE model, resulting in a vocabulary size of 12,409 terms.
To prevent the balancing challenge, the separated words are then padded. Padding occurs in such a way that each word has a vector of 100 characters

Modelling
The cleaned data was utilized by SVM, Nave Bayes, and the Random forest classifier, the basic neural network, and the convolution neural network.

Conclusion
I favor SVM with TFID vectorization over other models due to its significantly higher accuracy score (78%) in comparison to other models, including SNN with GLOVE encoding (63%), Naïve Bayes with TFID vectorization (68%), and Random Forest with TFID vectorization (77%). The high accuracy across all models mainly consists of numerous true positive values for the negative class label while performing poorly on other classes, indicating that the model excels at learning from negative data but struggles with neutral and positive values due to data imbalance, where negative samples significantly outnumber the other two classes, causing reduced accuracy across all machine learning models.

