# Malay Tweets Sentiment Analysis using DL4J

This project is an web application utilizing Java Spring Boot and DL4J to analyze the sentiment of real-time tweets streaming from Twitter.

The below image shows the interface of the application. 
The tweets are grouped according sentiment similarity, which are positive, negative and neutral 
(from left to right).

![Project screenshot](src/images/web_ui_01.png?raw=true "Screenshot of the project")


----------

There are 2 models used in this project, which are Word2Vec model and Long Short Term Memory (LSTM) model.

## Word2Vec Model

Word2Vec Model is a neural network model that processes text corpus into sets of vectors.
It is used to find the cosine distance or similarity between words. In the project, 
the Word2Vec architecture used is continuous skip-gram, which it
predicts surrounding window of context words by using the current word. The input
layer reads the words and encoded them separately as nodes. The hidden layer will
do the convergence job while the output layer will produce word vectors that have the 
same size as the input layer.


## LSTM Model
LSTM model is a modified version of recurrent neural networks(RNN) which aims to solve the
vanishing gradient problem as it is easier to remember past data
in memory. In this project, the model applied is a two layer LSTM model, which are the
LSTM layer and RNN output layer. In the LSTM model, the activation function applied
is *tanh* function which is used to determine the level of importance of the values
that pass through the input gate and output gate of each node. In the nodes,
sigmoid function is used to decide which value can pass through input and output gate
or be discarded in forget gate. After the data passing through LSTM layer, 
it will pass through the RNN output layer with *softmax* activation function 
to get the final value.



----------
### Why F1 Score?

F1 Score is a metric that takes both precision and recall into account, so it can be considered that 
the effect of both false positive and false negative are included in this formula. Also, as it is a trade-off
between precision and recall, so F1-score is required to compare performance of different model.

There are other evaluation metrics such as accuracy, precision and recall.
Accuracy is a simple evaluation metrics that can be easily understood but the public
as it just looks for the correct predictions among all cases. However,
accuracy may not be that accurate when the testing data is imbalanced, where the data amount of a class is 
greatly exceeds the data amount of the other classes. This evaluation metric will easily miss out 
the effect of false negative and false positive when the data is imbalanced.

Precision and recall shows the model performance according to false positive and false negative
respectively. Precision looks into the accuracy of positive predictions made while recall looks into
the percentage of positive class that can be correctly predicted. When the effect
of false positive is more significant, we shall select a model based on precision. If we
would like to have less false negative predictions, we should go for recall. Nevertheless, 
F1-score is still a better metrics as in most cases, false positives and false negative are 
both essential to be avoided.

### The Trade Offs & Suggested Improvements

For Word2Vec model, *minWordFrequency* is one of the parameter that will significantly affect the performance.
*minWordFrequency* refers to the minimum number of times a word must appear in the corpus given to Word2Vec model. 
A low *minWordFrequency* may lead to increase in noises while higher *minWordFrequency* may cause some losses
in the data. An increasing of *minWordFrequency* in this project may reduce the over-fitting issue as the text corpus is
considered large, which is 625,528 lines of tweets data. However, the accuracy of the prediction might be decreased in some
cases as some rare words will be filtered out if *minWordFrequency* is increased.

On the other hand, dropout may be applied to the LSTM model to improve the performance of the model as it may 
reduce over-fitting of model towards training data. The dropout ratio must be chosen carefully as it might 
reduce overall accuracy of the model.


### Acknowledgement

Thanks to the contribution of Husein Zolkepli and his team
in collecting and processing Malay-Dataset.

The datasets used in training are acquired from the following Github repository.
https://github.com/huseinzol05/malay-dataset



