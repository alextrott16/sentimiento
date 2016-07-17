# sentimiento

Exploring sentiment analysis of movie reviews using LSTM RNNs

## Creating data for training
- All the necessary data files are found in sentimiento/data, they can be found at the [Stanford NLP Sentiment Analysis Project page](http://nlp.stanford.edu/sentiment/)
- Sentences must be tokenized (already done) so they can be represented as vector sequences, which requires a dictionary of word vectors
- Currently, this supports vector dictionaries in the GLoVE format, as available [here](http://nlp.stanford.edu/projects/glove/)
- To prepare the data using a particular dictionary, run /data/prepare_data.ipynb after modifying the first 2 lines to use a different glove file / save name.
