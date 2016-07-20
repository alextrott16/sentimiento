# sentimiento

Exploring sentiment analysis of movie reviews using LSTM RNNs

## Creating data for training
- All the necessary data files are found in sentimiento/data, they can be found at the [Stanford NLP Sentiment Analysis Project page](http://nlp.stanford.edu/sentiment/)
- Sentences must be tokenized (already done) so they can be represented as vector sequences, which requires a dictionary of word vectors
- Currently, this supports vector dictionaries in the GLoVE format, as available [here](http://nlp.stanford.edu/projects/glove/)
- To prepare the data using a particular dictionary, run /data/prepare_data.ipynb after modifying the first 2 lines to use a different glove file / save name.


## Making and running networks
All the heavy lifting is done by:
- glover.py (turns data into trainables--see prepare_data.ipynb)
- network_components.py (manages the internals of all of the network components involved, modularized as classes)
- senti_net.py (makes the network components play nice with the project's data format)

For an example of how to use these, study the 'Run project.ipynb' notebook. Better yet, run it!. Currently, it will make a standard and bi-directional LSTM each for both the binary and fine-grained sentiment classification tasks. These 4 examples should make it fairly clear how one is intended to integrate the various modules listed above. Also, they train networks, which is cool.

## To do:
- Add an attentional reader
- Hyperparameter search
- ???
- Profit
