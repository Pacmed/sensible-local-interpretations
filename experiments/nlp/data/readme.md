# Datasets
- data taken from https://github.com/AcademiaSinicaNLPLab/sentiment_dataset

**Data** | Classes | Average sentence length | Dataset size | Vocab size | Number of words present in word2vec | Test size
--- | --- | --- | --- | --- | --- | ---
SST1 | 5 | 18 | 11855 | 17836 | 16262 | 2210
SST2 | 2 | 19 | 9613 | 16185 | 14838 | 1821

The following datasets are included in this directory:
  * **SST-1**: Stanford Sentiment Treebank - an extension of MR but with train/dev/test splits provided and fine-grained labels (very positive, positive, neutral, negative, very negative), re-labeled by Socher et al. (2013). [Link](http://nlp.stanford.edu/sentiment/)

    Note that data is actually provided at the phrase-level and hence we train the model on both phrases and sentences but only score on sentences at test time, as in Socher et al. (2013), Kalchbrenner et al. (2014), and Le and Mikolov (2014). Thus the training set is an order of magnitude larger than listed in the above table.
  * **SST-2** Same as SST-1 but with neutral reviews removed and binary labels.

## Data files

Dataset | Files
--- | ---
SST-1 | stsa.fine.\* (use phrases for train)
SST-2 | stsa.binary.\* (use phrases for train)
