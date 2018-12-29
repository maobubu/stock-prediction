# Implementaion of Leveraging Financial News for Stock Trend Prediction with Attention-Based Recurrent Neural Network
**"Leveraging Financial News for Stock Trend Prediction with Attention-Based Recurrent Neural Network"**
Huicheng Liu. _Preprint arxiv_
https://arxiv.org/abs/1811.06173

I won't write everthing in detail here so it is likely that you will face some error when runnning the scripits below (usually something wrong with path, and the commented codes). Please contace me if you want to do the entire process from scratch.

1. Download the meta data from https://drive.google.com/drive/folders/0B3C8GEFwm08QY3AySmE2Z1daaUE

2. Data preprocessing
```
cd stuff
```

Don't forget to change the paths in the scripts listed below.

Download the historical data for the S&P 500 index from yahoo finance and

Use
```
python3 split\ data.py
```
to get the labels.

Use
```
python3 extrac_whole.py
```
to extract the titles, abstracts and articles from the raw data.

Use
```
python3 read_j_news.py
```
to preprocess the news.

Use
```
python3 filter_news.py
```
to further preprocess the news and split them so they are reading for the models input.

3. Build the model.

The core codes are in the scripts/ directory.
```
the LSTM folder stores the script that helps me achieve the result wrote in the paper.
the CLSTM adds CNN before the prediction layer.
the Cha_LSTM adds character level embedding
the SLSTM uses the contents from the article which is much more complicate.
the WLSTM uses bag of words instead of LSTM
ELSTM adds elmo to the code but are almost impossible to train because it needs powerful GPU. Besides, it doesn't give us any performance improvment.
```
To run the scripts, you will have to change the path in train.py and also need Glove which can be download here http://nlp.stanford.edu/data/glove.840B.300d.zip

the script can be run with
```
./train.sh
```

The result will be shown in "log_result.txt" and "log.txt". You can see an example run from
```
65.53/LSTM
```

I really don't have much time to arrange all the codes and make them more clear for the readers. Please contact me if you need a "ready to run version".
