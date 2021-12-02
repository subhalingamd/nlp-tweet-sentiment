# Tweet Sentiment Mining

<!-- MarkdownTOC -->

1. [Motivation](#motivation)
1. [Problem Statement](#problem-statement)
1. [Dataset](#dataset)
	1. [Train Data](#train-data)
	1. [Test Data](#test-data)
1. [Methodology](#methodology)
1. [Running the code](#running-the-code)
	1. [Directory Structure](#directory-structure)
	1. [Requirements](#requirements)
	1. [Training](#training)
	1. [Testing](#testing)
1. [Results](#results)
1. [Credits](#credits)

<!-- /MarkdownTOC -->



<a id="motivation"></a>
## Motivation
The motivation of this assignment is to get practice with text categorization using classical Machine Learning algorithms.

<a id="problem-statement"></a>
## Problem Statement
The goal of the assignment is to build a sentiment categorization system for tweets. The input of the code will be a set of tweets and the output will be a prediction for each tweet – *positive* or *negative*.

<a id="dataset"></a>
## Dataset

<a id="train-data"></a>
### Train Data
**For training, we use the *(processed version of)* [Sentiment140](https://www.kaggle.com/kazanova/sentiment140) dataset.**
The format of each line in the training dataset is `<“label”, “tweet”>`. It has a total of *1.6 million* tweets. A label of *0 means negative sentiment* and a label of *4 means positive sentiment*.

A note from the creaters: 
> "Our approach was unique because our training data was automatically created, as opposed to having humans manual annotate tweets. In our approach, we assume that any tweet with positive emoticons, like :), were positive, and tweets with negative emoticons, like :(, were negative. We used the Twitter Search API to collect these tweets by using keyword search"

**Training data can be found at [`_data/training.zip`](_data/training.zip) *(to be unzipped)*.**

<a id="test-data"></a>
### Test Data
The final program will take input a set of tweets. Each line will have one tweet (without double quotes).
The program will output predictions (0 or 4) one per line – matching one prediction per tweet.

**Test data can be found at [`_data/test/`](_data/test/).**


<a id="methodology"></a>
## Methodology
Each tweet undergoes several pre-processing steps. Upon several experiments, we settled with the strategy as in [`preprocessing.py`](preprocessing.py), namely, replacement of hashtags / mentions / urls / punctuations / emoticons with placeholders, tweet normalization on words where letters are repeated for intensity & internet slang
dictionary, contractions removal, reversing polarity in case of negations (until next punctuations), stemming (Porter Stemmer), to list a few.

Post processing, we have settled with [`TfidfVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) for feature extraction and [`LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) for prediction.
The pipeline can be found [here](https://github.com/subhalingamd/nlp-tweet-sentiment/blob/0ff069dfdba4efe687e90e9e9ecf885f8c12d7f4/main.py#L69).

<a id="running-the-code"></a>
## Running the code
<a id="directory-structure"></a>
### Directory Structure
The main code (for both training and testing) is [`main.py`](main.py). The program uses [`constants.py`](constants.py),  [`preprocessing.py`](preprocessing.py) and [`utils.py`](utils.py) *(files names are self-explanatory)*. [`slang.txt`](slang.txt) contains list of internet slangs (one on each line) used for pre-processing.

[`stats.py`](stats.py) is made for EDA (not used by the main code).

[`_data/`](_data/) contains the dataset.

<a id="requirements"></a>
### Requirements
The following Python packages are required:
```
 pandas==1.1.3
 numpy==1.19.5
 scikit-learn==0.24.2
 scipy==1.5.3
 nltk==3.5
```
**Make sure `punkt` is downloaded in `nltk` to run `PorterStemmer`.**

<a id="training"></a>
### Training
```bash
bash run-train.sh <data_directory> <model_directory>
```
The script reads training data from `<data_directory>/training.csv` and saves the model after training at `<model_directory>`.

For format of input data, see section [3.i](#train-data).

<a id="testing"></a>
### Testing
```bash
bash run-test.sh <model_directory> <input_file_path> <output_file_path>
```
The script loads the trained model from `<model_directory>`, scores text in `<input_file_path>` and writes prediction to `<output_file_path>`. 

For format of input/output data, see section [3.ii](#test-data).



<a id="results"></a>
## Results
Accuracy on test set: `0.7833`


<a id="credits"></a>
## Credits
* Internet slangs dictionary: [[Link1]](https://en.wiktionary.org/wiki/Appendix:English_internet_slang) [[Link2]](https://github.com/Deffro/text-preprocessing-techniques)

* Emoticons dictionary: [[Link]](https://en.wikipedia.org/wiki/List_of_emoticons)

* Punctuations dictionary:: [[Link]](http://en.wikipedia.org/wiki/Punctuation)

* Contractions dictionary [[Link]](https://github.com/Deffro/text-preprocessing-techniques)

* [Forum] Sentiwordnet/POS tagging might not work well for tweets: [[Link]](https://stackoverflow.com/questions/38263039/sentiwordnet-scoring-with-python)



----
*This README uses texts from the assignment problem document provided in the course.*
