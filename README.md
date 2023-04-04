# Fake News Detection Using Scikit-Learn

This project aims to classify news articles as real or fake using a machine learning algorithm. The algorithm used is a Passive Aggressive Classifier, which is a type of online learning algorithm that is used for binary classification.

## Installation
The following packages are required to run this project:

* newsapi-python
* pandas
* scikit-learn

To install these packages, run the following command:

`pip install newsapi-python pandas scikit-learn`

## Data
The training data used to train the classifier is taken from [Kaggle](https://www.kaggle.com/c/fake-news/data).

## Step 1: Import Necessary Modules
The following modules are required to interact with the News API and create the model:

* NewsApiClient from newsapi
* random

## Step 2: Get News Data from the News API
To interact with the News API, an API key is required. The API key can be obtained by registering at https://newsapi.org/. After obtaining the API key, follow the steps below:

1. Call the `NewsApiClient()` method and pass the API key to this method.
2. Create a method to get the news data from the API using the `get_everything()` method.
3. Pass the required parameters to the `get_everything()` method:
   * sources
   * domains
   * from_param
   * to
   * language
   * sort_by
   * page
4. After getting the results from the API, pass the results to an array and return that array.

## Step 3: Get News Sources
The News API has over 3000 authenticated news sources. To get news from these sources, follow the steps below:

1. Get all the sources from the News API using the `get_sources()` method.
2. Add the ID of each source to a list.
3. Truncate the list to a size of 10, and get news from those sources using the `del` keyword.

## Step 4: Create a DataFrame Using News List
1. Use a loop to iterate through all the sources from the `sourceList` and use the `getNews()` method to get news from the sources.
2. Add all the returned news to a list.
3. Use the from_records() method from `pandas.DataFrame` to create a new DataFrame using the list.
4. Add new column headings to the DataFrame using the `dataframe.columns` attribute.

## Step 5: Create a DataFrame of Fake and Real News
1. Load the data from a `.csv` file available in the same directory with the name of the `news.csv` file using the `read_csv()` method from `pandas`. 
2. Add the column headings to the DataFrame. 
3. Use the `concat()` method from pandas to concat both `DataFrames`.

## Step 6: Train the Model
To create the training model, the following modules are required:

* `train_test_split` from `sklearn.model_selection`
* `CountVectorizer` from `sklearn.feature_extraction.text`
* `PassiveAggressiveClassifier` from `sklearn.linear_model`
* `accuracy_score` from `sklearn.metrics`

Follow the steps below to train the model:

1. Split the training and testing data from the DataFrame using the `train_test_split()` method.
2. Use 70% of the data for training and 30% for testing.
3. Pass the combination of `title`, `text`, and news labels to the `*arrays` parameter of the `train_test_split()` method.
4. Use `CountVectorizer` to create a matrix of token count from the text document.
5. Create a `PassiveAggressiveClassifier` model to classify real news from fake news.
6. Test the model using the test data and calculate the model's accuracy using the `accuracy_score` method.

By following these steps, we can create a machine learning model to detect fake news from real news.

## Credits
This project was created by [OpenAI](https://openai.com/) and is based on the tutorial available on [DataCamp](https://www.datacamp.com/tutorial/scikit-learn-fake-news).