# Import Modules
# pip install newsapi-python
from newsapi import NewsApiClient
import random
from datetime import datetime, timedelta

# Import Scikit Modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

# API - 61d7b1d842414eb1a94ff375b20d24ff
prev_date = datetime.today() - timedelta(days=30)
next_date = datetime.today() - timedelta(days=0)
p_date = str(prev_date.year)+'-'+'0'+str(prev_date.month)+'-'+'0'+str(prev_date.day)
c_date = str(next_date.year)+'-'+'0'+str(next_date.month)+'-'+'0'+str(next_date.day)

# Create a Get News Method
newsapi = NewsApiClient(api_key='61d7b1d842414eb1a94ff375b20d24ff')
def getNews(sourceId):
    newses = newsapi.get_everything(sources=sourceId,
                                    domains='bbc.co.uk, techcrunch.com',
                                    from_param=p_date,
                                    to=c_date,
                                    language='en',
                                    sort_by='relevancy',
                                    page=2)
    newsData = []
    for news in newses['articles']:
        list = [random.randint(0, 1000), news['title'],news['content'], 'REAL']
        newsData.append(list)
    return newsData

# Get News Sources
sources = newsapi.get_sources()
sourceList = []
for source in sources['sources']:
    sourceList.append(source['id'])
del sourceList[10:len(sourceList)]
print('New Sources: ', sourceList)

# Get News using Multiple Sources
dataList = []
for sourceId in sourceList:
    newses = getNews(sourceId)
    dataList = dataList + newses

print('Total News: ', len(dataList))

# Create a DataFrame of News
import pandas as pd
df = pd.DataFrame.from_records(dataList)
df.columns = ['','title','text','label']

# Load and Concat the DataFrame
trainData = pd.read_csv('../dataset/news.csv')
trainData.columns = ['', 'title', 'text', 'label']
data = [trainData, df]
df = pd.concat(data)

# Split the Training and Testing Data
training_x, testing_x, training_y, testing_y = train_test_split(
    df['text'], df.label, test_size=0.3, random_state=100)

# Feature Selection
count_vectorizer = CountVectorizer(stop_words='english', max_df=0.7)
feature_train = count_vectorizer.fit_transform(training_x)
feature_test = count_vectorizer.transform(testing_x)

# Initialise and Apply the Classifier
classifier = PassiveAggressiveClassifier(max_iter=50)
classifier.fit(feature_train, training_y)

# Test the classifier
prediction = classifier.predict(feature_test)
score = accuracy_score(testing_y, prediction)

print("Accuracy: ", score*100)

# Load the Test Data
test_data = pd.read_csv('../dataset/test_data.csv')
test_labels = test_data.label
test_data.head()

# Select Features and Get Predictions
test_data_feature = count_vectorizer.transform(test_data['text'])
prediction = classifier.predict(test_data_feature)

# Evaluate the Predictions
for i in range(len(test_labels)):
    print(test_labels[i], prediction[i])

score = accuracy_score(test_labels, prediction)
print("Accuracy: ", score*100, "%")