import pandas as pd

data = pd.read_csv("Option A - training.csv")

#add variables
data = data.assign(cacl = data.current_assets / data.current_liabilities)
data = data.assign(nisa = data.net_income / data.sales)
data = data.assign(tatl = data.total_assets / data.liabilities_total)
data = data.assign(wcta = (data.current_assets + data.inventory - data.current_liabilities)/data.total_assets)

#description
columns = ["cacl","nisa","tatl","wcta"]
data[columns].describe().T[['count','mean', 'std', 'min', 'max']]
data.groupby("failed")[columns].mean().T

#correlation
data.corr(method="pearson")
correlation = data.corr(method="pearson")

variables = ["cacl","nisa","tatl","wcta"]

data[variables].corr(method="pearson")

correlation = data[variables].corr(method="pearson")

#ttests
import researchpy as rp
failed_cacl = data[data["failed"] == 1]["cacl"]
non_failed_cacl = data[data["failed"] == 0]["cacl"]

rp.ttest(failed_cacl, non_failed_cacl)

descriptives, results = rp.ttest(failed_cacl, non_failed_cacl, equal_variances= False)

print(results)

failed_nisa = data[data["failed"] == 1]["nisa"]
non_failed_nisa = data[data["failed"] == 0]["nisa"]

rp.ttest(failed_nisa, non_failed_nisa)

descriptives, results = rp.ttest(failed_nisa, non_failed_nisa, equal_variances= False)

print(results)

failed_tatl = data[data["failed"] == 1]["tatl"]
non_failed_tatl = data[data["failed"] == 0]["tatl"]

rp.ttest(failed_tatl, non_failed_tatl)

descriptives, results = rp.ttest(failed_tatl, non_failed_tatl, equal_variances= False)

print(results)

failed_wcta = data[data["failed"] == 1]["wcta"]
non_failed_wcta = data[data["failed"] == 0]["wcta"]

rp.ttest(failed_wcta, non_failed_wcta)

descriptives, results = rp.ttest(failed_wcta, non_failed_wcta, equal_variances= False)

print(results)


 
#sentiment analysis     
wordlist = pd.read_csv("Option A - Sentiment_list_2018.csv")
wordlist["Positive"] = wordlist["Positive"].str.lower()
wordlist["Negative"] = wordlist["Negative"].str.lower()

positive = wordlist["Positive"].dropna().tolist()
negative = wordlist["Negative"].dropna().tolist()
all_words = positive + negative


#import fitz
#pdf = fitz.open("MCPIQ.pdf")
#text = ""
#for page in pdf:
    #text += page.getText().lower().strip()
    
#querywords = text.split()
#resultwords  = [word for word in querywords if word.lower() in all_words]

#positive_count={}
#for x in positive:
    #positive_count[x]=resultwords.count(x)

#negative_count={}
#for x in negative:
    #negative_count[x]=resultwords.count(x)
    
#positive_count = dict(sorted(positive_count.items() , reverse=True, key=lambda x: x[1]))
#negative_count = dict(sorted(negative_count.items() , reverse=True, key=lambda x: x[1]))


import matplotlib.pyplot as plt

n=20

first_n_pairs = {k: positive_count[k] for k in list(positive_count)[:n]}
fig = plt.figure(figsize=(14, 4),dpi=100)
plt.bar(list(first_n_pairs.keys()), first_n_pairs.values(), color='g')
plt.xlabel('Word', fontsize=10)
plt.ylabel('Frequency', fontsize=10)
plt.title(f"Frequency of {n} most common positive words")
plt.xticks(rotation=90)
plt.show()


first_n_pairs = {k: negative_count[k] for k in list(negative_count)[:n]}
fig = plt.figure(figsize=(14, 4),dpi=100)
plt.bar(list(first_n_pairs.keys()), first_n_pairs.values(), color='r')
plt.xlabel('Word', fontsize=10)
plt.ylabel('Frequency', fontsize=10)
plt.title(f"Frequency of {n} most common negative words")
plt.xticks(rotation=90)
plt.show()


#collect sentiment scores
def create_sentiment(pdf_file_name):
    import fitz
    pdf = fitz.open(pdf_file_name)
    text = ""
    for page in pdf:
        text += page.getText().lower().strip()

    querywords = text.split()
    resultwords = [word for word in querywords if word.lower() in all_words]

    positive_count = {}
    for x in positive:
        positive_count[x] = resultwords.count(x)

    negative_count = {}
    for x in negative:
        negative_count[x] = resultwords.count(x)

    positive_count = dict(sorted(positive_count.items(), reverse=True, key=lambda x: x[1]))
    negative_count = dict(sorted(negative_count.items(), reverse=True, key=lambda x: x[1]))

    totals = {"positive": sum(positive_count.values()), "negative": sum(negative_count.values())}

    # sentiment score
    sentiment = (totals["positive"] - totals["negative"]) / (totals["positive"] + totals["negative"])

    return sentiment


if __name__ == '__main__':
    import os
    pdf_file_path = "/Users/apple/.spyder-py3/Training1"
    for root, dirs, files in os.walk(pdf_file_path):
        for pdf_file in files:
            path = os.path.join(root, pdf_file)
            print(path)
            res_sentiment = create_sentiment(path)
            print(res_sentiment)




data = pd.read_excel("training_all.xlsx", "training_all")


#description
columns = ["Sentiment score"]
data[columns].describe().T[['count','mean', 'std', 'min', 'max']]
data.groupby("failed")[columns].mean().T

#correlation
data.corr(method="pearson")
correlation = data.corr(method="pearson")

variables = ["cacl","nisa","tatl","wcta","Sentiment score"]

data[variables].corr(method="pearson")

correlation = data[variables].corr(method="pearson")

#ttests
import researchpy as rp
failed_Sentiment = data[data["failed"] == 1]["Sentiment score"]
non_failed_Sentiment = data[data["failed"] == 0]["Sentiment score"]

rp.ttest(failed_Sentiment, non_failed_Sentiment)

descriptives, results = rp.ttest(failed_Sentiment, non_failed_Sentiment, equal_variances= False)

print(results)



#wordcloud
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
import fitz

directory = "Training1"
text = ""

for filename in os.listdir(directory):
    path = os.path.join(directory, filename)
    pdf = fitz.open(path)

    for page in pdf:
        text += page.getText().lower().strip()
        

wordcloud = WordCloud(max_font_size=100, max_words=100,
                      background_color="black").generate(text)

fig = plt.figure(figsize=(10, 10),dpi=100)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()



#logit model
import pandas as pd
import statsmodels.api as sm
import researchpy as rp

y = data["failed"]
X = data[["cacl","nisa","tatl","wcta","Sentiment score"]]

X = sm.add_constant(X)
model = sm.Logit(y, X).fit()
print(model.summary())

#Predict values
predicted = model.predict()
data["Predicted"] = predicted
data.groupby("failed")["Predicted"].mean()


#ttest the difference
data_failed = data[data["failed"] == 1]["Predicted"]
data_clean = data[data["failed"] == 0]["Predicted"]
rp.ttest(data_failed, data_clean)


#ROC

from sklearn.metrics import roc_auc_score as auc

auc(y, predicted)


from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

fpr, tpr, _ = roc_curve(y, predicted) # get the tpr and fpr

lineweight = 2

plt.figure(figsize = (6,6), dpi=100)
plt.plot(fpr, tpr, color='darkorange', lw=lineweight, label='ROC curve (area = %0.2f)' % auc(y, predicted))
plt.plot([0, 1], [0, 1], color='navy', lw=lineweight, linestyle='--')


plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve')
plt.legend(loc="lower right")
plt.show()


#neural network
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score as auc


import tensorflow


from keras.models import Sequential
from keras.layers import Dense

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics as stats



#standardize data
variables = ["cacl","nisa","tatl","wcta","Sentiment score"]
data_to_standardize = data[variables]
standardized = preprocessing.scale(data_to_standardize)
data[variables] = standardized


X = data[["cacl", "nisa","tatl","wcta","Sentiment score"]]
y = data["failed"]
y.sum() # 27 defaulters


# create training and testing samples

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.25)

y_train.sum() #17
y_test.sum() #10


# train the model - keras 

# define the  model
model = Sequential()

model.add(Dense(8, input_dim=X_train.shape[1], activation='sigmoid', kernel_initializer='he_normal'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=2)


# evaluate the keras model - note the accuracy measure is not the same as ROC accuracy
_, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy: %.2f' % (accuracy*100))



#plot roc

predictions = model.predict(X_test)
fpr, tpr, _ = roc_curve(y_test, predictions)

lineweight = 2

plt.figure(figsize = (6,6), dpi=100)
plt.plot(fpr, tpr, color='darkorange', lw=lineweight, label='ROC curve (area = %0.2f)' % auc(y_test, predictions))
plt.plot([0, 1], [0, 1], color='navy', lw=lineweight, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve')
plt.legend(loc="lower right")
plt.show()

auc(y_test, predictions)
