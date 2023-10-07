#!/usr/bin/env python
# coding: utf-8

# # Gender Classification using NLP Techniques

# In[1]:


import numpy as np # linear algebra
import pandas as pd


# In[2]:


data = pd.read_csv("gender-classifier-DFE-791531.csv",encoding='latin-1')


# In[3]:


data


# In[4]:


data.info()


# In[5]:


data = pd.concat([data.gender,data.description],axis=1)


# In[6]:


data.head()


# DATA CLEANING

# In[7]:


data.dropna(axis = 0, inplace = True)


# In[8]:


data.gender = [1 if each == "female" else 0 for each in data.gender]


# In[9]:


data.gender.unique()


# In[10]:


import re
first_description = data.description[4]
first_description


# In[11]:


description = re.sub("[^a-zA-Z]"," ",first_description)
description


# In[12]:


description = description.lower()
description


# In[13]:


import nltk # natural language tool kit
nltk.download("stopwords")
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')


# In[14]:


description = nltk.word_tokenize(description)


# In[15]:


description = [ word for word in description if not word in set(stopwords.words("english"))]


# In[16]:


description


# In[17]:


import nltk as nlp
lemma = nlp.WordNetLemmatizer()
description = [ lemma.lemmatize(word) for word in description ]
print(description)


# In[18]:


description = " ".join(description)
print(description)


# In[19]:


description_list = []
for description in data.description:
    description = re.sub("[^a-zA-Z]"," ",description)
    description = description.lower()
    description = nltk.word_tokenize(description)
    description = [ word for word in description if not word in set(stopwords.words("english"))]
    lemma = nlp.WordNetLemmatizer()
    description = [ lemma.lemmatize(word) for word in description ]
    description = " ".join(description)
    description_list.append(description)


# In[20]:


from sklearn.feature_extraction.text import CountVectorizer
max_features = 500
count_vectorizer = CountVectorizer(max_features=max_features,stop_words="english",)
sparce_matrix = count_vectorizer.fit_transform(description_list).toarray()
print("Most common {} words : {}".format(max_features,count_vectorizer.get_feature_names()))


# Perform EDA to understand data better

# In[21]:


y = data.gender.values # male ofr female classes
x = sparce_matrix


# In[22]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.1, random_state = 42)


# In[23]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)


# In[24]:


y_pred = nb.predict(x_test)

print("accuracy: ",nb.score(y_pred.reshape(-1,1),y_test))


# DATA VISUALIZATION

# In[25]:


from matplotlib import pyplot
pd.value_counts(data['gender']).plot(kind="pie", startangle = 90, shadow = True, radius = 1.2, autopct = '%1.1f%%')
pyplot.suptitle('Categorical Plotting of gender')
pyplot.xlabel("")
pyplot.ylabel("")
pyplot.show()


# In[26]:


data


# In[27]:


from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
Vectorize = TfidfVectorizer(stop_words='english', token_pattern=r'\w{1,}', max_features=35000)
X = Vectorize.fit_transform(data["description"])
y = data.gender 
le = preprocessing.LabelEncoder()
y = le.fit_transform(y.values)


# In[28]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
models = []
models.append(("k Nearest Neighbor",KNeighborsClassifier(n_neighbors=5)))
models.append(("Random Forest",RandomForestClassifier(n_estimators=100, max_depth=2)))
models.append(("Logistic Regression",LogisticRegression()))
models.append(("Naive Bayes",MultinomialNB()))
models.append(("Decision Tree",tree.DecisionTreeClassifier()))


# In[29]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
def cross_validate_evaluate(algorithm):
    
    # Build model
    clf = algorithm
    # Not Need
    def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
    def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
    def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
    def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]
    scoring = {'tp': make_scorer(tp), 'tn': make_scorer(tn),
               'fp': make_scorer(fp), 'fn': make_scorer(fn)}
    
    # Stratified k-Fold 
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    # Evaluation Indicator
    score_funcs = [
        'accuracy',
        'precision',
        'recall',
        'f1',
    ]
    # Cross Validation 
    scores = cross_validate(clf, X, y, cv=skf, scoring=score_funcs)
    print('accuracy:', scores['test_accuracy'].mean())
    print('precision:', scores['test_precision'].mean())
    print('recall:', scores['test_recall'].mean())
    print('f1:', scores['test_f1'].mean())
    
    #return scores

#if __name__ == '__main__':
#    main()
models_index=0
name_index=1
for models_index in range(len(models)):
    print("-----------"+str(models[models_index][name_index-1])+"-----------")
    cross_validate_evaluate(models[models_index][name_index])


# In[30]:


def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, x_train, y_train, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)


# In[31]:


from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(x_train, y_train)
rmse_cv(model_lasso).mean()


# In[32]:


index_list = []
for i in range(len(model_lasso.coef_)):
    index_list.append(i)
coef_dict = dict(zip(index_list,model_lasso.coef_))
tfidf_name = Vectorize.get_feature_names()
for i in range(len(coef_dict)):
    if coef_dict.get(i) > abs(0):
        print(tfidf_name[i] + " : " + str(coef_dict[i]))
        Vectorize.get_feature_names()


# In[33]:


print(model_lasso.intercept_)


# Build a neural network using Sklearn 

# In[34]:


from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

per = Perceptron(random_state = 42,max_iter = 20,tol = 0.01)
per.fit(x_train,y_train)

yhat_train_per = per.predict(x_train)
yhat_test_per = per.predict(x_test)

print(f"Perceptron:Accuracy in train is  0.2%f"%(accuracy_score(y_train,yhat_train_per)))
print(f"Perceptron:Accuracy in test is  0.2%f"%(accuracy_score(y_test,yhat_test_per)))

mlp = MLPClassifier(max_iter = 50,solver = 'sgd',verbose = 10,random_state = 42,learning_rate_init = 0.01,hidden_layer_sizes = (100,100,2))
mlp.fit(x_train,y_train)

yhat_train_mlp = mlp.predict(x_train)
yhat_test_mlp = mlp.predict(x_test)

print(f"Multilayer Perceptron:Accuracy in train is  0.2%f"%(accuracy_score(y_train,yhat_train_mlp)))
print(f"Multilayer Perceptron:Accuracy in test is  0.2%f"%(accuracy_score(y_test,yhat_test_mlp)))
                


# In[35]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)
y_pred = nb.predict(x_test)

print("accuracy: ",nb.score(y_pred.reshape(-1,1),y_test))

