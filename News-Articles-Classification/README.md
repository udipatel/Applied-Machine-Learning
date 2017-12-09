### **Project : News Article classfication: using Bag-of-words approach**

## **Introduction**

The goal of this project is to develop a model that will classify news articles based on topics like Business, Tech, Sports, Politics and Entertainment. The classifier will be built using Bag-of-words model. The Document-Term matrix, a high dimensional sparse matrix is built from the raw text files of the news articles. The Raw text files will be transformed into a sparse Matrix using advanced Natural Language Techniques like Vectorization, tf-idf transformation and removing “English” stop words. The focus is on choosing the best classifier from a variety of Machine learning algorithms and using cross validation and optimization techniques for best results.

## **Data**

Dataset used for classification is BBC news dataset. It Consists of 2225 documents from the BBC news website corresponding to stories in five topical areas from 2004-2005. Natural Classes: 5 (**business, entertainment, politics, sport, tech**)

*Courtsey :
D. Greene and P. Cunningham. "Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering", Proc. ICML 2006. All rights, including copyright, in the content of the original articles are owned by the BBC.*

## **Code Example**

The program imports the BBC datsets using the datasets.load_files python library

`bbcdataset = datasets.load_files(bbcfilespath,encoding = 'utf-8',decode_error = 'ignore')`
The load_files class imports all the files from the path specified and the returned dataset `bbcdataset`is a scikit-learn “bunch”: a simple holder object with fields that can be both accessed as python dict keys or object attributes for convenience, for instance the target_names holds the list of the requested category names: \n

```python
>>> bbcdataset.target_names`
['business', 'entertainment', 'politics', 'sport', 'tech']
```

The `data` attribute contains the text content of the files loaded, the `target` attribute contains the class which is indexed as integers and `filenames` attribute contains the names of the files loaded.Next we convert this text files into a vector by using `CountVectorizer` and `tf-idf transformation`. 

Below is a sample code that uses a Decision Tree to fit the data and predict the news articles class from the test set. As can be seen the accuracy obtained is approx 79%

```python
>>>from sklearn import tree
>>>tree_clf = tree.DecisionTreeClassifier()
>>>tree_clf = tree_clf.fit(X_train,y_train)
>>>tree_predict = tree_clf.predict(X_test) 
>>>np.mean(y_test == tree_predict)
0.79325842696629212
```

A general apprach is to apply various machine learrning algortihms to find out the best ones and then tune the parameters of the best alogorithms to further improve the accuracy of the classifier.






=================================================================================
Instruction to run IPYNB file.
=================================================================================

1. Import AML-News-classification.IPYNB in Jupyter notebook
2. Execute cells as required or Run all cells from the file options.
3. Results can be directly viewed in file RESULTS-AML-News-classification.html    
