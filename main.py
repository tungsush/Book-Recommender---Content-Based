import numpy as np
import pandas as pd
from numpy import load
from flask import Flask, render_template, request
# libraries for making count matrix and similarity matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# define a function that creates similarity matrix
# if it doesn't exist
def create_sim():
    data = pd.read_csv('data.csv')
    # creating a count matrix
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    # creating a similarity score matrix
    sim = cosine_similarity(count_matrix)
    #sim = load('similarity_matrix.npy')
    return data, sim


# defining a function that recommends 10 most similar books
def rcmd(b):
    b = b.lower()
    # check if data and sim are already assigned
    try:
        data.head()
        sim.shape
    except:
        data, sim = create_sim()
    # check if the movie is in our database or not
    if b not in data['book_title'].unique():
        return('This book is not in our database.\nPlease check if you spelled it correct.')
    else:
        # getting the index of the book in the dataframe
        i = data.loc[data['book_title']== b].index[0]

        # fetching the row containing similarity scores of the book
        # from similarity matrix and enumerate it
        lst = list(enumerate(sim[i]))

        # sorting this list in decreasing order based on the similarity score
        lst = sorted(lst, key = lambda x:x[1], reverse=True)

        # taking top 1- book scores
        # not taking the first index since it is the same book
        lst = lst[1:11]

        # making an empty list that will containg all 10 book recommendations
        l = []
        for i in range(len(lst)):
            a = lst[i][0]
            l.append(data['book_title'][a])
        return l

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/recommend")
def recommend():
    book = request.args.get('book')
    r = rcmd(book)
    book = book.upper()
    if type(r) == type('string'):
        return render_template('recommend.html', book=book, r=r, t='s')
    else:
        return render_template('recommend.html', book=book, r=r, t='l')



if __name__ == '__main__':
    app.run()
