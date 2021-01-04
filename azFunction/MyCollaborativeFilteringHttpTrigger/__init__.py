import logging

import azure.functions as func

import numpy as np
import pandas as pd
import pickle
import json
import os


DOT = 'dot'
COSINE = 'cosine'

class TinyModel():
    
    def __init__(self,embeddings):
        self.embeddings = embeddings

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    #data_path = "../data/"
    data_path = "/storage_mount/"
    articles_df = pd.read_csv(data_path+'articles_metadata.csv')

    logging.info("Prepare to unpickle")
    embeddings = pickle.load(open(data_path+"model_collaborative.p","rb"))
    logging.info("Model loaded")
    model = TinyModel(embeddings)

    user_id = int(req.params.get('user_id'))

    recos = user_recommendations(user_id,model,articles_df, measure=COSINE, k=5)

    recos = list(recos)

    return func.HttpResponse(
         str(recos),
         status_code=200
    )



def compute_scores(query_embedding, item_embeddings, measure=DOT):
    """Computes the scores of the candidates given a query.
    Args:
    query_embedding: a vector of shape [k], representing the query embedding.
    item_embeddings: a matrix of shape [N, k], such that row i is the embedding
      of item i.
    measure: a string specifying the similarity measure to be used. Can be
      either DOT or COSINE.
    Returns:
    scores: a vector of shape [N], such that scores[i] is the score of item i.
    """
    u = query_embedding
    V = item_embeddings
    
    if measure == COSINE:
        V = V / np.linalg.norm(V, axis=1, keepdims=True)
        u = u / np.linalg.norm(u)
    
    scores = u.dot(V.T)
    
    return scores

def user_recommendations(user_id,model,articles_df, measure=DOT, exclude_rated=False, k=6):

    scores = compute_scores(
        model.embeddings["user_id"][user_id], model.embeddings["article_id"], measure)
    score_key = measure + ' score'
    
    print(scores.shape)
    #print(interests_df.shape)
    
    df = pd.DataFrame({
        score_key: list(scores),
        'article_id': articles_df['article_id']
    })
    if exclude_rated:
        # remove movies that are already rated
        rated_movies = ratings[ratings.user_id == user_id]["movie_id"].values
        df = df[df.movie_id.apply(lambda movie_id: movie_id not in rated_movies)]
    
    return df.sort_values([score_key], ascending=False).head(k)['article_id'].values