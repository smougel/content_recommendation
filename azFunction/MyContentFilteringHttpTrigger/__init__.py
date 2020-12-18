import logging

import azure.functions as func

import numpy as np
import pandas as pd
import pickle
import json
import os

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    
    
    #data_path = "../data/"
    data_path = "/storage_mount/"
    
    articles_df = pd.read_csv(data_path+'articles_metadata.csv')
    articles_emb = pickle.load(open(data_path+"articles_embeddings.pickle","rb"))
    clicks_by_hour_df = pd.DataFrame()
    for i in range(385):
        index = str(i).zfill(3)
        clicks_df = pd.read_csv(data_path+'clicks/clicks_hour_'+index+'.csv')
        clicks_by_hour_df = clicks_by_hour_df.append(clicks_df)
    
    user_id = int(req.params.get('user_id'))
    
    recos = get_recommendations(user_id,articles_df,clicks_by_hour_df,articles_emb)

    return func.HttpResponse(
         str(recos),
         status_code=200
    )

    
def get_recommendations(user_id,articles_df,clicks_by_hour_df,articles_emb,top_k = 5):
    
    article_interest_df = clicks_by_hour_df[clicks_by_hour_df.user_id == user_id]['click_article_id']
    articles_categories = articles_df[articles_df.article_id.isin(article_interest_df)].category_id
    category_freqs = articles_df[articles_df.article_id.isin(article_interest_df)].category_id.value_counts()
    
    cf = category_freqs.index.to_series()
    cat=cf.to_numpy()[0]

    selected_article = articles_categories[articles_categories==cat].index[0]
    exclude_list = articles_categories[articles_categories==cat].index.to_numpy()
    current_emb = articles_emb[selected_article]
    
    similarities = np.dot(current_emb,np.transpose(articles_emb))

    to_retrieve = (top_k + len(exclude_list))-1
    selected = similarities.argsort()[-to_retrieve:]

    filtered = set(selected) - set(exclude_list)

    
    return list(filtered)
    