from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def recommender(name):
    anime_df = pd.read_pickle('data.pickle')

    tfidf = TfidfVectorizer(stop_words='english')

    anime_df['Description']=anime_df['Description'].fillna('')

    tfidf_matrix = tfidf.fit_transform(anime_df['Description'].tolist()).toarray()

    dt_tfidf = pd.DataFrame(tfidf_matrix,columns = tfidf.get_feature_names()).set_index(anime_df['Title'])


    target= [dt_tfidf.loc[name].values.tolist()]
    final = dt_tfidf.drop(name)
    results_tfidf = [cosine_similarity(target, 
                                    [final.loc[a].values.tolist()])[0][0] for a in final.index]
    
    return tuple(anime[1] for anime in sorted(zip(results_tfidf,final.index), reverse=True)[:5])