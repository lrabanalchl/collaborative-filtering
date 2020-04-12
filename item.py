from surprise import KNNBasic

import numpy as np

import heapq
from operator import itemgetter

from input_data import MovieLens


ml = MovieLens()
data = ml.load_movielens()
trainset = data.build_full_trainset()

algo = KNNBasic(sim_options={'name': 'cosine', 'user_based': False})
algo.fit(trainset)

def recommendations_item_based_cf(trainset, raw_uid, quality_threshold=4.0):
    user = trainset.to_inner_uid(raw_uid)
    
    user_ratings = trainset.ur[user]
    
    knn = [(movie, rating) for movie, rating in user_ratings if rating >= quality_threshold]
    knn = heapq.nlargest(len(knn), knn, key=lambda x: x[1])

    candidates = defaultdict(float)
    for movie, rating in knn:
        similar_movies = algo.sim[movie]
        for movie_id, score in enumerate(similar_movies):
            candidates[movie_id] += (rating / 5) * score

    already_watched = {movie_id: 1 for movie_id, _ in trainset.ur[user]}

    n_recommendations = 0
    print("We recommend:")
    for movie, final_score in sorted(candidates.items(), 
                                     key=itemgetter(1), 
                                     reverse=True):
        if movie not in already_watched:
            n_recommendations += 1
            movie_id = trainset.to_raw_iid(movie)
            print(ml.get_movie_name(int(movie_id)), final_score)
            if n_recommendations >= 10:
                break

    # COLD START PROBLEM: RANDOM EXPLORATION
    random_movie = ml.get_random_movie()
    print(ml.get_movie_name(random_movie.iloc[0]))
    
    # TOP-N RECOMMENDATIONS FOR GIVEN USER 
    recommendations_item_based_cf(trainset, 28)
