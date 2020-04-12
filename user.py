from surprise import KNNBasic

import numpy as np

import heapq
from operator import itemgetter

from input_data import MovieLens

ml = MovieLens()
data = ml.load_movielens()
trainset = data.build_full_trainset()

algo = KNNBasic(sim_options={'name': 'cosine', 'user_based': True})
algo.fit(trainset)

def recommendations_user_based_cf(trainset, raw_uid, quality_threshold=0.9, recs=10):
    user = trainset.to_inner_uid(raw_uid)
    similarities = algo.sim[user]

    similar_users = list()
    for similar_user, score in enumerate(similarities):
        if (similar_user != user) and (score >= quality_threshold):
            similar_users.append((similar_user, score))
    knn = heapq.nlargest(len(similar_users), similar_users, key=lambda x: x[1])

    candidates = defaultdict(float)
    for similar_user, score in knn:
        similar_user_ratings = trainset.ur[similar_user]
        for movie_id, rating in similar_user_ratings:
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
            if n_recommendations >= recs:
                break

    # COLD START PROBLEM: RANDOM EXPLORATION
    random_movie = ml.get_random_movie()
    print(ml.get_movie_name(random_movie.iloc[0]))

# PRODUCE TOP-N RECOMMENDATIONS FOR GIVEN USER
recommendations_user_based_cf(trainset, raw_uid=28)
