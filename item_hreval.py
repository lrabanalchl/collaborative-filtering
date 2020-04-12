from surprise.model_selection import LeaveOneOut

from input_data import MovieLens
from hit_rate_metric import hit_rate


ml = MovieLens()
data = ml.load_movielens()

def top_n_item_based_cf(trainset, user, quality_threshold=4.0, recs=10):
    
    user_ratings = trainset.ur[user]
    
    knn = [(movie, rating) for movie, rating in user_ratings if rating >= quality_threshold]

    candidates = defaultdict(float)
    for movie, rating in knn:
        similar_movies = algo.sim[movie]
        for movie_id, score in enumerate(similar_movies):
            candidates[movie_id] += (rating / 5) * score

    already_watched = {movie_id: 1 for movie_id, _ in trainset.ur[user]}

    n_recommendations = 0
    for movie, final_score in sorted(candidates.items(), 
                                     key=itemgetter(1), 
                                     reverse=True):
        if movie not in already_watched:
            n_recommendations += 1
            movie_id = trainset.to_raw_iid(movie)
            top_n[int(trainset.to_raw_uid(user))].append((int(movie_id), 0.0))
            if n_recommendations >= recs:
                break

    # COLD START PROBLEM: RANDOM EXPLORATION
    random_movie = ml.get_random_movie()
    top_n[int(trainset.to_raw_uid(user))].append((random_movie.iloc[0], 0.0))
    
    
LOOCV = LeaveOneOut(n_splits=1, random_state=1)

for trainset, testset in LOOCV.split(data):

    algo = KNNBasic(sim_options={'name': 'cosine', 'user_based': False})
    algo.fit(trainset)

    top_n = defaultdict(list)
    for uiid in range(trainset.n_users):
        top_n_item_based_cf(trainset, uiid, recs=40, quality_threshold=4.5)

print("Hit Rate: ", hit_rate(top_n, testset))
