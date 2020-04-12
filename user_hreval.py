from surprise.model_selection import LeaveOneOut

from input_data import MovieLens

def top_n_user_based_cf(trainset, user, quality_threshold=0.9, recs=10):
    similarities = algo.sim[user]

    similar_users = list()
    for similar_user, score in enumerate(similarities):
        if (similar_user != user) and (score >= quality_threshold):
            similar_users.append((similar_user, score))

    candidates = defaultdict(float)
    for similar_user, score in similar_users:
        similar_user_ratings = trainset.ur[similar_user]
        for movie_id, rating in similar_user_ratings:
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

    algo = KNNBasic(sim_options={'name': 'cosine', 'user_based': True})
    algo.fit(trainset)

    top_n = defaultdict(list)
    for uiid in range(trainset.n_users):
        top_n_user_based_cf(trainset, uiid, recs=40, quality_threshold=0.95)

print("Hit Rate: ", hit_rate(top_n, testset))
