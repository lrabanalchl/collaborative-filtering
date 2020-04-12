def hit_rate(top_n, left_out_testset):
    
    hits = 0
    for user_id, left_out_movie_id, _ in left_out_testset:
        # Is it in the predicted top 10 for this user?
        hit = False
        for movie_id, estimated_rating in top_n[int(user_id)]:
            if int(left_out_movie_id) == int(movie_id):
                hit = True
                break
        if hit:
            hits += 1

    return hits / len(left_out_testset)
