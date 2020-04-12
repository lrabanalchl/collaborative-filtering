import pandas as pd

from surprise import Reader
from surprise import Dataset

from collections import defaultdict


class MovieLens:

    def __init__(self, ratings_path, movies_path):

        self.movies_df = pd.read_csv(movies_path)
        self.ratings_df = pd.read_csv(ratings_path)

        self.movie_id_to_name = {}
        self.name_to_movie_id = {}

        for movie_id in self.movies_df['movieId'].unique():
            mask = self.movies_df['movieId'] == movie_id
            title = self.movies_df[mask]['title'].values[0]
            self.movie_id_to_name[movie_id] = title
            self.name_to_movie_id[title] = movie_id

    def load_movielens(self):
        """Loads the movielens dataset from the Pandas DataFrame."""

        reader = Reader()
        cols = ['userId', 'movieId', 'rating']
        return Dataset.load_from_df(self.ratings_df[cols], reader=reader)

    def get_user_ratings(self, user_id):
        """Give me the raw user_id, I will give you a list of tuples 
        displaying the history of movies and ratings of this user.
        """

        mask = self.ratings_df['userId'] == user_id
        return self.ratings_df[mask].apply(lambda x: (int(x[1]), x[2]), 
                                           axis=1).tolist()
        

    def get_popularity_ranks(self):
        """Gives the most popular movies based on the number of times a
        specific movie has been rated.
        """
        n_ratings_per_movie = self.ratings_df['movieId'].value_counts()
        
        rankings = {}
        for rank, movie_id in enumerate(n_ratings_per_movie.index):
            rankings[movie_id] = rank + 1

        return rankings

    def get_genres(self):
        genres_per_movie = defaultdict(list)
        genre_ids = {}
        max_genre_id = 0

        for index, genres in enumerate(self.movies_df['genres'].str.split('|').values):
    
            genre_ids_per_movie = list()
            for genre in genres:
                if genre not in genre_ids:
                    genre_ids[genre] = max_genre_id
                    max_genre_id += 1
                genre_ids_per_movie.append(genre_ids[genre])

            genres_per_movie[self.movies_df.loc[index, 'movieId']] = genre_ids_per_movie
       
        for movie_id, genre_ids_per_movie in genres_per_movie.items():
            bitfield = [0] * max_genre_id
            for genre_id in genre_ids_per_movie:
                bitfield[genre_id] = 1
            genres_per_movie[movie_id] = bitfield

        return genres_per_movie

    def get_years(self):
        df = self.movies_df.copy()

        df['year'] = df['title'].str.extract(r'([1-2][0-9]{3})')

        movies_with_year = df[df['year'].notnull()]
        movies_with_year['year'] = movies_with_year['year'].astype(int)

        years = defaultdict(int)
        for movie_id in movies_with_year['movieId'].unique():
            mask = movies_with_year['movieId'] == movie_id
            year = movies_with_year[mask]['year'].values[0]
            years[movie_id] = year

        return years

    def get_movie_name(self, movie_id):
        if movie_id in self.movie_id_to_name:
            return self.movie_id_to_name[movie_id]
        return ""
        
    def get_movie_id(self, movie_name):
        if movie_name in self.name_to_movie_id:
            return self.name_to_movie_id[movie_name]
        return 0
