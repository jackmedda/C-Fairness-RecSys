import os
from typing import (
    Iterator,
    Tuple,
    Dict,
    Any
)


def parse_ratings_data(
        dir_path: str
) -> Iterator[Tuple[int, Dict[str, Any]]]:
    ratings_file_path = os.path.join(dir_path, 'ratings.dat')
    movie_info_map = {}
    for _, movie_example in parse_movies_data(dir_path):
        movie_info_map[movie_example['movie_id']] = movie_example

    with open(ratings_file_path, encoding='utf-8') as ratings_file:
        for row_num, row in enumerate(ratings_file.read().splitlines(keepends=False)):
            _row = row.split('::')
            ex = {
                'user_id': _row[0],
                'movie_id': _row[1],
                'user_rating': _row[2],
                'timestamp': _row[3]
            }
            ex.update(movie_info_map[ex['movie_id']])
            yield row_num, ex


def parse_movies_data(
        dir_path: str
) -> Iterator[Tuple[int, Dict[str, Any]]]:
    movies_file_path = os.path.join(dir_path, 'movies.dat')

    with open(movies_file_path, encoding='utf-8') as movies_file:
        for row_num, row in enumerate(movies_file.read().splitlines(keepends=False)):
            _row = row.split('::')
            yield row_num, {
                'movie_id': _row[0],
                'movie_title': _row[1],
                'movie_genres': _row[2].split('|')
            }
