import os
from typing import (
    Iterator,
    Tuple,
    Dict,
    Any
)

import pandas as pd


def parse_plays_data_1K(
        dir_path: str
) -> Iterator[Tuple[int, Dict[str, Any]]]:
    plays_file_path = os.path.join(dir_path, 'userid-timestamp-artid-artname-traid-traname.tsv')
    df_plays = pd.read_csv(
        plays_file_path,
        sep='\t',
        encoding='utf-8',
        names=['user_id', 'timestamp', 'musicbrainz-artist-id', 'artist-name', 'musicbrainz-track-id', 'track-name']
    )

    df_plays = df_plays.groupby(['user_id', 'musicbrainz-artist-id'], dropna=True).agg(
        user_id=pd.NamedAgg(column="user_id", aggfunc="first"),
        plays=pd.NamedAgg(column="musicbrainz-artist-id", aggfunc="count"),
        artist_id=pd.NamedAgg(column="musicbrainz-artist-id", aggfunc="first"),
        artist_name=pd.NamedAgg(column="artist-name", aggfunc="first")
    ).reset_index(drop=True)

    order_user_id = sorted(df_plays['user_id'].unique())
    map_user_id = dict(zip(order_user_id, range(1, len(order_user_id) + 1)))

    # order_artist_id = sorted(df_plays['musicbrainz-artist-id'].unique())
    # map_artist_id = dict(zip(order_artist_id, range(1, len(order_artist_id) + 1)))
    order_artist_id = sorted(df_plays['artist_id'].unique())
    map_artist_id = dict(zip(order_artist_id, range(1, len(order_artist_id) + 1)))

    # order_track_id = sorted(df_plays['musicbrainz-track-id'].unique())
    # map_track_id = dict(zip(order_track_id, range(1, len(order_track_id) + 1)))

    # df_plays['artist_id'] = [str(a_id) for a_id in df_plays['musicbrainz-artist-id'].map(map_artist_id)]
    df_plays['artist_id'] = [str(a_id) for a_id in df_plays['artist_id'].map(map_artist_id)]
    df_plays['user_id'] = [str(u_id) for u_id in df_plays['user_id'].map(map_user_id)]
    # df_plays['track_id'] = [str(t_id) for t_id in df_plays['musicbrainz-track-id'].map(map_track_id)]

    df_plays['artist_name'].fillna('Name_Not_Available', inplace=True)
    # df_plays['track-name'].fillna('Name_Not_Available', inplace=True)

    for row_num, (_, row) in enumerate(df_plays.iterrows()):
        ex = {
            'user_id': row['user_id'],
            'artist_id': row['artist_id'],
            'plays': row['plays'],
            'artist_name': row['artist_name']
        }
        yield row_num, ex

    # for row_num, (_, row) in enumerate(df_plays.iterrows()):
    #     ex = {
    #         'user_id': row['user_id'],
    #         'timestamp': row['timestamp'],
    #         'artist_id': row['artist_id'],
    #         'artist_name': row['artist-name'],
    #         'track_id': row['musicbrainz-track-id'],
    #         'track_name': row['track-name']
    #     }
    #     yield row_num, ex


def parse_users_data_1K(
        dir_path: str
) -> Iterator[Tuple[int, Dict[str, Any]]]:
    users_file_path = os.path.join(dir_path, 'userid-profile.tsv')
    df_users = pd.read_csv(
        users_file_path,
        sep='\t',
        encoding='utf-8'
    )

    df_users.dropna(subset=['#id'], inplace=True)

    order_user_id = sorted(df_users['#id'].unique())
    map_user_id = dict(zip(order_user_id, range(1, len(order_user_id) + 1)))

    df_users['#id'] = [str(u_id) for u_id in df_users['#id'].map(map_user_id)]

    df_users['gender'].fillna('NA', inplace=True)
    df_users['gender'] = df_users['gender'].map(str.upper)
    df_users['age'].fillna(-1, inplace=True)
    df_users['country'].fillna('NA', inplace=True)
    df_users['registered'].fillna('NA', inplace=True)

    return __parse_users_data(df_users)


def parse_plays_data_360K(
        dir_path: str
) -> Iterator[Tuple[int, Dict[str, Any]]]:
    plays_file_path = os.path.join(dir_path, 'usersha1-artmbid-artname-plays.tsv')
    df_plays = pd.read_csv(
        plays_file_path,
        sep='\t',
        encoding='utf-8',
        names=['user-mboxsha1', 'musicbrainz-artist-id', 'artist-name', 'plays']
    )

    df_plays.dropna(subset=['musicbrainz-artist-id'], inplace=True)
    df_plays = df_plays[df_plays['plays'] > 0]  # dataset contains one row with plays = 0 (to be cleaned)

    order_artist_id = sorted(df_plays['musicbrainz-artist-id'].unique())
    map_artist_id = dict(zip(order_artist_id, range(1, len(order_artist_id) + 1)))

    order_user_id = sorted(df_plays['user-mboxsha1'].unique())
    map_user_id = dict(zip(order_user_id, range(1, len(order_user_id) + 1)))

    df_plays['artist_id'] = [str(a_id) for a_id in df_plays['musicbrainz-artist-id'].map(map_artist_id)]
    df_plays['user_id'] = [str(u_id) for u_id in df_plays['user-mboxsha1'].map(map_user_id)]
    df_plays['artist-name'].fillna('Name_Not_Available', inplace=True)

    for row_num, (_, row) in enumerate(df_plays.iterrows()):
        ex = {
            'user_id': row['user_id'],
            'artist_id': row['artist_id'],
            'artist_name': row['artist-name'],
            'plays': row['plays']
        }
        yield row_num, ex


def parse_users_data_360K(
        dir_path: str
) -> Iterator[Tuple[int, Dict[str, Any]]]:
    users_file_path = os.path.join(dir_path, 'usersha1-profile.tsv')
    df_users = pd.read_csv(
        users_file_path,
        sep='\t',
        encoding='utf-8',
        names=['#id', 'gender', 'age', 'country', 'registered']
    )

    order_user_id = sorted(df_users['#id'].unique())
    map_user_id = dict(zip(order_user_id, range(1, len(order_user_id) + 1)))

    df_users['#id'] = [str(u_id) for u_id in df_users['#id'].map(map_user_id)]

    df_users['gender'].fillna('NA', inplace=True)
    df_users['gender'] = df_users['gender'].map(str.upper)
    df_users['age'].fillna(-1, inplace=True)
    df_users['country'].fillna('NA', inplace=True)
    df_users['registered'].fillna('NA', inplace=True)

    return __parse_users_data(df_users)


def __parse_users_data(df_users: pd.DataFrame) -> Iterator[Tuple[int, Dict[str, Any]]]:
    for row_num, (_, row) in enumerate(df_users.iterrows()):
        ex = {
            'user_id': row['#id'],
            'user_gender': row['gender'],
            'user_age': row['age'],
            'user_country': row['country'],
            'user_registration_date': row['registered']
        }
        yield row_num, ex
