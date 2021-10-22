"""lastfm dataset."""

import os
import inspect
import sys
sys.path.append(os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(inspect.getsourcefile(lambda: 0)))))
))

import textwrap
from typing import Optional, Callable, Iterator, Tuple, Dict, Any

import tensorflow_datasets as tfds
import tensorflow as tf
import data.datasets.lastfm.lastfm_parsing as lastfm_parsing

_DESCRIPTION = """
 ## Last.fm 360K
 
 . What is this?

    This dataset contains <user, artist, plays> tuples (for ~360,000 users) collected from Last.fm API,
    using the user.getTopArtists() method.

 . Files:

    usersha1-artmbid-artname-plays.tsv (MD5: be672526eb7c69495c27ad27803148f1)
    usersha1-profile.tsv               (MD5: 51159d4edf6a92cb96f87768aa2be678)
    mbox_sha1sum.py                    (MD5: feb3485eace85f3ba62e324839e6ab39)

 . Data Statistics:

    File usersha1-artmbid-artname-plays.tsv:

      Total Lines:           17,559,530
      Unique Users:             359,347
      Artists with MBID:        186,642
      Artists without MBID:     107,373

 . Data Format:

    The data is formatted one entry per line as follows (tab separated "\t"):

    File usersha1-artmbid-artname-plays.tsv:
      user-mboxsha1 \t musicbrainz-artist-id \t artist-name \t plays
    
    File usersha1-profile.tsv:
      user-mboxsha1 \t gender (m|f|empty) \t age (int|empty) \t country (str|empty) \t signup (date|empty)

 . Example:

    usersha1-artmbid-artname-plays.tsv:
      000063d3fe1cf2ba248b9e3c3f0334845a27a6be \t a3cb23fc-acd3-4ce0-8f36-1e5aa6a18432 \t u2 \t 31
      ...

    usersha1-profile.tsv
      000063d3fe1cf2ba248b9e3c3f0334845a27a6be \t m \t 19 \t Mexico \t Apr 28, 2008
      ...

 ## Last.fm 1K

. What is this?

    This dataset contains <user, timestamp, artist, song> tuples collected from Last.fm API, 
    using the user.getRecentTracks() method.

    This dataset represents the whole listening habits (till May, 5th 2009) for nearly 1,000 users.

 . Files:

    userid-timestamp-artid-artname-traid-traname.tsv (MD5: 64747b21563e3d2aa95751e0ddc46b68)
    userid-profile.tsv                               (MD5: c53608b6b445db201098c1489ea497df)

 . Data Statistics:

    File userid-timestamp-artid-artname-traid-traname.tsv

      Total Lines:           19,150,868
      Unique Users:                 992
      Artists with MBID:        107,528
      Artists without MBDID:     69,420

 . Data Format:

    The data is formatted one entry per line as follows (tab separated, "\t"):

    userid-timestamp-artid-artname-traid-traname.tsv
      userid \t timestamp \t musicbrainz-artist-id \t artist-name \t musicbrainz-track-id \t track-name

    userid-profile.tsv:
      userid \t gender ('m'|'f'|empty) \t age (int|empty) \t country (str|empty) \t signup (date|empty)

 . Example:

    userid-timestamp-artid-artname-traid-traname.tsv:
      user_000639 \t 2009-04-08T01:57:47Z \t MBID \t The Dogs D'Amour \t MBID \t Fall in Love Again?
      user_000639 \t 2009-04-08T01:53:56Z \t MBID \t The Dogs D'Amour \t MBID \t Wait Until I'm Dead
      ...

    userid-profile.tsv:
      user_000639 \t m \t Mexico \t Apr 27, 2005
      ...

 . License:

    The data contained in lastfm-dataset-1K.tar.gz is distributed with permission of Last.fm. 
    The data is made available for non-commercial use.
    Those interested in using the data or web services in a commercial context should contact: 

    partners [at] last [dot] fm

    For more information see Last.fm terms of service

 . Acknowledgements:

    Thanks to Last.fm for providing the access to this data via their web services. 
    Special thanks to Norman Casagrande.

 . Contact:

    This data was collected by Òscar Celma @ MTG/UPF
"""

_CITATION = """
@book{Celma:Springer2010,
author = {Celma, O.},
title = {{Music Recommendation and Discovery in the Long Tail}},
publisher = {Springer},
year = {2010}
}
"""


class LastFMConfig(tfds.core.BuilderConfig):
    """BuilderConfig for LastFM dataset."""

    def __init__(self,
                 format_version: Optional[str] = None,
                 table_option: Optional[str] = None,
                 download_url: Optional[str] = None,
                 parsing_fn: Optional[Callable[[str], Iterator[Tuple[int, Dict[
                     str, Any]]], ]] = None,
                 **kwargs) -> None:
        """Constructs a MovieLensConfig.
    Args:
      format_version: a string to identify the format of the dataset, one of
        '_FORMAT_VERSIONS'.
      table_option: a string to identify the table to expose, one of
        '_TABLE_OPTIONS'.
      download_url: a string url for downloading the dataset.
      parsing_fn: a callable for parsing the data.
      **kwargs: keyword arguments forwarded to super.
    """
        super(LastFMConfig, self).__init__(**kwargs)
        self._format_version = format_version
        self._table_option = table_option
        self._download_url = download_url
        self._parsing_fn = parsing_fn

    @property
    def format_version(self) -> str:
        return self._format_version

    @property
    def table_option(self) -> str:
        return self._table_option

    @property
    def download_url(self) -> str:
        return self._download_url

    @property
    def parsing_fn(
            self) -> Optional[Callable[[str], Iterator[Tuple[int, Dict[str, Any]]], ]]:
        return self._parsing_fn


class Lastfm(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for lastfm dataset."""

    BUILDER_CONFIGS = [
        LastFMConfig(
            name='1K-plays',
            description=textwrap.dedent("""\
                      File userid-timestamp-artid-artname-traid-traname.tsv

                      Total Lines:           19,150,868
                      Unique Users:                 992
                      Artists with MBID:        107,528
                      Artists without MBDID:     69,420"""),
            version='1.0.0',
            format_version='1K',
            table_option='plays',
            download_url='http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-1K.tar.gz',
            parsing_fn=lastfm_parsing.parse_plays_data_1K,
        ),
        LastFMConfig(
            name='1K-users',
            description=textwrap.dedent("""\
                      File userid-profile.tsv

                      This dataset contains data of 992 users.
                      """),
            version='1.0.0',
            format_version='1K',
            table_option='users',
            download_url='http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-1K.tar.gz',
            parsing_fn=lastfm_parsing.parse_users_data_1K,
        ),
        LastFMConfig(
            name='360K-plays',
            description=textwrap.dedent("""\
                          File usersha1-artmbid-artname-plays.tsv:

                          Total Lines:           17,559,530
                          Unique Users:             359,347
                          Artists with MBID:        186,642
                          Artists without MBID:     107,373"""),
            version='1.2.0',
            format_version='360K',
            table_option='plays',
            download_url='http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-360K.tar.gz',
            parsing_fn=lastfm_parsing.parse_plays_data_360K,
        ),
        LastFMConfig(
            name='360K-users',
            description=textwrap.dedent("""\
                          File usersha1-profile.tsv
            
                          This dataset contains data of 359,347 users."""),
            version='1.2.0',
            format_version='360K',
            table_option='users',
            download_url='http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-360K.tar.gz',
            parsing_fn=lastfm_parsing.parse_users_data_360K,
        )
    ]

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""

        # plays_features_dict_1K = {
        #     'user_id':
        #         tf.string,
        #     'timestamp':
        #         tf.string,
        #     'artist_id':
        #         tf.string,
        #     'artist_name':
        #         tf.string,
        #     'track_id':
        #         tf.string,
        #     'track_name':
        #         tf.string
        # }

        plays_features_dict_360K = {
            'user_id':
                tf.string,
            'artist_id':
                tf.string,
            'artist_name':
                tf.string,
            'plays':
                tf.int64,
        }

        users_features_dict = {
            'user_id':
                tf.string,
            'user_gender':
                tf.string,
            'user_age':
                tf.int32,
            'user_country':
                tf.string,
            'user_registration_date':
                tf.string
        }

        features_dict = {}
        if self.builder_config.table_option == 'users':
            features_dict.update(users_features_dict)
        elif self.builder_config.format_version == '1K':
            # features_dict.update(plays_features_dict_1K)
            features_dict.update(plays_features_dict_360K)
        else:
            features_dict.update(plays_features_dict_360K)

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(features_dict),
            supervised_keys=None,
            homepage='http://ocelma.net/MusicRecommendationDataset/index.html',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        extracted_path = dl_manager.download_and_extract(
            self.builder_config.download_url,)

        dir_path = os.path.join(
            extracted_path,
            f'lastfm-dataset-{self.builder_config.format_version}',
        )

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={'dir_path': dir_path},
            ),
        ]

    def _generate_examples(self, dir_path):
        """Yields examples by calling the corresponding parsing function."""
        for ex in self.builder_config.parsing_fn(dir_path):
            yield ex
