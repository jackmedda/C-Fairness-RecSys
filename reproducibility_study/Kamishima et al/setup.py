import re
from setuptools import setup, find_packages
from kamiers import (__version__, __license__, __author__)

(author, author_email) = \
    re.search('([^()]+)\s+\(\s+(.+)\s+\)', __author__).groups()

with open('readme.md') as in_file:
    long_description = in_file.read()

setup(
    name='kamiers',
    version=__version__,
    license=__license__,
    author=author,
    author_email=author_email,
    url='https://github.com/tkamishima/kamiers',
    description='kamiers: '
                'test codes for independence-enhanced recommender systems',
    long_description=long_description,
    keywords='recommender system',
    packages=find_packages(
        exclude=['*.event', '*.item', '*.user']
    ),
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn',
        'six',
        'kamrecsys'],
    test_suite='nose.collector',
)
