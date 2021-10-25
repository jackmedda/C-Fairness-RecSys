
# Parity LBM  and standard LBM with ordinal data

See requirements.txt file for module dependencies.

## Download Movielens 1m dataset:

- Download https://files.grouplens.org/datasets/movielens/ml-1m.zip
- Extract  users.dat, ratings.dat and movies.dat in data folder

## Pre-Processing Movielens 1M dataset:

```
python3 make_dataset_movielens.py
```
Creates a pickle file containing test sets and data ready to process.

## Launch experiments  on Movielens 1M dataset with gender as sensitive attribute:

```
python3 start_experiment.py
```

- Use with higher number of groups:

```
python3 start_experiment.py --nq=50 --nl=50
```

- experiments on a different test set (from 0 to 4):

```
python3 start_experiment.py --block=4
```


## Getting results:

```
python3 results_experiments.py
```
