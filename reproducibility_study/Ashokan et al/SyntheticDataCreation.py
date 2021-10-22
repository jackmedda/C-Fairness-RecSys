import pandas as pd
import random as rd

# the following distributions are based on Yao and Huang 2017 and used for the synthetic data evaluation in
# Ashokan and Haas 2021

# for creating the synthetic data
L = [[0.8, 0.2, 0.2], [0.8, 0.8, 0.2], [0.2, 0.8, 0.8], [0.2, 0.2, 0.8]]

# Create the pandas DataFrame
df_L = pd.DataFrame(L, columns=['Fem', 'Stem', 'Masc'], index=['W', 'WS', 'MS', 'M'])

O_unif = [[0.4, 0.4, 0.4], [0.4, 0.4, 0.4], [0.4, 0.4, 0.4], [0.4, 0.4, 0.4]]

# Create the pandas DataFrame
df_O_unif = pd.DataFrame(O_unif, columns=['Fem', 'Stem', 'Masc'], index=['W', 'WS', 'MS', 'M'])

O_bias = [[0.6, 0.2, 0.1], [0.3, 0.4, 0.2], [0.1, 0.3, 0.5], [0.05, 0.5, 0.35]]

# Create the pandas DataFrame
df_O_bias = pd.DataFrame(O_bias, columns=['Fem', 'Stem', 'Masc'], index=['W', 'WS', 'MS', 'M'])

pop_dist_unif = [0.25, 0.25, 0.25, 0.25]
pop_dist_imbalanced = [0.4, 0.1, 0.4, 0.1]

pop_dist_cumulative_unif = [0.25, 0.5, 0.75, 1]
pop_dist_cumulative_imbalanced = [0.4, 0.5, 0.9, 1]

num_users = 400
num_items = 300

num_item_types = 3
num_user_types = 4


def create_synthetic_data(population_distribution=pop_dist_cumulative_imbalanced, observation_model=df_O_bias,
                          rating_model=df_L, num_users=num_users, num_items=num_items):
    '''
    This method creates the synthetic data based on Yao and Huang 2017 as well as Ashokan and Haas 2021
    Args:
        population_distribution: the distribution of user types in the population
        observation_model: the probabilities of observations
        rating_model: the ratings dataframe
        num_users: the number of users
        num_items: the number of different items

    Returns: a dataframe with the observed ratings, and another dataframe with the unobserved ratings

    '''
    rating_data = []
    rating_unobserved_data = []

    # note: here we create an equal number of items
    items = []
    num_per_type = int(num_items/num_item_types)
    items.extend(num_per_type*['Fem'])
    items.extend(num_per_type * ['Stem'])
    items.extend(num_per_type*['Masc'])

    for i in range(1, num_users + 1):

        # first, determine user type
        new_user_rd = rd.random()
        if new_user_rd < population_distribution[0]:
            new_user = 'W'
        elif new_user_rd < population_distribution[1]:
            new_user = 'WS'
        elif new_user_rd < population_distribution[2]:
            new_user = 'MS'
        else:
            new_user = 'M'

        # second, determine rating for each item
        new_ratings = []
        for j in range(0, num_items):

            new_item_rd = rd.random()
            if new_item_rd < rating_model.loc[new_user][items[j]]:
                new_ratings.append(2)
            else:
                new_ratings.append(1)

        # third, determine if ratings are observed
        for j in range(0, num_items):
            new_obs_rd = rd.random()
            if new_obs_rd < observation_model.loc[new_user][items[j]]:
                if new_user in ['W', 'WS']:
                    rating_data.append([i, new_user, j, items[j], 'F', new_ratings[j]])
                else:
                    rating_data.append([i, new_user, j, items[j], 'M', new_ratings[j]])
            else:
                if new_user in ['W', 'WS']:
                    rating_unobserved_data.append([i, new_user, j, items[j], 'F', rating_model.loc[new_user][items[j]]+1])
                else:
                    rating_unobserved_data.append([i, new_user, j, items[j], 'M', rating_model.loc[new_user][items[j]]+1])

    ratings_df = pd.DataFrame(data=rating_data, columns=['user', 'user_type', 'item', 'item_type', 'gender', 'rating'])
    ratings_unobserved_df = pd.DataFrame(data=rating_unobserved_data,
                                         columns=['user', 'user_type', 'item', 'item_type', 'gender', 'rating'])

    return ratings_df, ratings_unobserved_df

