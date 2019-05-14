from data import CITIES, BUSINESSES, USERS, REVIEWS, TIPS, CHECKINS
from itertools import chain
from collections import Counter

import pandas as pd
import random

def reviewed_businesses(user_id):
    """Returns a list of the businesses the current user left a review on """
    user_reviews =[]
    for reviews in REVIEWS.values():
        for review in reviews:
            if review['user_id'] == user_id:
                user_reviews.append(review['business_id'])
    return(user_reviews)


def user_categories(user_businesses):
    """ Return the categories of the restaurants where the user left a review
    Do this with a dict so that the most reviewed type business has a higher number.
    A list of businesses that the user reviewed on needs to be given
    """
    user_categories =[]
    for businesses in BUSINESSES.values():
        for business in businesses:
            if business['business_id'] in user_businesses:
                categories = business["categories"].split(", ")
                user_categories.append(categories)
    
    return Counter(chain.from_iterable(user_categories))

def categories_dataframe():
    """ Make a dataframe with the business_id's and their categories"""
    all_data = list()
    for businesses in BUSINESSES.values():
        for business in businesses:
            business_id = business['business_id']
            categories = business['categories']
    
    # add to the data collected so far
            all_data.append([business_id, categories])

    # create the DataFrame
    categories_df = pd.DataFrame(all_data, columns=['business_id', 'categories'])
    return categories_df

def extract_categories():
    """" extract the categories"""
    businesses = categories_dataframe()
    categories_b = businesses.apply(lambda row: pd.Series([row['business_id']] + row['categories'].split(", ")), axis=1)
    stack_categories = categories_b.set_index(0).stack()
    df_stack_categories = stack_categories.to_frame()
    df_stack_categories['business_id'] = stack_categories.index.droplevel(1)
    df_stack_categories.columns = ['categories', 'business_id']
    return df_stack_categories.reset_index()[['business_id', 'categories']]

def utility_categories(df):
    """ Returns the utility matrix for all the businesses and their categories"""
    return df.pivot_table(index = 'business_id', columns = 'categories', aggfunc = 'size', fill_value=0)

def business_categories(categories):
    """ Return a list of businesses that fit the categories of the user.
    Score the businesses based on how many of the categories it contains"""

def business_rating(user_id, user_businesses):
    """ Return a dict with the ratings that users gave for a list of businesses"""
    user_ratings = {}
    for reviews in REVIEWS.values():
        for review in reviews:
            if review['business_id'] in user_businesses and review['user_id'] == user_id:
                user_ratings[review['business_id']] = review['stars']
    return user_ratings

def recommend(user_id=None, business_id=None, city=None, n=10):
    
    """
    Returns n recommendations as a list of dicts.
    Optionally takes in a user_id, business_id and/or city.
    A recommendation is a dictionary in the form of:
        {
            business_id:str
            stars:str
            name:str
            city:str
            adress:str
        }
    """

    print(user_id)

    # if the user is not logged in, give random options
    if user_id == None:
        if not city:
            city = random.choice(CITIES)
            return random.sample(BUSINESSES[city], n)

    # if the user is logged in
    reviewed = reviewed_businesses(user_id)
    print(reviewed)

    rating = business_rating(user_id, reviewed)
    print(rating)

    categories_dict = user_categories(reviewed)
    print(categories_dict)

    categories_dataframe = extract_categories()
    utility_matrix = utility_categories(categories_dataframe)
    print(utility_matrix.head())

    if not city:
        city = random.choice(CITIES)
        return random.sample(BUSINESSES[city], n)