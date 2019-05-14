from data import CITIES, BUSINESSES, USERS, REVIEWS, TIPS, CHECKINS
from itertools import chain
from collections import Counter

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
    
def business_categories(categories):
    """ Return a list of businesses that fit the categories of the user.
    Score the businesses based on how many of the categories it contains"""


def business_rating(user_id):
    """ Return a dataframe with the ratings that users gave for a list of businesses"""



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

    categories_dict = user_categories(reviewed)
    print(categories_dict)

    if not city:
        city = random.choice(CITIES)
        return random.sample(BUSINESSES[city], n)