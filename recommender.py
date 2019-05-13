from data import CITIES, BUSINESSES, USERS, REVIEWS, TIPS, CHECKINS

import random

def reviewed_businesses(user_id):
    # Returns a list of the businesses the current user left a review on 
    user_reviews =[]
    for city, reviews in REVIEWS.items():
        for review in reviews:
            if review['user_id'] == user_id:
                user_reviews.append(review['business_id'])
                return(user_reviews)


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
# if the user is not logged in, give random options
    if user_id == None:
        if not city:
            city = random.choice(CITIES)
            return random.sample(BUSINESSES[city], n)

    # if the user is logged in
    reviewed = reviewed_businesses(user_id)
    print(reviewed)

    if not city:
        city = random.choice(CITIES)
        return random.sample(BUSINESSES[city], n)