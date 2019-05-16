from helpers import *


from data import CITIES, BUSINESSES, USERS, REVIEWS, TIPS, CHECKINS

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

    useful_busines = business_categories(utility_matrix, categories_dict)
    print(useful_busines)

    if not city:
        city = random.choice(CITIES)
        return random.sample(BUSINESSES[city], n)