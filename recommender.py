from helpers import *
from data import *

# from data import CITIES, BUSINESSES, USERS, REVIEWS, TIPS, CHECKINS

def density(dataframe):
    # give the density of the dataframe (how much is not NaN)
    df2 = dataframe.to_sparse()
    dense = df2.density
    print("density utility matrix: ",dense)
    return dense


def training_test():
    # make a training and test set
    df = json_to_df_stars()
    df_training, df_test = split_data(df, 0.75)
    return(df_training, df_test)


def predictions_item_based():
    df_training, df_test = training_test()
    
    # make utility and similarity matrix
    df_utility_ratings = pivot_ratings(df_training)
    df_similarity_ratings = create_similarity_matrix_cosine(df_utility_ratings)

    # predict ratings and calculate mse
    predicted_ratings = predict_ratings(df_similarity_ratings, df_utility_ratings, df_test)
    # predicted_ratings = predicted_ratings.dropna()

    return predicted_ratings

def predictions_content_based():
    df_training, df_test = training_test()

    # make the utility matrix with the amount of stars
    df_utility_stars = pivot_ratings(df_training)
    # create the dataframe with the business id's and categories
    categories_dataframe = extract_categories()
    # make utility matrix with the categories
    df_utility_categories = pivot_genres(categories_dataframe)
    # make a similarity matrix with the use of the categories utility matrix
    df_similarity_categories = create_similarity_matrix_categories(df_utility_categories)

    # predict the ratings
    predicted_ratings = predict_ratings(df_similarity_categories, df_utility_stars, df_test)
    # predicted_ratings = predicted_ratings.dropna()

    density(df_utility_stars)
    return predicted_ratings


def hybrid_based():
    item_prediction = predictions_item_based()
    content_prediction = predictions_content_based()
    combined = item_prediction
    combined['predict_content'] = content_prediction['predicted rating']
    combined['predicted rating'] = combined[['predicted rating', 'predict_content']].mean(axis=1)
    return item_prediction.drop(columns=['predict_content'])

training, test = training_test()
print("the size of the training set: ", training.size)
print("the size of the test set: ", test.size)

print("mse hybrid based: ", mse(hybrid_based()))
print("mse content based: ", mse(predictions_content_based()))
print("mse item based: ", mse(predictions_item_based()))

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
    # predictions = predictions_item_based()
    # #filter out the NaN
    # user_pred = predictions.loc[predictions['user_id'] == user_id].dropna()

    # options = user_pred['business_id'].tolist()
    # print(options)
    
    # # if there are not enough recommendations with item based, add the recommendations with hybrid
    # if len(options) < n:
    #     predictions =  hybrid_based()
    #     #filter out the NaN
    #     user_pred = predictions.loc[predictions['user_id'] == user_id].dropna()
    #     options = options + user_pred['business_id'].tolist()
    
    # # if there are still not enough options, add random recommendations
    # if len(options) < n:
    #     extra_businesses = random.sample(BUSINESSES[random.choice(CITIES)], n - len(options))
    #     print(extra_businesses)
    # recommend_this = []


    # for city in CITIES:
    #     for option in options:
    #         business = get_business(city, option)
    #         recommend_this = recommend_this.append(business)

    # return recommend_this + extra_businesses

    if not city:
        city = random.choice(CITIES)
        return random.sample(BUSINESSES[city], n)