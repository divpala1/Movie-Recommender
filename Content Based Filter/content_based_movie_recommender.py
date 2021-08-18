import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

""" 
PROGRAM EXPLAINED

Step-5
This type of cosine similarity matrix is formed. All the scores tell us how much 2 movies are similar.
Movie|   0   |   1   |   2   |   3   |...           
  0  |   1   |  0.8  |  0.3  |  0.5  |...  
  1  |  0.2  |   1   |  0.9  |  0.4  |...
  2  |  0.6  |  0.9  |   1   |  0.7  |...
  3  |  0.7  |  0.2  |  0.4  |   1   |...
 ...

Step-6
Movie index is extracted from the title using the helper function. And then a enumerated list of the cosine similarity is retrieved for the movie.
For e.g. if it is movie number 1, the list [(0, 0.2), (1, 1), (2, 0.9), (3, 0.4)...] is retrieved.

Step-7
The list from the step above is sorted in descending order of cosine similarities.

Step-8
The list of indexes is converted to list of names and is printed.
"""


# Helper functions
def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]


def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]


##################################################


# STEP 1: Read CSV File
df = pd.read_csv("movie_dataset.csv")
# print(df.columns)


# STEP 2: Select Features
features = ["keywords", "cast", "genres", "director"]


# STEP 3: Creating a column in DF which has all the selected features combined.

# After the try-except block ran, the problem revealed to be that some "keywords" were missing/NaN (NaN is type=float). To encounter that, the missing values were filled using fillna().
df[features] = df[features].fillna("")


def combine_features(row):
    try:
        # When the above function is executed with only the line below, an error is encountered (TypeError: unsupported operand type(s) for +: 'float' and 'str'). To troubleshoot that, a try-except block is applied which will print the row(s) which is/are causing the error.
        return row["keywords"] + " " + row["cast"] + " " + row["genres"] + " " + row["director"]

    except:
        print("Error: ", row[features])


# "df.apply(function_name, axis=)" implements the "function_name" function on all the given rows (if axis=1) or on all the columns (if axis=0).
df["combined_features"] = df.apply(combine_features, axis=1)
# print("Combined features:\n", df["combined_features"].head())


# STEP 4: Create count matrix from this new combined column
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])


# STEP 5: Compute the Cosine Similarity based on the count_matrix
cosine_sim = cosine_similarity(count_matrix)

movie_user_likes = "Avatar"


# STEP 6: Get index of this movie from its title
movie_index = get_index_from_title(movie_user_likes)

# Creating a list which has the following format: (0, 0.5), (1, 0.9), (2, 0.75)... All the the cosine similarity values now are indexed.
similar_movies = list(enumerate(cosine_sim[movie_index]))


# STEP 7: Get a list of similar movies in descending order of cosine similarity score. Meaning the most similar movie will be first.

# The lambda function returns the second item i.e. item at index 1 of the enumerated list i.e. the cosine similarity value. Format of list: (index, cosine similarity value)
sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)


# STEP 8: Print titles of first 50 movies
for i in range(50):
    print(get_title_from_index(sorted_similar_movies[i][0]))  # "sorted_similar_movies" is a list of tuple items in format:[(index, cosine similarity), (index, cosine similarity)...] so sorted...[i][0] will get the i-th item, and the 0th index value (name of the movie) of the i-th item.
