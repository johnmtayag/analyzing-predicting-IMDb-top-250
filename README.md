# Analyzing and Predicting IMDb Top 250 Films

# Background
IMDb (Internet Movie Database) is an online database that lists details on multiple forms of visual media, most notably films and TV shows, and hosts 
millions of users. The details listed include casts, production crew, and plot summaries, as well as both critical reviews and user ratings. 
Using this information, IMDb maintains a list of the highest rated 250 movies updated constantly as new reviews are submitted.

Using a dataset uploaded by Mustafa Cicek on Kaggle, I would like to analyze the top 250 movies over the years to try answer a few questions:

* What features make a popular movie?<br>
* How have movies changed over the years?<br>
* Which actors and directors have had the most success over the years?<br>

I will also build a model to try and predict whether or not a movie a movie will be considered popular (as in, how high on the list will the movie be ranked?). 
I will detail this further in the machine learning section.

# Dataset
https://www.kaggle.com/datasets/mustafacicek/imdb-top-250-lists-1996-2020

The dataset was compiled and uploaded by Mustafa Cicek from Kaggle, and the information was scraped from IMDb using Beautiful Soup. 
As the top 250 list is constantly changing, the data from each year was extracted at midnight of December 31st (PST) to reflect the final rankings of that year.

There are 6250 rows (25 years of top 250 movies), and there are 16 features:

>Ranking: The ranking of a movie in a given year<br>
>IMDbyear: The year for a given Top 250 list<br>
>IMDBlink: The movie's IMDb url<br>
>Title: Title of the movie<br>
>Date: Movie's release date<br>
>RunTime: Total runtime of the movie (in minutes)<br>
>Genre: Different genres the movie flls under<br>
>Rating: The IMDb score of the movie - a weighted average of IMDb user ratings<br>
>Score: The metascore of the movie - a weighted average of professional critical reviews<br>
>Votes: Total # of votes on IMDb for a movie<br>
>Gross: Total gross of a movie (in millions of dollars)<br>
>Director: Director of the movie<br>
>Cast1: Performer 1 for a movie<br>
>Cast2: Performer 2 for a movie<br>
>Cast3: Performer 3 for a movie<br>
>Cast4: Performer 4 for a movie

While individual user preferences and movie qualities may differ greatly, these features are broad enough to allow for a relatively non-biased analysis.

# Analysis

The data was explored under multiple categories, including genre, rating, and total gross earnings. The top directors and actors were also analyzed.

Machine learning models were built to test whether this dataset was enough to predict movie rankings. This analysis focused on decision trees, random forests,
and SVMs. Predicting individual ranks from 1 to 250 is very difficult, so instead, the data was binned into categories:

> 1. Rank 1-10<br>
> 2. Rank 11-50<br>
> 3. Rank 51-100<br>
> 4. Rank 101-250<br>

These categories were chosen arbitrarily, but serve to group the movies into bins that are easier to work with. It is much easier to predict if a film will
reach the top 10 instead of predicting the exact rank. 
