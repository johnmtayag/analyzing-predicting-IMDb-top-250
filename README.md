# Analyzing and Predicting IMDb Top 250 Films

# Introduction
IMDb (Internet Movie Database) is an online database that lists details on multiple forms of visual media, most notably films and TV shows, and hosts millions of users. The details listed include casts, production crew, and plot summaries, as well as both critical reviews and user ratings. Using this information, IMDb maintains a list of the highest rated 250 movies updated constantly as new reviews are submitted.

Using a dataset uploaded by Mustafa Cicek on Kaggle, I analyzed the top 250 movies over the years to try answer a few questions:

* What features make a popular movie?<br>
* How have movies changed over the years?<br>
* Which actors and directors have had the most success over the years?<br>

I also built a model to try and predict whether or not a movie a movie would be considered popular (as in, given the information provided in the dataset, how high on the list would a given movie be ranked?). 

This was one of my first data analysis and machine learning projects, so some details were not fully fleshed out. Were I to return to this project, I would focus a lot more on tuning my model hyperparameters to achieve better results. However, I am satisfied with my results for now. I edited and updated some of the visuals and the code for better readability, but the analysis, for the most part, remains the same.

I extracted the most notable findings from this analysis and have summarized them here - for a more in-depth analysis, please take a look at the [.ipynb file](Analyzing_IMDb_top_250.ipynb).

## Dataset
https://www.kaggle.com/datasets/mustafacicek/imdb-top-250-lists-1996-2020

The dataset was compiled and uploaded by Mustafa Cicek from Kaggle, and the information was scraped from IMDb using Beautiful Soup. The data consists of the IMDb top 250 list for every year from 1996 to 2020. As the top 250 list is constantly changing, the data from each year was extracted at midnight of December 31st (PST) to reflect the final rankings of that year.

There are 6250 rows (25 years of top 250 movies) and 16 features:

>* **Ranking**: The ranking of a movie in a given year<br>
>* **IMDbyear**: The year for a given Top 250 list<br>
>* **IMDBlink**: The movie's IMDb url<br>
>* **Title**: Title of the movie<br>
>* **Date**: Movie's release date<br>
>* **RunTime**: Total runtime of the movie (in minutes)<br>
>* **Genre**: Different genres the movie flls under<br>
>* **Rating**: The IMDb score of the movie - a weighted average of IMDb user ratings<br>
>* **Score**: The metascore of the movie - a weighted average of professional critical reviews<br>
>* **Votes**: Total # of votes on IMDb for a movie<br>
>* **Gross**: Total gross of a movie (in millions of dollars)<br>
>* **Director**: Director of the movie<br>
>* **Cast1**: Performer 1 for a movie<br>
>   * The Cast data takes the first four cast members listed on the site which is, itself, based on the film credits order. Thus, Cast1 should represent the main leading actor/actress.
>* **Cast2**: Performer 2 for a movie<br>
>* **Cast3**: Performer 3 for a movie<br>
>* **Cast4**: Performer 4 for a movie

# Analysis



## Feature Overview

First I examined overall statistics regarding the data which would help inform and guide the analysis:
<p align="center">
    <img src="images\nunique_data.PNG" height="300"><br>
    <img src="images\describe_data.PNG" height="275"><br>
    <em>Descriptive statistics about the numeric columns</em>
</p>
<br>

Out of 6250 rows, only the score and gross columns had missing values, which implies they may be less reliable indicators of movie rankings. Many of the columns also had enormous ranges - in particular, the votes and gross columns which ranged from a few thousand to millions of votes and $10,000 to hundreds of millions respectively. Interestingly, both columns had opposite skews - the average movie had around 460,000 votes at the upper end of the votes range and earned 70 million USD at the lower end of the gross range. This result implies that while most movies on the list are generally popular, they were not necessarily box-office record-breakers.

Out of the 4 cast columns, Cast1 contained 477 unique values while the others contained over 600. This makes sense since Cast1 represents the top-billed actor/actress who would be more likely to play prominent roles in multiple movies. 

The most notable statistics, however, involved the movie release dates (the Date column). The average movie in the dataset was released in 1978, and the 75% percentile is 1998. Considering the dataset contains the top 250 list from 1996 to 2020, this implies that the list is fairly consistently dominated by older movies. Looking at the number of unique values in the dataset, this explains why there are only 732 unique movies out of a possible 6250.

Plotting the percentage of new movies for each year's list shows a slow downward trend, and it's important to note that <b>all</b> percentages are already very low (under 10%!). The calculated correlation coefficient is -0.61, reflecting a weak negative correlation between the variables.

<p align="center">
    <img src="images\perc_new.PNG" height="250"><br>
</p>
<br>

## Genres

There are 23 different genres represented in the dataset, but most movies are associated with multiple genres - the most common is Drama with 4309 movies while the least common is Documentary with only 2 movies.

<p align="center">
    <img src="images\top10_bot10_genres.PNG" height="600"><br>
    <em>The top 10 and bottom 10 movie genres represented in the dataset</em>
</p>
<br>

Taking a look specifically at the top 50 movies of each year, dramas (but more specifically, crime dramas) remain the most common movie on the list - the next most common grouping are action adventures, though they are included in less genre groupings than crime dramas. These top four individual genres (dramas, crime, action, adventure) from the top 10s also rank very highly in the top 250 overall, meaning they are both common and popular genres. Notably, while comedies are 4th in the top 250 category, they are 7th in the top 10 category, implying that while comedies are common, they are less likely to achieve high rankings on the list.

<p align="center">
    <img src="images\genre_breakdown.PNG" height="350"><br>
    <em><b>Left</b>: The most common genre groupings of the top 10 movies</em><br>
    <em><b>Right</b>: The most common individual genres of the top 10 movies</em>
</p>
<br>

## Gross Earnings

The average movie on the IMDb top 250 list grossed $70.19 million. That being said, as pointed out earlier, most movies earned under $100 million (with the 75th percentile at $90 million). Some movies well exceeded that threshold, with the highest grossing movie on the list being *Star Wars: Episode VIII - The Force Awakens* earning a whopping $960 million.

To determine if gross earnings correlated with rankings, I categorized each movie into bins according to their gross earnings and plotted the corresponding best ranking on the list with a log scale on the x-axis to account for the data skew toward very small gross values:

**Bins**
1. Less than $1 million
2. $1 million - $5 million
3. $5 million - $10 million
4. $10 million - $50 million
5. $50 million - $100 million
6. $100 million - $250 million
7. Greater than $250 million

<p align="center">
    <img src="images\gross_vs_rank.PNG" height="500"><br>
    <em>The best ranking of each movie plotted against its total gross earnings along with the mean value for each bin</em><br>
</p>
<br>

As seen in the chart above, the more money that a movie earned at the box office, the better the rank it achieved on the list, though this trend is much more apparent at higher gross values. This makes sense considering gross earnings is an analogue to ticket sales - higher earnings most likely means more people watched the movie

## Rating and Score

These two categories are closely related, but fundamentally different - the rating value is computed as the weighted average of IMDb user ratings while the score value is computed as the weighted average of professional critical reviews. In other words, the rating value is determined by the larger public while the score value is determined by a select few.

When plotting movie rankings by rating and score, neither are particularly strong indicators, but higher ratings do appear to correlate with better rankings while scores appear to have no correlation at all. The rating variable has a correlation coefficient of -0.53 while score has a much more neutral coefficient of -0.28.

<p align="center">
    <img src="images\rank_ratingvsscore.PNG" height="600"><br>
</p>
<br>

### Votes

This category is related to the Rating category in that it represents the number of ratings that a movie received (as opposed). When plotting movie ranks by number of votes, the scatter plot looks similar to the Rating plot, but with a higher concentration of movies across all rankings with almost 0 votes. Still, the correlation coefficient is -0.47, confirming a weak negative correlation. 

<p align="center">
    <img src="images\votes_vs_rank.PNG" height="500"><br>
</p>
<br>

# Actors and Directors

For these analyses, I explored both the full dataset and a subset that only included movies in the top 50 rankings. For the actors category, I further subdivided the set into either an aggregated Cast category or only the Cast1 subset.

Out of all movies in the dataset, only 3 actors had worked on at least 10 - Robert De Niro (15), Leonardo DiCaprio (10), and Tom Hanks (10). When limiting the dataset to only Cast1 values, Robert De Niro was still at the top with 10 movies total. However, when limiting the dataset to only the top 50 ranked movies for each year, Tom Hanks outranked Robert De Niro with 7 movies overall where he is credited as the top-billed actor.

<p align="center">
    <img src="images\actor_combined.PNG" height = "800"><br>
</p>
<br>

Overall, the director with the most movies on the top 250 list was Alfred Hitchcock (13), while Stanley Kubrick and Steven Spielberg both ranked 2nd with 10 each. However, when limiting the dataset to only the top 50 ranked movies for each year, Christopher Nolan had the most movies (6) while Alfred Hitchcock fell to 3rd place with only 4.

<p align="center">
    <img src="images\dir_combined.PNG" height="500"><br>
</p>
<br>


# Machine Learning

I wanted to test if the information in the dataset was enough to accurately predict a movie's rankings. As one of my first machine learning projects, I kept the implementation relatively simple. However, I still had to heavily preprocess the data in order to get working models:

> Title, IMDBlink, IMDByear: These columns were deleted as they acted like "ID" variables<br>
> Age column was created to reflect the number of years that a movie was released before it ranked on the charts<br>
> Score, Gross, Ratings, and Votes were all scaled to be on a range from -1 to 1<br>
> The Genre column was split up and one-hot-encoded<br>
> The Director and Cast columns were all binary-encoded

The resulting preprocessed data frame contained 79 numeric columns that was ready for modeling.

I used the following models to fit the data:

* Decision tree: This model is fairly simple and could possibly output a useful decision tree
    * Model: tree.DecisionTreeClassifier(random_state = 15)
* Random forests: This ensemble method should be fast and accurate, but won't provide a decision tree
    * Model: RandomForestClassifier(random_state = 15)
* SVM: This is a powerful classification method, but it may be hard to tune and may take longer to execute
    * Model: SVC(kernel = 'rbf', gamma = .0075, C=1000, random_state = seed)

As there are 250 ranks on the list, I first ran a decision tree model with all 250 classes to see how well it would fit the data. As expected, the model performed very poorly with only a 6.93% accuracy.

To address this, I decided to first categorize the classes into bins to reduce the number of classes entirely. Not only would this improve model performance, but it also makes sense considering a movie classifier that predicts 250 ranks is not that useful in real life. For example, if I wanted to create a movie recommender, I would want it to provide subsets of movie recommendations, not just individual recommendations. 

To bin the data, I needed to decide what I wanted to model. I decided on two approaches:

* 5 sets of equal width bins
    * The data was grouped into 5 bins, each containing 50 ranks
    * 1-50, 51-100, 101-150, 151-200, 201-250
        * This would be an unbiased grouping method
* 4 sets of progressively larger bins
    * The data was grouped into 4 bins, each containing larger numbers of ranks
    * 1-10, 11-50, 51-100, 101-250
        * This would act similarly to the recommender example I described above. I'm more interested in a model that can predict the top 10, but it would be interesting to see how well it can predict the other groupings as well.

### Results:

<p align="center">
    <img src="images\5bins_conmats.PNG" height = "400">
    <img src="images\4bins_conmats.PNG" height = "400"><br>
    <em><b>Left</b>: The confusion matrices from the 5 sets of equal-width-bins data<br>
    <b>Right</b>: The confusion matrices from the 4 sets of progressively larger bins data</em><br><br>
    <img src="images\5bins_results.PNG" height = "600">
    <img src="images\4bins_results.PNG" height = "600"><br>
    <em><b>Left</b>: The results from the 5 sets of equal-width-bins data<br>
    <b>Right</b>: The results from the 4 sets of progressively larger bins data</em><br>
</p>
<br>

Overall, both methods of binning resulted in more accurate models, but the 5 bins of equal size method (5 bins) performed worse than the 4 bins of progressively larger size method (4 bins). For both methods, the confusion matrices showed that the model had a hard time correctly predicting outside of the top 50, but it was more pronounced for the 5 bin method. The 5 bin method also performed worse on all other metrics.

Between the individual models, the random forests model performed the best on most categories. It had the highest accuracy, precision, and recall, though the decision tree was significantly faster by a factor of 10. However, the SVM model was the slowest by a factor of 100, and it still performed worse than the random forests method. 

Also, while the decision tree method executed the fastest, it performed worse across the other metrics. It also provided an extremely large tree which would be hard to parse through, though this is due to the complexity of the data:

<p align="center">
    <img src="images\dtree_top10.PNG" height="500"><br>
</p>
<br>

For this dataset and my needs, the random forests method outperformed the other models. Were I to repeat this analysis on a much larger dataset, however, the decision tree method may still be useful as it performed quickly without resulting in an unusable model.