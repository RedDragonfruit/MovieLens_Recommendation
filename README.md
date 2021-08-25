# MovieLens Reccomendation

**Authors**: Tony Bai, Eddie Lin, Douglas Lu, Shane Mangold

## Overview

Our Group was tasked to build and explore a recommendation system based on the MovieLens dataset. In this project we explore Content-Based and Collaborative Recommendation systems built on 100,000 user movie ratings.

Most internet products we use today are powered by recommender systems. YouTube, Netflix, Amazon, Pinterest, and long list of other internet products all rely on recommender systems to filter millions of contents and make personalized recommendations to their users. Recommender systems are well-studied and proven to provide tremendous values to internet businesses and their consumers.

There are majorly six types of recommender systems which work primarily in the Media and Entertainment industry:

* Collaborative Recommender system
* Content-based recommender system
* Knowledge based recommender system
* Hybrid recommender system
* Demographic based recommender system
* Utility based recommender system

## Business Problem

We have been tasked to improve on an existing recommendation system. We will explore the traditional Recommendation systems and attempt to build a hybrid model that will use multiple recommendation systems in order to provide imporoved reccomendations.

## Data

MovieLens 100K dataset has been used for this project. MovieLens is a rating dataset from the MovieLens website, which has been collected over some period. Stable benchmark dataset. 100,000 ratings from 1000 users on 1700 movies. Released on 4/1998. Further information regarding this dataset can be found here.

A little about the dataset:

MovieLens data sets were collected by the GroupLens Research Project at the University of Minnesota.

This data set consists of:

* 100,000 ratings (1-5) from 943 users on 1682 movies.
* Each user has rated at least 20 movies.
* Simple demographic info for the users (age, gender, occupation, zip)

## Methods

### Content Based Filtering

We first explored content based filtering using TF-IDF vectorizer to calcualte the distance between the data points. We chose to use cosine similarity to measure the distance between these vectors. **Cosine** similarity is a measure of similarity between two non-zero vectors of an inner product space that measures the cosine of the angle between them. Given two vectors of attributes, A and B, the cosine similarity, cos(θ), is represented using a dot product and magnitude as Inline-style. From the TF-IDF vectorizer we would then return a list of movies that are most cosine similar.

### Collaborative Filtering

Item-Item:

Item-item collaborative filtering, or item-based, or item-to-item, is a form of collaborative filtering for recommender systems based on the similarity between items calculated using people's ratings of those items. In this instance, we are trying to find movie that are similar to each other based on people's ratings of those items (in this case movies). This algorithm takes far less resources and time than user-user due to its fixed number of movies and also to the fact that we dont require a similarity score between all users. 

We used Item to Item to constuct a function where we created a similarity matrix of top rated movies of the selected user. From there we get the pairwise distance between the selected movie and all other movies. The function then returns a unique list of movies that are most similar (closest pairwise distance from those movies)



## Results

Present your key results. For Phase 1, this will be findings from your descriptive analysis.

***
Questions to consider:
* How do you interpret the results?
* How confident are you that your results would generalize beyond the data you have?
***

Here is an example of how to embed images from your sub-folder:

### Visual 1
![graph1](./images/viz1.png)

## Conclusions

Provide your conclusions about the work you've done, including any limitations or next steps.

***
Questions to consider:
* What would you recommend the business do as a result of this work?
* What are some reasons why your analysis might not fully solve the business problem?
* What else could you do in the future to improve this project?
***

## For More Information

Please review our full analysis in [our Jupyter Notebook](./dsc-phase1-project-template.ipynb) or our [presentation](./DS_Project_Presentation.pdf).

For any additional questions, please contact **name & email, name & email**

## Repository Structure

Describe the structure of your repository and its contents, for example:

```
├── __init__.py                         <- .py file that signals to python these folders contain packages
├── README.md                           <- The top-level README for reviewers of this project
├── dsc-phase1-project-template.ipynb   <- Narrative documentation of analysis in Jupyter notebook
├── DS_Project_Presentation.pdf         <- PDF version of project presentation
├── code
│   ├── __init__.py                     <- .py file that signals to python these folders contain packages
│   ├── visualizations.py               <- .py script to create finalized versions of visuals for project
│   ├── data_preparation.py             <- .py script used to pre-process and clean data
│   └── eda_notebook.ipynb              <- Notebook containing data exploration
├── data                                <- Both sourced externally and generated from code
└── images                              <- Both sourced externally and generated from code
```
