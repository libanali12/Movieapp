#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
import numpy as np
import matplotlib as plt 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics.pairwise import cosine_similarity
from itertools import permutations
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score 
from scipy.spatial.distance import pdist, squareform
import sklearn.metrics.pairwise as pw
from scipy import sparse
from sklearn.metrics.pairwise import pairwise_distances


# In[33]:


url = 'https://raw.githubusercontent.com/libanali12/Movieapp/main/user_ratings.csv'
user_ratings= pd.read_csv(url)


# # Exploratory Data Analysis

# In[34]:


user3 = user_ratings.copy()


# In[35]:


#creating list with unique genres
genres = list(set('|'.join(list(user3["genres"].unique())).split('|')))
genres.remove('(no genres listed)')

#Creating dummy columns for each genre
for genre in genres:
    user3[genre] = user3['genres'].map(lambda val: 1 if genre in val else 0)


# In[36]:


genres=['Animation', 'Documentary', 'Western', 'Children', 'War', 'Horror',
       'Action', 'Romance', 'Crime', 'Comedy', 'Thriller', 'Adventure',
       'Musical', 'Film-Noir', 'Mystery', 'Drama', 'Sci-Fi', 'Fantasy',
       'IMAX']


# In[37]:


genere_counts = user3.loc[:,genres].sum().sort_values(ascending=False)


# In[38]:


user3['year'] = user3.title.str.extract("\((\d{4})\)", expand=True)


# In[39]:


yearly_release_counts = user3.groupby(user3['year']).size().sort_values(ascending=False)


# In[40]:


rating_counts = user3.groupby(user3['rating']).size().sort_values(ascending=False)


# # Simple Reccomendation 

# In[41]:


Average_ratings = pd.DataFrame(user_ratings.groupby('title')['rating'].mean())
Average_ratings['Total Ratings'] = pd.DataFrame(user_ratings.groupby('title')['rating'].count())


# In[42]:


C= Average_ratings['rating'].mean()
m= Average_ratings['Total Ratings'].quantile(0.9)
q_movies = Average_ratings.copy().loc[Average_ratings['Total Ratings'] >= m]


# In[43]:


def weighted_rating(x, m=m, C=C):
    v = x['Total Ratings']
    R = x['rating']
    return (v/(v+m) * R) + (m/(m+v) * C)


# In[44]:


q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
q_movies = q_movies.sort_values('score', ascending=False).reset_index()


# In[45]:


pop_movies = q_movies[['title','score']].head(16)


# # Collaborative Based Recommendation

# In[46]:


user_ratings1=user_ratings.copy()


# In[47]:


#creating list with unique genres
genres = list(set('|'.join(list(user_ratings1["genres"].unique())).split('|')))
genres.remove('(no genres listed)')

#Creating dummy columns for each genre
for genre in genres:
    user_ratings1[genre] = user_ratings1['genres'].map(lambda val: 1 if genre in val else 0)


# In[48]:


user_ratings1.drop('genres', axis=1,inplace= True)


# In[49]:


categories = ['Film-Noir', 'Adventure', 'Children',
           'IMAX', 'Crime', 'Documentary', 'Fantasy', 'Musical', 'Romance',
           'Mystery', 'Thriller', 'Animation', 'Action', 'Comedy', 'War', 'Drama',
           'Western', 'Sci-Fi', 'Horror']


# In[51]:


@st.cache
def item_based_recom(input_dataframe,title):    
    pivot_item_based = pd.pivot_table(input_dataframe,
                                      index='title',
                                      columns=['userId'], values='rating')  
    sparse_pivot = sparse.csr_matrix(pivot_item_based.fillna(0))
    recommender = pw.cosine_similarity(sparse_pivot)
    recommender_df = pd.DataFrame(recommender, 
                                  columns=pivot_item_based.index,
                                  index=pivot_item_based.index)
    ## Item Rating Based Cosine Similarity
    cosine_df = pd.DataFrame(recommender_df[title].sort_values(ascending=False))
    cosine_df.reset_index(level=0, inplace=True)
    cosine_df.columns = ['title','cosine_sim']
    return cosine_df


# In[52]:


@st.cache
def pairwise_row_diff(dataframe,row1, row2,column_names):
#     display(dataframe)
     # Creates 2 Matrix to compare cosine similarity
    matrix_row1 = [[dataframe.loc[row1,cat] for cat in column_names]] 
    matrix_row2 = [[dataframe.loc[row2,cat] for cat in column_names]] 
    return round(pw.cosine_similarity(matrix_row1,matrix_row2)[0][0],5)


# In[53]:


@st.cache
def item_and_genre_based_recom(cosine_df,movies_df,categories,pairwise_row_diff=pairwise_row_diff):    
## Item Rating and Gender Based Cosine Similarity
    top_cos_genre = pd.merge(cosine_df, user1, on='title')
    # Creating column with genre cosine similarity
    top_cos_genre['genre_similarity'] = [pairwise_row_diff(top_cos_genre,0,row,categories) 
                                          for row in top_cos_genre.index.values]
    return top_cos_genre[['title','cosine_sim','genre_similarity']]


# In[54]:


user1 = user_ratings1.groupby(['title']).sum().reset_index()


# In[55]:


term = user_ratings['title'].unique()


# In[56]:


top_results=10


# # Content Based Recommender

# In[57]:


@st.cache
def get_recommendation(df):
    cross_df=pd.crosstab(df['title'],df['genres'])
    jaccard_distance = pdist(cross_df.values,metric='cosine')
    square_jaccard_distance = squareform(jaccard_distance)
    jaccard_similarity_array = 1 - square_jaccard_distance
    distance_df= pd.DataFrame(jaccard_similarity_array,
                         index=cross_df.index,
                         columns=cross_df.index)
    return distance_df


# In[58]:


def main():
    st.title("Movie Recommendation App")
    menu = ["Home","Simple Recommender","Collaborative Based Recommendation","Content Based Recommender"]
    choice = st.sidebar.selectbox("Menu",menu)
    if choice == "Home":
        st.subheader("Home")
        st.write('Movie Dataset')
        st.dataframe(user_ratings.head(5))
        st.subheader("Exploratory Data Analysis")
        fig1 = px.bar(genere_counts, x=genere_counts.values, y=genere_counts.index,
             color_discrete_sequence=px.colors.diverging.Geyser,
             height=600, width=900)
        fig1.update_layout(template="plotly_white",xaxis_showgrid=False,
                  yaxis_showgrid=False)
        fig1.update_traces( marker_line_color='rgb(8,48,107)',
                  marker_line_width=2, opacity=0.6)
        fig1.update_layout(showlegend=False, title="Genres Distribution",
                  xaxis_title="Count",
                  yaxis_title="Genres")
        fig1.update_xaxes(showline=True, linewidth=1, linecolor='black')
        fig1.update_yaxes(showline=True, linewidth=1, linecolor='black')
        st.plotly_chart(fig1)
        fig2 = px.bar(yearly_release_counts, x=yearly_release_counts.values, y=yearly_release_counts.index,
             color_discrete_sequence=px.colors.sequential.Turbo,
             height=600, width=900)
        fig2.update_layout(template="plotly_white",xaxis_showgrid=False,
                  yaxis_showgrid=False)
        fig2.update_traces( marker_line_color='rgb(8,48,107)',
                  marker_line_width=2, opacity=0.6)
        fig2.update_layout(showlegend=False, title="Genres Distribution",
                  xaxis_title="Release Year",
                  yaxis_title="Movie Release Years")
        fig2.update_xaxes(showline=True, linewidth=1, linecolor='black')
        fig2.update_yaxes(showline=True, linewidth=1, linecolor='black')
        st.plotly_chart(fig2)
        fig3 = px.bar(rating_counts, x=rating_counts.index, y=rating_counts.values,
             color_discrete_sequence=px.colors.sequential.Viridis,
             height=600, width=900)
        fig3.update_layout(template="plotly_white",xaxis_showgrid=False,
                  yaxis_showgrid=False)
        fig3.update_traces( marker_line_color='rgb(8,48,107)',
                  marker_line_width=2, opacity=0.6)
        fig3.update_layout(showlegend=False, title="Distribution Of Movie Rating",
                  xaxis_title="Rating",
                  yaxis_title="Count")
        fig3.update_xaxes(showline=True, linewidth=1, linecolor='black')
        fig3.update_yaxes(showline=True, linewidth=1, linecolor='black')
        st.plotly_chart(fig3)
    elif choice == "Simple Recommender":
        st.subheader("Simple Recommender")
        fig = px.bar(pop_movies, x='score', y='title', color='title',
             color_discrete_sequence=px.colors.diverging.Geyser,
             height=600, width=900)
        fig.update_layout(template="plotly_white",xaxis_showgrid=False,
                  yaxis_showgrid=False)
        fig.update_traces( marker_line_color='rgb(8,48,107)',
                  marker_line_width=2, opacity=0.6)
        fig.update_layout(showlegend=False, title="Rating",
                  xaxis_title="Rating out of 5",
                  yaxis_title="Movie Title")
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black')
        st.plotly_chart(fig)
        st.dataframe(q_movies)
    elif choice == "Collaborative Based Recommendation":
        st.subheader("Collaborative Based Recommendation")
        search_term = st.selectbox(label = "Choose a movie you have seen", options = term)
        if st.button("Recommend"):
            content=item_and_genre_based_recom(item_based_recom(user_ratings1,search_term),user1,categories)           .sort_values('cosine_sim',ascending=False)            .sort_values('genre_similarity',ascending=False)[:top_results]
            final_content= content.sort_values('cosine_sim',ascending=False).reset_index(drop=True)
            st.write('Top 10 Most Similar Movies Based On Your Choice')
            st.write(final_content['title'])
    else:
        st.subheader("Content Based Recommender")
        search_term = st.selectbox(label = "Choose a movie you have seen", options = term)
        if st.button("Recommend"):
            content1= get_recommendation(user_ratings)
            result= content1[search_term].sort_values(ascending=False)
            final_result= result[result>0]
            st.write('Top Most Similar Movies Based On Your Choice')
            st.write(final_result.index[1:11])
            
if __name__ == '__main__':
    main() 


# In[ ]:




