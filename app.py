import streamlit as st
import pickle
import pandas as pd
import requests
import gc
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity



@st.cache(allow_output_mutation=True)
def load_data():
    movies=pickle.load(open('movies.pkl','rb'))
    cv = CountVectorizer(max_features=5000,stop_words='english')
    matrix = cv.fit_transform(movies["new_features"])
    cosine_sim = cosine_similarity(matrix)
    return movies, cosine_sim, pickle.load(open('review_1.pkl','rb')), pickle.load(open('review_2.pkl','rb')), pickle.load(open('IMDB_id.pkl','rb')), pickle.load(open('sorted_name.pkl','rb'))

def get_url(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=a65e4b33e800212c365ef0c02aa2d6d8&language=en-US".format(movie_id)
    data = requests.get(url).json()
    if 'poster_path' not in data:
        return ""
    poster_path = str(data['poster_path'])
    if poster_path.endswith(".jpg"):
        return "https://image.tmdb.org/t/p/w500/" + poster_path
    else:
        return ""

def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    contain=[]
    contain.append(movie)
    for i in distances:
        movie_id = movies.iloc[i[0]].id
        movie_name=movies.iloc[i[0]].title
        url=get_url(movie_id)
        if movie_name in contain or url=="":
            continue
        # fetch the movie poster
        else:
            recommended_movie_posters.append(url)
            recommended_movie_names.append(movie_name)
            contain.append(movie_name)
            if len(recommended_movie_names)==6:
                break

    return recommended_movie_names,recommended_movie_posters


def hide():
    hide_img_fs = '''
    <style>
    button[title="View fullscreen"]{
        visibility: hidden;}
    </style>
    '''
    st.markdown(hide_img_fs, unsafe_allow_html=True)

if "button_clicked" not in st.session_state:
    st.session_state.button_clicked=False

def callback():
    st.session_state.button_clicked=True

def show_story(movie): 
    index= movies[movies['title'] == movie].index[0]  
    text='Story: '+str(movies.iloc[index].overview)
    st.write(text)

def show_info(index):
    data=movies.iloc[index]
    director, date, runtime, rate, genres = ['']*5
    
    if len(data.director)>0:
        director=data.director[0]
    date=str(data.release_date)
    runtime=str(int(data.runtime))
    rate=str(data.vote_average)
    for genre_type in data.genres:
        genres=genres+ genre_type+', '
    genres=genres[:-2]

    return director, date, runtime, rate, genres

def get_trailer(movie_id):
    url="https://api.themoviedb.org/3/movie/{}/videos?api_key=a65e4b33e800212c365ef0c02aa2d6d8&language=en-US".format(movie_id)
    data = requests.get(url).json()
    if 'results' not in data:
        return ''
    if len(data['results'])>0 and 'key' in data['results'][0]:
        suffix = data['results'][0]['key']
        url="https://www.youtube.com/watch?v="+suffix
        return url
    else:
        return ''

def homepage(movie):
    homepage_link=imdb_id[movie].to_string()[5:]
    if homepage_link.startswith('tt'):
        return "https://www.imdb.com/title/"+homepage_link+ "/?ref_=tt_urv"
    else:
        return ''


def show_comment(movie):
    L=[]
    if movie in review_1:
        for h in review_1[movie]:
            if type(h)==float:
                break
            else:
                L.append(h)
    else:
        for h in review_2[movie]:
            if type(h)==float:
                break
            else:
                L.append(h)
    return L


def show_cast_img(movie):
    index = movies[movies['title'] == movie].index[0]  
    movie_id=movies.iloc[index].id
    url='https://api.themoviedb.org/3/movie/{}/credits?api_key=a65e4b33e800212c365ef0c02aa2d6d8&language=en-US'.format(movie_id)
    data = requests.get(url).json()
    L_url=[]
    L_name=[]
    if 'cast' not in data:
        return L_name,L_url
    for cast in data['cast']:
        if type(cast['profile_path'])!=str:
            continue
        url_img="https://image.tmdb.org/t/p/w500"+cast['profile_path']
        L_url.append(url_img)
        L_name.append(cast['name'])
        if len(L_url)==5:
            break
    return L_name,L_url

def sentiment_analysis(review_list):
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    cnt=1
    for reviews in review_list:
        st.write("Comment {}: ".format(cnt))
        encoded_text = tokenizer(reviews, return_tensors='pt')
        output = model(**encoded_text)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        scores_dict = {
            'sentiment: negative' : scores[0],
            'sentiment: neutral' : scores[1],
            'sentiment: positive' : scores[2]
        }
        result = max(scores_dict, key=scores_dict.get)
        st.write(result)
        st.write(reviews)
        cnt+=1



movies, similarity,review_1, review_2, imdb_id, sorted_name= load_data()

st.header('Movie Recommender System')
sorted_name=['select a movie']+list(sorted_name)
selected_movie = st.selectbox(
    "Select a movie from the dropdown",
    sorted_name
)


if selected_movie=='select a movie':
    st.write('please select a movie')
else:
    id=movies.iloc[movies[movies['title'] == selected_movie].index[0]].id
    index= movies[movies['title'] == selected_movie].index[0]  
    url=get_url(id)
    if url!="":
        st.image(url)

    show_story(selected_movie)
    director, date, runtime, rate, genres = show_info(index)
    st.write('Director: '+director)
    st.write('Release date: '+date)
    st.write('Runtime: '+runtime+ ' minutes')
    st.write('Genres: '+genres)
    st.write('IMDB rating: '+rate + '⭐')

    get_trailer_url=get_trailer(id)
    if get_trailer_url=='':
        st.write("Trailer: not available")
    else:
        st.write("Trailer: [link](%s)" % get_trailer_url)

    homepage_link=homepage(selected_movie)
    if homepage_link=='':
        st.write("Homepage: not available")
    else:
        st.write("Homepage: [link](%s)" % homepage_link)

    view_cast=st.button('View cast')
    if view_cast==True:
        castname,casturl=show_cast_img(selected_movie)
        num=len(castname)
        if num==0:
            st.write("not available")
        else:
            _cast=st.columns(num)
            for i in range(0,num):
                with _cast[i]:
                    st.text(castname[i])
                    st.image(casturl[i])
                    hide()

    st.markdown('''<b style='text-align: left; color: #e06666;'> Fetching movie reviews with sentiment analysis will take a time.</b>''',
                    unsafe_allow_html=True)
    show_review=st.button('View comments')
    if show_review==True:
        reviews_list=show_comment(selected_movie)
        if len(reviews_list)==0:
            st.write('No comment available')
        else:
            sentiment_analysis(reviews_list)


    num=st.slider('Number of movies you want Recommended:', 1,5,0)
    recommended_movie_names,recommended_movie_posters = recommend(selected_movie)
    if num>0:
        if num==1:
            st.write("Top {} similar movie to \"".format(num),selected_movie,"\" is:")
        else:
            st.write("Top {} similar movies to \"".format(num),selected_movie,"\" are:")

        _list=st.columns(num)
        sub_botton=[False]*num
        for i in range(0,num):
            with _list[i]:
                st.text(recommended_movie_names[i])
                st.image(recommended_movie_posters[i])
                hide()
                sub_botton[i]=st.button('Overview', key=i)
                if sub_botton[i]:
                    show_story(recommended_movie_names[i])

gc.collect()



