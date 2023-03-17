import math
import streamlit as st

from score import score


if __name__ == '__main__':
    st.set_page_config(page_title='Letterscord', page_icon=':star:')

    # st.image('img/letterboxd-stars-title.png', width=350)
    st.title('Letterscord')
    st.text('[letterboxd-review-scorer]')

    with st.sidebar:
        st.subheader('Settings')
        platform_mapping = {
            'Letterboxd': '0.5 - 5 (Letterboxd)',
            'IMDB': '1 - 10 (IMDB)',
            'Rotten Tomatoes': '0 - 100% (Rotten Tomatoes)'
        }
        platform = st.radio(
            'What is your required score range?',
            ['Letterboxd', 'IMDB', 'Rotten Tomatoes'],
            format_func=lambda x: platform_mapping[x]
        )

    text_input = st.text_area('Review text area', placeholder='Write your review here.', label_visibility='hidden')
    button = st.button('Calculate')
    if text_input or button:
        score = score(text_input)[0]

        if platform == 'Letterboxd':
            score_lb = math.ceil(10 * score)/2
            stars = '★' * math.floor(score_lb)
            if score_lb % 1 == 0.5:
                stars += '½'
            stars += '☆'*math.floor(5-score_lb)
            st.subheader(f'Movie rating: {stars} ({score_lb})')
        elif platform == 'IMDB':
            score_imdb = math.ceil(10 * score)
            st.subheader(f'Movie rating: :star:**{score_imdb}**/10')
        elif platform == 'Rotten Tomatoes':
            score_rt = round(100 * score)
            if score_rt >= 75:
                score_rt_verbal = 'Certified_Fresh'
            elif score_rt >= 60:
                score_rt_verbal = 'Fresh'
            else:
                score_rt_verbal = 'Rotten'
            st.subheader(f'Movie rating: ({score_rt_verbal.replace("_", " ")}) {score_rt}%')
            # st.image(f'img/RT_{score_rt_verbal}.png')

