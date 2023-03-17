import math
import streamlit as st

from score import score


if __name__ == '__main__':
    st.set_page_config(page_title='Letterscord', page_icon='img/letterboxd-stars-circle.png', layout='centered')

    st.image('img/letterboxd-stars-title.png', width=350)
    st.text('[letterboxd-review-scorer]')

    input = st.text_area('', placeholder='Write your review here.')
    button = st.button('Calculate')
    if input or button:
        score = score(input)[0]
        score_lb = math.ceil(10*score)/2
        stars = '★' * math.floor(score_lb)
        if score_lb%1 == 0.5:
            stars += '½'
        stars += '☆'*math.floor(5-score_lb)

        st.subheader(f'Movie rating based on the review: {stars} ({score_lb})')

