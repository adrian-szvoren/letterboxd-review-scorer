import math
import streamlit as st

from score import score


if __name__ == '__main__':
    st.set_page_config(page_title='Letterscord', page_icon=':star:')

    st.title('Letterscord')
    st.text('[letterboxd-review-scorer]')

    text_input = st.text_area(
        'Review text area',
        placeholder='Write your review here.',
        label_visibility='collapsed'
    )

    platform = st.selectbox(
        'Platform selection',
        [
            '',
            'Letterboxd (0.5 - 5)',
            'IMDB (1 - 10)',
            'Rotten Tomatoes (0 - 100%)'
        ],
        label_visibility='collapsed'
    )

    score = score(text_input)[0]

    score_verbal = ''
    if 'Letterboxd' in platform:
        score_lb = math.ceil(10 * score)/2
        stars = '★' * math.floor(score_lb)
        if score_lb % 1 == 0.5:
            stars += '½'
        stars += '☆' * math.floor(5-score_lb)
        score_verbal = f'{stars} ({score_lb})'
    elif 'IMDB' in platform:
        score_imdb = math.ceil(10 * score)
        score_verbal = f':star: **{score_imdb}**/10'
    elif 'Rotten Tomatoes' in platform:
        score_rt = round(100 * score)
        if score_rt >= 60:
            score_rt_verbal = ':tomato:'
        else:
            score_rt_verbal = ':microbe:'
        score_verbal = f'{score_rt_verbal} {score_rt}%'
    st.subheader(f'Movie rating: {score_verbal}')
