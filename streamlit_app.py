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
        'Range and platform selection',
        [
            '',
            '0.5 - 5 (Letterboxd)',
            '1 - 10 (IMDB)',
            '0 - 100% (Rotten Tomatoes)'
        ],
        label_visibility='collapsed'
    )

    score = score(text_input)[0]

    if platform != '':
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader(f'Movie rating:')

    if 'Letterboxd' in platform:
        score_lb = math.ceil(10 * score)/2
        stars = '★' * math.floor(score_lb)
        if score_lb % 1 == 0.5:
            stars += '½'
        stars += '☆'*math.floor(5-score_lb)
        with col2:
            st.subheader(f'{stars}')
    elif 'IMDB' in platform:
        score_imdb = math.ceil(10 * score)
        with col2:
            st.subheader(f':star: **{score_imdb}**/10')
    elif 'Rotten Tomatoes' in platform:
        score_rt = round(100 * score)
        if score_rt >= 75:
            score_rt_verbal = 'Certified_Fresh'
        elif score_rt >= 60:
            score_rt_verbal = 'Fresh'
        else:
            score_rt_verbal = 'Rotten'
        with col2:
            col21, col22 = st.columns([1, 7])
            with col21:
                st.image(f'img/RT_{score_rt_verbal}.png', width=50)
            with col22:
                st.subheader(f'{score_rt}%')

