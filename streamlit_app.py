import math
import streamlit as st

from score import score


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


if __name__ == '__main__':
    st.set_page_config(page_title='Letterscord', page_icon='img/letterboxd-stars.png', layout='centered')
    local_css('style.css')

    col1, mid, col2 = st.columns([1, 1, 13])
    with col1:
        st.image('img/letterboxd-stars.png', width=100)
    with col2:
        st.title('Letterscord')

    #st.title('Letterscord')
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

