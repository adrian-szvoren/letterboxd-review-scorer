import altair as alt
import configparser
import math
import streamlit as st

from predict import load_models, predict_score


@st.cache_resource
def st_load_models(config):
    return load_models(config)


def scale_scores(scores, platform):
    if 'Letterboxd' in platform:
        scores['score'] = scores['score'] / 2
    elif 'IMDB' in platform:
        pass
    elif 'Rotten Tomatoes' in platform:
        scores['score'] = scores['score'] * 10
    else:
        raise RuntimeError

    scores['confidence'] = round(100 * scores['confidence'], 2)
    return scores


if __name__ == '__main__':
    st.set_page_config(page_title='Letterscord', page_icon=':star:')
    config = configparser.ConfigParser()
    config.read('config.ini')

    st.title('Letterscord')
    st.text('[letterboxd-review-scorer]')

    config_dict = {s: dict(config.items(s)) for s in config.sections()}
    model, tokenizer, vocab, device, min_length, pad_index = st_load_models(config_dict)

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

    scores = None
    try:
        scores = predict_score(text_input, model, tokenizer, vocab, device, min_length, pad_index)
        scores = scale_scores(scores, platform)
        top = scores.iloc[scores['confidence'].idxmax()]

        score = top['score']
        score_verbal = ''
        if 'Letterboxd' in platform:
            stars = '★' * math.floor(score)
            if score % 1 == 0.5:
                stars += '½'
            stars += '☆' * math.floor(5 - score)
            score_verbal = f'{stars} ({score})'
        elif 'IMDB' in platform:
            score_verbal = f':star: **{int(score)}**/10'
        elif 'Rotten Tomatoes' in platform:
            if score >= 60:
                score_rt_verbal = ':tomato:'
            else:
                score_rt_verbal = ':microbe:'
            score_verbal = f'{score_rt_verbal} {score}%'
        else:
            raise RuntimeError
        st.subheader(f'Movie score: {score_verbal}')

        with st.expander('Score confidence details'):
            plot = alt.Chart(scores).mark_bar().encode(
                x=alt.X('score:O', axis=alt.Axis(labelAngle=0)),
                y=alt.Y('confidence:Q', title='confidence (%)'),
                color=alt.condition(
                    alt.FieldOneOfPredicate('score', [score]),
                    alt.value('#FCC201'),
                    alt.value('#14171C')
                )
            )
            st.altair_chart(plot, use_container_width=True)
    except RuntimeError or TypeError:
        pass
