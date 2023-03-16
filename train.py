import pickle

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from data import load_data


if __name__ == '__main__':
    data_path = 'data/scale'

    data = load_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(data['clean'], data['rating'], test_size=0.1)

    pipe = make_pipeline(
        TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english')),
        LinearRegression()
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print(f'MAE: {mean_absolute_error(y_test, y_pred)}')
    print(f'MAPE: {mean_absolute_percentage_error(y_test, y_pred)}')
    print(f'RMSE: {mean_squared_error(y_test, y_pred, squared=False)}')

    pickle.dump(pipe, open('model/pipe.pickle', 'wb'))
