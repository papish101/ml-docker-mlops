import joblib


def test_model_exists():

    model = joblib.load("app/model.pkl")

    assert model is not None


def test_model_prediction():

    model = joblib.load("app/model.pkl")

    prediction = model.predict([
        [5.1, 3.5, 1.4, 0.2]
    ])

    assert prediction[0] in [0, 1, 2]
