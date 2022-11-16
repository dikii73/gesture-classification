import sys
import joblib
import utils


def test(path_model:str='model/RFC.sav', path_json:str=''):
    """
        test model from data

        input:
            path_model (str) = path model to test (defaul RandomForestClassifier)
            path_json (str) = path json data
        output:
            accuracy (float) = accuracy model on test

    """

    model = joblib.load(path_model)

    x_test, y_test = utils.create_test_data(path_json)
    return model.score(x_test, y_test) * 100

if __name__ == '__main__':
    accuracy = test(path_json=sys.argv[1])
    print(f"Accuracy: {accuracy:0.2f} %")
