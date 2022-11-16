import json
import numpy


def create_test_data(path:str):
    """
        read json file and create list data x and y for test

        input:
            path (str) = path to json file

        output:
            x (list) = [ [x0, y0, z0, ... x20, y20, z20], ... ]
            y (list) = [class, ...]
    """
    # load data from json
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    file.close()

    x_data = []
    y_data = []

    for i in data:
        try:
            sourse_key_points = numpy.array(i['sourceKeyPoints'])
        except:
            sourse_key_points = numpy.array(i['SourceKeyPoints'])

        try:
            value_y = i['gesture']
        except:
            value_y = i['Gesture']

        # changes incorrect classes to 0
        if value_y > 12:
            value_y = 0

        y_data.append(value_y)
        x_data.append(sourse_key_points)

    return x_data, y_data
