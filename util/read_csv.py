def read_encoded_csv(csv_path):

    import numpy as np

    print('[LOG] Reading file: ' + csv_path)
    data_file = open(csv_path, "r")
    features = data_file.readline().split(',')
    print(f'[LOG] Number of features in "{csv_path}":', len(features))
    data = []
    for row in data_file:
        row = row.replace('\n', '').split(',')
        float_row = []
        for x in row:
            float_row.append(float(x))
        data.append(float_row)
    data = np.array(data)
    return features, data