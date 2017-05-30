import pandas as pd


def preprocess_data(data, **kwargs):
    types = [
        'CASH_IN',
        'CASH_OUT',
        'DEBIT',
        'PAYMENT',
        'TRANSFER'
    ]
    data.type.replace(types, range(5), inplace=True)
    y = data['isFraud'].values

    separate_types = False
    col_names = False
    for opt, val in kwargs.items():
        if opt == 'separate_types' and val:
            separate_types = True
        elif opt == 'col_names' and val:
            col_names = True

    if separate_types:
        X = data[data.columns.difference(['isFraud', 'type'])]
        types = data['type'].values
        X_type = []
        y_type = []
        for i in range(5):
            y_type.append(y[types == i])
            if col_names:
                X_type.append(X[types == i])    
            else:
                X_type.append(X[types == i].values)
        return X_type, y_type
    else:
        X = data[data.columns.difference(['isFraud'])]
        if col_names:
            return X, y
        else:
            return X.values, y


def load_data(filename, **kwargs):
    cols = [
        'type',
        'amount',
        'oldbalanceOrg',
        'newbalanceOrig',
        'oldbalanceDest',
        'newbalanceDest',
        'isFraud'
    ]
    data = pd.read_csv(filename, usecols=cols)
    X, y = preprocess_data(data, **kwargs)
    return X, y

