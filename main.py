from collections import Counter

from lib.smote_data import load_data


def main():
    X, y = load_data('smote_data.npz', 'data.csv')
    print(Counter(y))


if __name__ == '__main__':
    main()

