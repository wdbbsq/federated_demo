import pickle


def save_as_file(obj, path):
    file = open(path, 'wb')
    pickle.dump(obj, file)
    file.close()


def read_from_file(filename):
    file = open(filename, 'rb')
    d = pickle.load(file)
    file.close()
    return d
