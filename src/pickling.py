import pickle

# TODO: is it useful to dump more than just random_state, on top of the model?
# for instance, train_size, score on test set...


def save_model(model_name, dataset_name, is_multilabel, model, random_state, folder):
    with open(folder + "/" + model_name + "_" + dataset_name + ".pickle", "wb") as f:
        pickle.dump((model, is_multilabel, random_state), f)
        f.close()


def load_model(folder, model_name, dataset_name):
    with open(folder + "/" + model_name + "_" + dataset_name + ".pickle", "rb") as f:
        data = pickle.load(f)
        f.close()
    return data[0], data[1], data[2]  # model, is_multilabel, random_state


def save_datasets(folder, datasets, name):
    with open(folder + "/" + name + ".pickle", "wb") as f:
        pickle.dump(datasets, f)
        f.close()


def load_datasets(folder, name):
    with open(folder + "/" + name + ".pickle", "rb") as f:
        data = pickle.load(f)
        f.close()
    return data
