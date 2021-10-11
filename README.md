# Machine Learning Project

Group project by *Anne-Laure Ehresmann*, *Aria Adibi*, *Javier Fernández-Rial Portela*

## DATA
The datasets are expected to be in the format as given in this repository. Any modification to the datasets may affect the reproducibility of this project.

For the project to properly load and preprocess the datasets, you will need to first modify the file "path_to_data.txt", you should write the absolute path of the "data" folder.
Because loading and preprocessing the datasets the first time take a long time, we provide a *save_datasets* and *load datasets*, which lets you pickle a set of datasets once you load them for the first time. In its current state, the driver is designed to look for two such pickle files in the "pickled_datasets" folder, in the data folder. If they exist, it will use them directly, otherwise it will create them and save them accordingly.

## MODELS
We provide two main functions for running the models on the datasets:

*fit_and_save_models*:
    Given a set of datasets, algorithms, a random_state, and folder name, runs each of the algorithms on each of the datasets, and saves the trained model in the given folder. In its current state, the driver is designed to run all the classification algorithms on all the classification datasets, and save them in a folder marked "models", in the "data" directory.

*load_models_and_predict*:
    Given a set of datasets, algorithms, a random_state, and a folder name, for each algorithm, for each dataset, look for a model located in the given folder, trained on that specific dataset, for that specific algorithm, and run it on the test set (whose split should be the same as the one obtained during training, given the same random_state)
    In its current state, the driver is designed to, for all datasets, for all algorithms, fetch all best trained models in the "best_models" folder, located in the "data" directory.

## GRAPHS
