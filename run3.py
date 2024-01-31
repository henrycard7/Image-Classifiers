import os.path
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from GIST import calculate_gist_descriptors
import shutil
import concurrent.futures


TRAINING_FEATURES_AND_LABELS_PATH = 'training_features_and_labels.csv'
TESTING_FEATURES_PATH = 'testing_features.csv'
BEST_SVM_PARAMS = {'C': 2.0, 'kernel': 'rbf', 'gamma': 'auto'}


def create_training_dataset_process_folder(folder_path: str):
    """Calculate the gist descriptor of every image in a folder
    :param folder_path: path of the folder
    :return: gist descriptor of every image in the folder and their associated label
    """

    label = os.path.basename(folder_path)

    print(f"Started {label}...")
    image_paths = [
        os.path.join(folder_path, imageName)
        for imageName in os.listdir(folder_path)
        if imageName.lower().endswith(('.jpg', '.jpeg'))
    ]
    gists = calculate_gist_descriptors(image_paths)
    print(f"...done {label}")

    return gists, [label] * len(gists)


def create_training_dataset():
    """Generate the gist descriptor of every image in the training set and store it in a csv file with its label"""

    print("Creating training dataset...")

    data = []
    targets = []

    # extract all gist descriptors and targets
    with concurrent.futures.ProcessPoolExecutor() as executor:

        # gets all folder paths in the training directory
        folder_paths = []
        for filename in os.listdir('training'):
            folder_path = os.path.join('training', filename)
            if os.path.isdir(folder_path):
                folder_paths.append(folder_path)

        # split folder processing across cores
        results = [executor.submit(create_training_dataset_process_folder, folder_path) for folder_path in folder_paths]

        # wait for all results
        concurrent.futures.wait(results)

        # combines results
        for result in results:
            gists, labels = result.result()
            data.extend(gists)
            targets.extend(labels)

    print('...extracted features of all images')

    # convert to numpy array
    data = np.array(data)
    targets = np.array(targets)

    # add features to dataframe
    num_of_features = data.shape[1]
    df = pd.DataFrame(data, columns=['Feature ' + str(i) for i in range(num_of_features)])
    print('...stored data in dataframe')

    # add targets to dataframe
    df['targets'] = targets
    print('...stored targets in dataframe')

    # store in a file for later use
    df.to_csv(TRAINING_FEATURES_AND_LABELS_PATH, index=False)
    print('...stored dataset in a file')


def create_testing_dataset_process_images(image_paths: list[str]):
    """Calculate the gist descriptor of every image in a group of images
    :param image_paths: paths of every image in the group
    :return: gist descriptor of every image in the group and their filename
    """

    print("Group started...")
    descriptors = calculate_gist_descriptors(image_paths)
    image_names = [os.path.basename(image_path) for image_path in image_paths]
    print("...done group")

    return descriptors, image_names


def create_testing_dataset():
    """Generate the gist descriptor of every image in the testing set and store it in a csv file with its image name"""

    print("Creating testing dataset...")

    descriptors = []
    image_names = []

    with concurrent.futures.ProcessPoolExecutor() as executor:

        image_paths = [
            os.path.join('testing', filename)
            for filename in os.listdir('testing')
            if filename.lower().endswith(('.jpg', '.jpeg'))
        ]

        # split image paths into groups (of 100) for multiprocessing
        group_size = 100
        grouped_image_paths = [image_paths[i:i + group_size] for i in
                             range(0, len(image_paths), group_size)]

        # send each group of image paths to each core
        results = [executor.submit(create_testing_dataset_process_images, group) for group in grouped_image_paths]

        # wait for all results
        concurrent.futures.wait(results)

        # combine results
        for result in results:
            ds, image_name = result.result()
            descriptors.extend(ds)
            image_names.extend(image_name)

    print('...extracted features of all images')

    # convert from python list to numpy array
    descriptors = np.array(descriptors)

    # add features to dataframe
    num_of_features = descriptors.shape[1]
    df = pd.DataFrame(descriptors, columns=['Feature ' + str(i) for i in range(num_of_features)])
    print('...stored data in dataframe')

    # add image name to dataframe
    df['names'] = image_names
    print('...stored image names in dataframe')

    # store in a file
    df.to_csv(TESTING_FEATURES_PATH, index=False)
    print('...stored dataset in a file')


def find_best_svm_params():
    """Find the best parameters for the SVM classifier"""

    print("Finding best SVM classifier parameters...")

    # parameters to search through
    model_param_grid = {
        'C': np.arange(0.1, 100, 0.1),
        'kernel': ['rbf', 'poly', 'sigmoid'],
        'gamma': ['auto', 'scale', 0.1, 1, 10]
    }

    # load the dataset
    df = pd.read_csv(TRAINING_FEATURES_AND_LABELS_PATH)
    X = np.array(df.drop('targets', axis=1))
    y = np.array(df.targets)

    # standardise the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # find best parameters
    svm = SVC(random_state=69)
    grid_search = GridSearchCV(
        estimator=svm,
        param_grid=model_param_grid,
        cv=10,
        scoring='accuracy',
        verbose=1,
        n_jobs=-1
    )
    grid_search.fit(X, y)
    print("...found best parameters")
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)


def test_model():
    """Use 10-fold cross validation to test the model on the training data"""

    print("Testing model...")

    # load the dataset
    df = pd.read_csv(TRAINING_FEATURES_AND_LABELS_PATH)
    X = np.array(df.drop('targets', axis=1))
    y = np.array(df.targets)

    accuracies = []

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=69)
    for train_index, test_index in kf.split(X, y):

        # split into training and testing sets
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        # standardise the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # train SVM classifier
        svm = SVC(kernel=BEST_SVM_PARAMS['kernel'], C=BEST_SVM_PARAMS['C'], gamma=BEST_SVM_PARAMS['gamma'],
                  random_state=69)
        svm.fit(X_train, y_train)
        y_hat = svm.predict(X_test)

        # calculate accuracy
        accuracy = accuracy_score(y_test, y_hat)
        accuracies.append(accuracy)

    print("...done")

    # print test results
    print('Average accuracy:', np.mean(accuracies))
    print('Highest accuracy:', np.max(accuracies))
    print('Lowest accuracy:', np.min(accuracies))


def predict_testing_dataset():
    """
    Predict the testing dataset an RBF classifier trained with the training dataset.
    The predictions are written to 'run3.txt'
    """

    print("Predicting classes for the testing dataset...")

    # load the datasets
    df_train = pd.read_csv(TRAINING_FEATURES_AND_LABELS_PATH)
    X_train = np.array(df_train.drop('targets', axis=1))
    y_train = np.array(df_train.targets)
    df_test = pd.read_csv(TESTING_FEATURES_PATH)
    X_test = np.array(df_test.drop('names', axis=1))

    # standardise the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # train RBF model
    svm = SVC(kernel=BEST_SVM_PARAMS['kernel'], C=BEST_SVM_PARAMS['C'], gamma=BEST_SVM_PARAMS['gamma'], random_state=69)
    svm.fit(X_train, y_train)

    # predict classes for test data
    y_pred = svm.predict(X_test)
    print("...done")

    # write predictions to file
    with open('run3.txt', 'w') as file:
        for path, prediction in zip(df_test.names, y_pred):
            image_name = os.path.basename(path)
            file.write(image_name + ' ' + prediction + '\n')
    print("...written predictions to 'run3.txt'")


def create_predicted_testing_directory():
    """
    Create a folder directory organised according to the predictions in 'run3.txt'
    :return:
    """

    print("Creating directory according to testing predictions...")

    # make root directory
    shared_directory = 'predicted_testing[run3]'
    os.makedirs(shared_directory)

    # copy images to the folder associated with their predicted class
    with open('run3.txt', 'r') as file:
        for line in file:

            image_name, assigned_class = line.strip().split()

            # create class directory if it doesn't exist
            class_directory = os.path.join(shared_directory, assigned_class)
            if not os.path.exists(class_directory):
                os.makedirs(class_directory)

            # copy image
            shutil.copy2(os.path.join('testing', image_name), class_directory)

    print("...done")


if __name__ == '__main__':

    # create_training_dataset()

    # DON'T UNCOMMENT. IT TAKES AGES TO DO
    # find_best_svm_params()

    # test_model()

    # create_testing_dataset()
    # predict_testing_dataset()
    # create_predicted_testing_directory()

    pass
