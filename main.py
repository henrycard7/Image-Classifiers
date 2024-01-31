import imageio.v3 as iio
import os.path as path
import json
import joblib
import numpy as np

import run1
import run2
import run3

CATEGORIES = ["bedroom", "Coast", "Forest", "Highway", "industrial", "Insidecity", "kitchen", "livingroom", "Mountain",
              "Office", "OpenCountry", "store", "Street", "Suburb", "TallBuilding"]
TRAINING_IMG_FOLDER = "trainingImages"
TESTING_IMG_FOLDER = "testingImages"
TRAINED_VECTORS_FOLDER = "trainingData_run1"
TRAINED_MODEL_FOLDER = "trainingData_run2"
TRAINING_IMG_PER_CATEGORY = 80
#run1 variables
K = 49
#run2 variables
SAMPLES = 200000
CLUSTERS = 2000


def train_run1():
    for categoryI in range(0, CATEGORIES.__len__()):
        category = CATEGORIES[categoryI]
        fileName = TRAINED_VECTORS_FOLDER + '\\' + category + '.txt'
        if path.isfile(fileName):
            open(fileName, 'w').close()
        with open(fileName, 'a') as file:
            for i in range(0, TRAINING_IMG_PER_CATEGORY):
                image = iio.imread(TRAINING_IMG_FOLDER + '\\' + category + '\\' + str(i) + ".jpg")
                vector = run1.imgToVector(image)
                file.write(str(vector.tolist()))
                file.write('\n')
    print("run1 trained")


def train_run2():
    vectors = []
    joinedVectors = []
    for categoryI in range(0, CATEGORIES.__len__()):
        categoryVectors = []
        category = CATEGORIES[categoryI]
        for i in range(0, TRAINING_IMG_PER_CATEGORY):
            image = iio.imread(TRAINING_IMG_FOLDER + '\\' + category + '\\' + str(i) + ".jpg")
            categoryVectors += run2.extractPatches(image).tolist()
        joinedVectors += categoryVectors
        vectors.append(categoryVectors)
    kmeansModel = run2.kMeans(run2.select(SAMPLES, joinedVectors), CLUSTERS)
    joblib.dump(kmeansModel, TRAINED_MODEL_FOLDER + "\\kmeans_run2.joblib")
    labels = kmeansModel.predict(joinedVectors)
    histograms = run2.createArrays(vectors, labels, CLUSTERS)
    with open(TRAINED_MODEL_FOLDER + "\\Histograms.txt", 'w') as file:
        file.write(str(histograms.tolist()))
    print("run2 trained")


def test_run1():
    imgfile = input("File URI: ")
    image = iio.imread(imgfile)
    allVecs = run1.importVectors()
    result = run1.categorise(image, True, allVecs, K)
    print(result)
    return result


def test_run2():
    image = iio.imread(TRAINING_IMG_FOLDER + "\\TallBuilding\\1.jpg")
    print(do_run2(image))


def do_run1(img: np.ndarray) -> str:
    allVecs = run1.importVectors()
    result = run1.categorise(img, False, allVecs, K)
    return result


def do_run2(img: np.ndarray) -> str:
    with open(TRAINED_MODEL_FOLDER + "\\Histograms.txt", 'r') as file:
        histogramData = json.loads(file.readline())
    inverseFrequencies, adjustedHistograms = run2.getInverseFrequencies(histogramData, CLUSTERS)
    vectors = run2.extractPatches(img)
    kmeansModel = joblib.load(TRAINED_MODEL_FOLDER + "\\kmeans_run2.joblib")
    labels = kmeansModel.predict(vectors)
    histogram = run2.oneArray(labels, CLUSTERS)
    histogram *= inverseFrequencies
    categoryIndex = run2.classify(histogram, adjustedHistograms)
    return CATEGORIES[categoryIndex]


def testAccuracy_run1(K):
    score = 0
    scores = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    maxScore = (100 - TRAINING_IMG_PER_CATEGORY) * CATEGORIES.__len__()
    allVecs = run1.importVectors()
    for categoryI in range(0, CATEGORIES.__len__()):
        category = CATEGORIES[categoryI]
        for i in range(TRAINING_IMG_PER_CATEGORY, 100):
            image = iio.imread(TRAINING_IMG_FOLDER + '\\' + category + '\\' + str(i) + ".jpg")
            result = run1.categorise(image, False, allVecs, K)
            if result == category:
                score += 1
                scores[categoryI] += 1
        print("loading ", int((categoryI / CATEGORIES.__len__()) * 100), "%")
    accuracy = score
    print("Accuracy all: ", (score / maxScore) * 100, "%")
    print("===")
    for i in range(0, CATEGORIES.__len__()):
        print("Accuracy ", CATEGORIES[i], ": ", (scores[i] / (maxScore / CATEGORIES.__len__())) * 100, "%")
    return accuracy


def testAccuracy_run2():
    score = 0
    scores = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    maxScore = (100 - TRAINING_IMG_PER_CATEGORY) * CATEGORIES.__len__()
    for categoryI in range(0, CATEGORIES.__len__()):
        category = CATEGORIES[categoryI]
        for i in range(TRAINING_IMG_PER_CATEGORY, 100):
            image = iio.imread(TRAINING_IMG_FOLDER + '\\' + category + '\\' + str(i) + ".jpg")
            result = do_run2(image)
            if result == category:
                score += 1
                scores[categoryI] += 1
        print("loading ", int((categoryI / CATEGORIES.__len__()) * 100), "%")
    print("Accuracy all: ", (score / maxScore) * 100, "%")
    print("===")
    for i in range(0, CATEGORIES.__len__()):
        print("Accuracy ", CATEGORIES[i], ": ", (scores[i] / (maxScore / CATEGORIES.__len__())) * 100, "%")


def tune_run1():
    highestScore = 0
    bestK = 0
    for i in range(1, 100):
        print(i)
        score = testAccuracy_run1(i)
        if score > highestScore:
            highestScore = score
            bestK = i
    print(bestK)


def testFile_run1():
    fileName = 'run1.txt'
    if path.isfile(fileName):
        open(fileName, 'w').close()
    with open(fileName, 'a') as file:
        for i in range(0, 2988):
            print(i)
            if path.isfile(TESTING_IMG_FOLDER + "\\" + str(i) + ".jpg"):
                image = iio.imread(TESTING_IMG_FOLDER + "\\" + str(i) + ".jpg")
                file.write(str(i) + ".jpg " + str.lower(str(do_run1(image))))
                file.write('\n')


def testFile_run2():
    fileName = 'run2.txt'
    if path.isfile(fileName):
        open(fileName, 'w').close()
    with open(fileName, 'a') as file:
        for i in range(0, 2988):
            print(i)
            if path.isfile(TESTING_IMG_FOLDER + "\\" + str(i) + ".jpg"):
                image = iio.imread(TESTING_IMG_FOLDER + "\\" + str(i) + ".jpg")
                file.write(str(i) + ".jpg " + str.lower(str(do_run2(image))))
                file.write('\n')


if __name__ == '__main__':
    # ==RUN1==
    #train_run1()
    #test_run1()
    #testAccuracy_run1()
    #tune_run1()
    #testFile_run1()

    # ==RUN2==
    #train_run2()
    #test_run2()
    #testAccuracy_run2()
    #tune_run2()
    #testFile_run2()

    # ==RUN3==
    #run3.create_training_dataset()
    #run3.create_testing_dataset()
    #run3.predict_testing_dataset()

    print()
