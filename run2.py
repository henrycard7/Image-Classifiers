import numpy as np
import random

import py5
from sklearn.cluster import KMeans
from scipy import spatial

PATCH_SIZE = 8
STRIDE_LENGTH = 4

# returns list of feature vectors, given an image
def extractPatches(img):
    patches = []
    for y in range(0, img.shape[0] - PATCH_SIZE + 1, STRIDE_LENGTH):
        for x in range(0, img.shape[1] - PATCH_SIZE + 1, STRIDE_LENGTH):
            patch = img[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            if not np.std(patch) == 0:
                patch = (patch - np.mean(patch)) / np.std(patch)
            else:
                patch = patch - np.mean(patch)
            patches.append(patch.flatten())
    return np.array(patches)


# randomly selects vectors from list of vectors, given number to select
def select(numberOfSamples,vectors):
    selectedVectors = []
    test = len(vectors)
    indexes = list(range(len(vectors)))
    random.shuffle(indexes)
    for x in range(numberOfSamples):
        selectedVectors.append(vectors[indexes[x]])
    return np.array(selectedVectors)


# returns k-means model given vectors and number of clusters
def kMeans(patches, n_clusters):
    kmeans = KMeans(n_clusters)
    kmeans.fit_predict(patches)
    return kmeans


# returns list of histograms of word counts given list of quantised vectors, per category
def createArrays(categoryVectors, labels, n_clusters):
    histogramList = np.zeros((15, n_clusters))
    startIndex = 0
    for x in range(0, 15):
        anotherList = categoryVectors[x]
        for y in range(startIndex, startIndex + len(anotherList)):
            if labels[y] != -1:
                histogramList[x][labels[y]] += 1
        startIndex += len(anotherList)
    return histogramList


# returns histogram of word counts given list of quantised vectors
def oneArray(labels, n_clusters):
    histogram = np.zeros(n_clusters)
    for x in range(len(labels)):
        if labels[x] != -1:
            histogram[labels[x]] += 1
    return histogram


# Finds closest matching category histogram to img histogram, to return closest category
def classify(histogram, histogramList):
    highScore = 0
    highestCategory = -1
    for x in range(len(histogramList)):
        similarity = 1 - spatial.distance.cosine(histogram, histogramList[x])
        if similarity > highScore:
            highScore = similarity
            highestCategory = x
    return highestCategory


def getInverseFrequencies(histogramList: np.ndarray, n_clusters: int):
    means = np.zeros(n_clusters)
    highestMean = 0
    for list in histogramList:
        for i in range(0, n_clusters):
            means[i] += list[i]
            if means[i] > highestMean:
                highestMean = means[i]
    means /= n_clusters
    highestMean /= n_clusters
    for a in range(0, n_clusters):
        means[a] = (1 - (py5.remap(means[a], 0, highestMean, 0, 1))) ** 10
    adjustedHistograms = np.ndarray(np.shape(histogramList))
    for e in range(0, 15):
        adjustedHistograms[e] = histogramList[e] * means
    return means, adjustedHistograms
