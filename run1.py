import math
import numpy as np
from PIL import Image as pillow
import json

CATEGORIES = ["bedroom", "Coast", "Forest", "Highway", "industrial", "Insidecity", "kitchen", "livingroom", "Mountain", "Office", "OpenCountry", "store", "Street", "Suburb", "TallBuilding"]
TRAINED_VECTORS_FOLDER = "trainingData_run1"
TINY_IMG_SIZE = 16
TRAINING_IMG_PER_CATEGORY = 80
CATEGORIES_COUNT = CATEGORIES.__len__()
SIZE_SQUARED = int(math.pow(TINY_IMG_SIZE, 2))


# Convert an image to a vector
def imgToVector(image) -> np.ndarray:
    image.astype(dtype=float)
    result = cropToSquare(image)
    result = downscale(result)
    result = concatanate(result)
    result = zeroMean(result)
    result = unitLength(result)
    return result


# Crops an image to a square, about the center
def cropToSquare(img: np.ndarray) -> np.ndarray:
    height = img.shape[0]
    width = img.shape[1]
    wdiff = max(width - height, 0)
    hdiff = max(height - width, 0)
    halfwdiff = wdiff / 2.0
    halfhdiff = hdiff / 2.0
    # Set bounds
    l = math.floor(halfwdiff)
    r = width - (math.ceil(halfwdiff))
    u = math.floor(halfhdiff)
    d = height - (math.ceil(halfhdiff))
    # Crop
    newImg = img[u:d, l:r]
    return newImg


# Downscales image to 16x16
def downscale(img) -> np.ndarray:
    global TINY_IMG_SIZE
    pillowImg = pillow.fromarray(img)
    pillowImg = pillowImg.resize((TINY_IMG_SIZE, TINY_IMG_SIZE))
    return np.array(pillowImg)


# Flattens 2D array to 1D array
def concatanate(arr: np.ndarray) -> np.ndarray:
    arr = arr.flatten()
    return arr


# Sets mean of array to 0
def zeroMean(arr: np.ndarray) -> np.ndarray:
    # Get current mean
    mean = np.mean(arr)
    # Subtract mean from each value
    newArr = np.subtract(arr, mean)
    return newArr


# Converts vector to unit vector
def unitLength(arr: np.ndarray):
    length = np.linalg.norm(arr)
    arr = arr / length
    return arr


# Imports all vectors of training images, with categories
def importVectors() -> np.ndarray:
    global CATEGORIES, CATEGORIES_COUNT, SIZE_SQUARED, TRAINING_IMG_PER_CATEGORY, TRAINED_VECTORS_FOLDER
    allCategories = np.zeros((CATEGORIES_COUNT, TRAINING_IMG_PER_CATEGORY, SIZE_SQUARED), dtype=float)
    for i in range(0, CATEGORIES_COUNT):
        with (open(TRAINED_VECTORS_FOLDER + '\\' + CATEGORIES[i] + '.txt', 'r') as file):
            for e in range(0, TRAINING_IMG_PER_CATEGORY):
                arrText = file.readline()
                allCategories[i][e] = np.array(json.loads(arrText[:-1]))
    return allCategories


# Use k-nearest neighbour to match image to the closest category
def categorise(img: np.ndarray, printNeighbours: bool, allVecs: np.ndarray, K) -> str:
    global CATEGORIES, CATEGORIES_COUNT, TRAINING_IMG_PER_CATEGORY
    distances = []
    # Get all distances
    for x in range(0, CATEGORIES_COUNT):
        category = CATEGORIES[x]
        for y in range(0, TRAINING_IMG_PER_CATEGORY):
            distances.append((int(vectorDistance(imgToVector(img), allVecs[x][y])), category))
    # Sort distances
    distances.sort(key=lambda d: d[0])
    tops = distances[:K]
    cats = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(0, K):
        cats[CATEGORIES.index(tops[i][1])] += 1
    highest = 0
    highestIndex = 0
    # Find most common category
    for e in range(0, CATEGORIES_COUNT):
        if cats[e] > highest:
            highest = cats[e]
            highestIndex = e
    if printNeighbours:
        for e in range(0, CATEGORIES_COUNT):
            if cats[e] != 0:
                print(CATEGORIES[e], ": ", cats[e])
    return CATEGORIES[highestIndex]


# Calculates vector distance between two points
def vectorDistance(v1: np.ndarray, v2: np.ndarray) -> float:
    global SIZE_SQUARED
    powerSum = 0
    for i in range(0, SIZE_SQUARED):
        powerSum += abs(math.pow(v1[i] - v2[i], 2))
    return math.sqrt(powerSum)
