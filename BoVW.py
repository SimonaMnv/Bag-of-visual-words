import seaborn as sns
from cv2 import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
from sklearn.cluster import MiniBatchKMeans, AffinityPropagation, MeanShift
import shutil


#########################################
# SimonaMnv								#
#########################################


# Dataset Used: https://www.kaggle.com/prasunroy/natural-images


def img_train_test_split(img_source_dir, train_ratio):
    # Custom function that splits a folder with a bunch of images to train and test folders based on a percentage given

    # Creating Train / Test folders (One time use)
    shutil.rmtree(img_source_dir + "/test")
    shutil.rmtree(img_source_dir + "/train")
    root_dir = img_source_dir
    classes_dir = ['/airplane', '/car', '/motorbike', '/person']

    for cls in classes_dir:
        os.makedirs(root_dir + '/train' + cls)
        os.makedirs(root_dir + '/test' + cls)

        # Creating partitions of the data after shuffling
        src = root_dir + cls  # Folder to copy images from

        allFileNames = os.listdir(src)
        np.random.shuffle(allFileNames)
        test_FileNames, val_FileNames, train_FileNames = np.split(np.array(allFileNames),
                                                                  [int(
                                                                      len(allFileNames) * (1 - train_ratio)),
                                                                      int(len(allFileNames) * (1 - train_ratio))])

        train_FileNames = [src + '/' + name for name in train_FileNames.tolist()]
        test_FileNames = [src + '/' + name for name in test_FileNames.tolist()]

        print('Total images: ', len(allFileNames))
        print('Training: ', len(train_FileNames))
        print('Testing: ', len(test_FileNames))

        # Copy-pasting images
        for name in train_FileNames:
            shutil.copy(name, root_dir + '/train' + cls)
        for name in test_FileNames:
            shutil.copy(name, root_dir + '/test' + cls)


def load_images_from_folder(folder, inputImageSize):
    # return a dictionary that holds all images category by category.

    images = {}
    for filename in os.listdir(folder):
        category = []
        path = folder + "/" + filename
        for cat in os.listdir(path):
            img = cv2.imread(path + "/" + cat)
            if img is not None:
                # grayscale it
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # resize it, if necessary
                img = cv2.resize(img, (inputImageSize[0], inputImageSize[1]))
                category.append(img)
        images[filename] = category
        print('...Finished parsing images.')
    return images


def detector_features(images, choice):
    # Creates descriptors using an approach of your choise. e.g. ORB, SIFT, SURF, FREAK, MOPS, ετψ
    # Takes one parameter that is images dictionary
    # Return an array whose first index holds the decriptor_list without an order
    # And the second index holds the sift_vectors dictionary which holds the descriptors but this is seperated class by class

    print(' . start detecting points and calculating features for a given image set')
    detector_vectors = {}
    descriptor_list = []
    if choice == 1:
        detectorToUse = cv2.xfeatures2d.SIFT_create()
    if choice == 2:
        detectorToUse = cv2.ORB_create()
    if choice == 3:
        detectorToUse = cv2.BRISK_create()

    for nameOfCategory, availableImages in images.items():
        features = []
        for img in availableImages:  # reminder: val
            kp, des = detectorToUse.detectAndCompute(img, None)
            descriptor_list.extend(des)
            features.append(des)
        detector_vectors[nameOfCategory] = features
        print(' . finished detecting points and calculating features for a given image set')
    return [descriptor_list, detector_vectors]  # be aware of the []! this is ONE output as a list


def kmeansVisualWordsCreation(k, descriptor_list, choice):
    # A k-means clustering algorithm who takes 2 parameter which is number
    # of cluster(k) and the other is descriptors list(unordered 1d array)
    # Returns an array that holds central points.

    print(' . calculating central points for the existing feature values.')
    if choice == 1:  # Mini Batch K-Means
        batchSize = np.ceil(descriptor_list.__len__() / 50).astype('int')
        model = MiniBatchKMeans(n_clusters=k, batch_size=batchSize, verbose=0)
        model.fit(descriptor_list)
        visualWords = model.cluster_centers_  # a.k.a. centers of reference
    if choice == 2:  # TODO: Test more values here
        model = AffinityPropagation(max_iter=100, convergence_iter=2)
        model.fit(descriptor_list)
        visualWords = model.cluster_centers_
    print(' . done calculating central points for the given feature set.')
    return visualWords, model


def mapFeatureValsToHistogram(DataFeaturesByClass, visualWords, TrainedModel):
    # Creation of the histograms. To create our each image by a histogram. We will create a vector of k values for each
    # image. For each keypoints in an image, we will find the nearest center, defined using training set
    # and increase by one its value

    # depending on the approach you may not need to use all inputs
    histogramsList = []
    targetClassList = []
    numberOfBinsPerHistogram = visualWords.shape[0]
    for categoryIdx, featureValues in DataFeaturesByClass.items():
        for tmpImageFeatures in featureValues:  # yes, we check one by one the values in each image for all images
            tmpImageHistogram = np.zeros(numberOfBinsPerHistogram)
            tmpIdx = list(TrainedModel.predict(tmpImageFeatures))
            clustervalue, visualWordMatchCounts = np.unique(tmpIdx, return_counts=True)
            tmpImageHistogram[clustervalue] = visualWordMatchCounts
            # do not forget to normalize the histogram values
            numberOfDetectedPointsInThisImage = tmpIdx.__len__()
            tmpImageHistogram = tmpImageHistogram / numberOfDetectedPointsInThisImage
            # now update the input and output coresponding lists
            histogramsList.append(tmpImageHistogram)
            targetClassList.append(categoryIdx)

    plt.hist(histogramsList[0], bins=numberOfBinsPerHistogram, histtype='stepfilled', color='r')
    plt.title("Visual Bag of words Histogram for img1")
    plt.show()
    plt.hist(histogramsList[1], bins=numberOfBinsPerHistogram, histtype='stepfilled', color='b')
    plt.title("Visual Bag of words Histogram for img2")
    plt.show()
    return histogramsList, targetClassList


def images_report(testImages, y_pred_test, blabla):
    # Plot the images
    all_test_Images = testImages['airplane'] + testImages['car'] + testImages['motorbike'] + testImages['person']

    # By concatenating them all together i have achieved: e.x: all_test_images[350] = y_pred_test[350],
    # meaning, that I can now match each input picture with its corresponding PREDICTED class.
    # Now shuffle them both together to get random images
    c = list(zip(all_test_Images, y_pred_test))
    random.shuffle(c)
    all_test_Images, y_pred_test = zip(*c)

    # Now plot random images with their prediction - Call after each classifier for different results.
    fig = plt.figure(figsize=(8, 8))
    columns = 4
    rows = 4
    plt.title(blabla)
    for i in range(1, columns * rows + 1):
        img = all_test_Images[i]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img, cmap='gray')
        if y_pred_test[i] == 0:
            plt.xlabel("airplane")
        if y_pred_test[i] == 1:
            plt.xlabel("car")
        if y_pred_test[i] == 2:
            plt.xlabel("motorcycle")
        if y_pred_test[i] == 3:
            plt.xlabel("person")
    plt.show()


def CM_report(y_test, y_pred_test, blabla):
    # confusion matrix
    labels = ['airplane', 'car', 'motorcycle', 'person']
    CM = confusion_matrix(y_test, y_pred_test)
    sns.heatmap(CM, annot=True, xticklabels=labels, yticklabels=labels, fmt='g', cmap='Blues')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(blabla)
    plt.show()


def classify(choice_detector, choice_clusterer):
    # define a fixed image size to work with
    inputImageSize = [200, 200, 3]  # define the FIXED image size

    # Load our GENERAL dataset. Not split into train and test yet.
    dataset = 'dataset/natural_images'

    # Split our GENERAL dataset into train and test using a custom function that splits based on percentage given
    train_ratio = 0.05
    img_train_test_split(dataset, train_ratio)

    # separate the train and test dataset
    TrainImagesFilePath = 'dataset/natural_images/train'
    TestImagesFilePath = 'dataset/natural_images/test'

    # load all the images
    trainImages = load_images_from_folder(TrainImagesFilePath,
                                          inputImageSize)  # take all images category by category for train set

    # calculate points and descriptor values per image
    trainDataFeatures = detector_features(trainImages, choice_detector)
    # Takes the descriptor list which is unordered one
    TrainDescriptorList = trainDataFeatures[0]

    # create the central points for the histograms using k means.
    # here we use a rule of the thumb to create the expected number of cluster centers
    numberOfClasses = trainImages.__len__()  # retrieve num of classes from dictionary
    possibleNumOfCentersToUse = 10 * numberOfClasses
    visualWords, TrainedModel = kmeansVisualWordsCreation(possibleNumOfCentersToUse, TrainDescriptorList,
                                                          choice_clusterer)

    # Takes the sift feature values that is separated class by class for train data, we need this to calculate the histograms
    trainBoVWFeatureVals = trainDataFeatures[1]

    # create the train input train output format
    trainHistogramsList, trainTargetsList = mapFeatureValsToHistogram(trainBoVWFeatureVals, visualWords, TrainedModel)
    X_train = np.stack(trainHistogramsList, axis=0)

    # Convert Categorical Data For Scikit-Learn
    from sklearn import preprocessing

    # Create a label (category) encoder object
    labelEncoder = preprocessing.LabelEncoder()
    labelEncoder.fit(trainTargetsList)
    # convert the categories from strings to names
    y_train = labelEncoder.transform(trainTargetsList)

    # train and evaluate the classifiers
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    print('Accuracy of K-NN classifier on training set: {:.2f}'.format(knn.score(X_train, y_train)))

    from sklearn.tree import DecisionTreeClassifier

    clf = DecisionTreeClassifier().fit(X_train, y_train)
    print('Accuracy of Decision Tree classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))

    from sklearn.naive_bayes import GaussianNB

    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    print('Accuracy of GNB classifier on training set: {:.2f}'.format(gnb.score(X_train, y_train)))

    from sklearn.svm import SVC

    svm = SVC()
    svm.fit(X_train, y_train)
    print('Accuracy of SVM classifier on training set: {:.2f}'.format(svm.score(X_train, y_train)))

    # ----------------------------------------------------------------------------------------
    # now run the same things on the test data.
    # DO NOT FORGET: you use the same visual words, created using training set.

    # clear some space
    del trainImages, trainBoVWFeatureVals, trainDataFeatures, TrainDescriptorList

    # load the test images
    testImages = load_images_from_folder(TestImagesFilePath,
                                         inputImageSize)  # take all images category by category for train set

    # calculate points and descriptor values per image
    testDataFeatures = detector_features(testImages, choice_detector)

    # Takes the sift feature values that is seperated class by class for train data, we need this to calculate the histograms
    testBoVWFeatureVals = testDataFeatures[1]

    # create the test input / test output format
    testHistogramsList, testTargetsList = mapFeatureValsToHistogram(testBoVWFeatureVals, visualWords, TrainedModel)
    X_test = np.array(testHistogramsList)
    y_test = labelEncoder.transform(testTargetsList)

    # knn predictions
    # now check for both train and test data, how well the model learned the patterns
    y_pred_train = knn.predict(X_train)
    y_pred_test = knn.predict(X_test)
    # calculate the scores
    KNN_acc_train = accuracy_score(y_train, y_pred_train)
    KNN_acc_test = accuracy_score(y_test, y_pred_test)
    KNN_pre_train = precision_score(y_train, y_pred_train, average='macro')
    KNN_pre_test = precision_score(y_test, y_pred_test, average='macro')
    KNN_rec_train = recall_score(y_train, y_pred_train, average='macro')
    KNN_rec_test = recall_score(y_test, y_pred_test, average='macro')
    KNN_f1_train = f1_score(y_train, y_pred_train, average='macro')
    KNN_f1_test = f1_score(y_test, y_pred_test, average='macro')

    # print the scores
    print('Accuracy scores of K-NN classifier are:',
          'train: {:.2f}'.format(KNN_acc_train), 'and test: {:.2f}.'.format(KNN_acc_test))
    print('Precision scores of K-NN classifier are:',
          'train: {:.2f}'.format(KNN_pre_train), 'and test: {:.2f}.'.format(KNN_pre_test))
    print('Recall scores of K-NN classifier are:',
          'train: {:.2f}'.format(KNN_rec_train), 'and test: {:.2f}.'.format(KNN_rec_test))
    print('F1 scores of K-NN classifier are:',
          'train: {:.2f}'.format(KNN_f1_train), 'and test: {:.2f}.'.format(KNN_f1_test))
    print('')

    # EXTRA REPORTS!
    images_report(testImages, y_pred_test, "K-NN")  # Plot images with their classification depending on the classifier

    CM_report(y_test, y_pred_test, "K-NN Confusion Matrix")  # Plot CM depending on the classifier

    # naive Bayes
    # now check for both train and test data, how well the model learned the patterns
    y_pred_train = gnb.predict(X_train)
    y_pred_test = gnb.predict(X_test)
    # calculate the scores
    NB_acc_train = accuracy_score(y_train, y_pred_train)
    NB_acc_test = accuracy_score(y_test, y_pred_test)
    NB_pre_train = precision_score(y_train, y_pred_train, average='macro')
    NB_pre_test = precision_score(y_test, y_pred_test, average='macro')
    NB_rec_train = recall_score(y_train, y_pred_train, average='macro')
    NB_rec_test = recall_score(y_test, y_pred_test, average='macro')
    NB_f1_train = f1_score(y_train, y_pred_train, average='macro')
    NB_f1_test = f1_score(y_test, y_pred_test, average='macro')

    # print the scores
    print('Accuracy scores of GNB classifier are:',
          'train: {:.2f}'.format(NB_acc_train), 'and test: {:.2f}.'.format(NB_acc_test))
    print('Precision scores of GBN classifier are:',
          'train: {:.2f}'.format(NB_pre_train), 'and test: {:.2f}.'.format(NB_pre_test))
    print('Recall scores of GNB classifier are:',
          'train: {:.2f}'.format(NB_rec_train), 'and test: {:.2f}.'.format(NB_rec_test))
    print('F1 scores of GNB classifier are:',
          'train: {:.2f}'.format(NB_f1_train), 'and test: {:.2f}.'.format(NB_f1_test))
    print('')

    # EXTRA REPORTS!
    images_report(testImages, y_pred_test, "GNB")  # Plot images with their classification depending on the classifier

    CM_report(y_test, y_pred_test, "GNB Confusion Matrix")  # Plot CM depending on the classifier

    # support vector machines
    # now check for both train and test data, how well the model learned the patterns
    y_pred_train = svm.predict(X_train)
    y_pred_test = svm.predict(X_test)
    # calculate the scores
    SVM_acc_train = accuracy_score(y_train, y_pred_train)
    SVM_acc_test = accuracy_score(y_test, y_pred_test)
    SVM_pre_train = precision_score(y_train, y_pred_train, average='macro')
    SVM_pre_test = precision_score(y_test, y_pred_test, average='macro')
    SVM_rec_train = recall_score(y_train, y_pred_train, average='macro')
    SVM_rec_test = recall_score(y_test, y_pred_test, average='macro')
    SVM_f1_train = f1_score(y_train, y_pred_train, average='macro')
    SVM_f1_test = f1_score(y_test, y_pred_test, average='macro')

    # print the scores
    print('Accuracy scores of SVM classifier are:',
          'train: {:.2f}'.format(SVM_acc_train), 'and test: {:.2f}.'.format(SVM_acc_test))
    print('Precision scores of SVM classifier are:',
          'train: {:.2f}'.format(SVM_pre_train), 'and test: {:.2f}.'.format(SVM_pre_test))
    print('Recall scores of SVM classifier are:',
          'train: {:.2f}'.format(SVM_rec_train), 'and test: {:.2f}.'.format(SVM_rec_test))
    print('F1 scores of SVM classifier are:',
          'train: {:.2f}'.format(SVM_f1_train), 'and test: {:.2f}.'.format(SVM_f1_test))
    print('')

    # EXTRA REPORTS!
    images_report(testImages, y_pred_test, "SVM")  # Plot images with their classification depending on the classifier

    CM_report(y_test, y_pred_test, "SVM Confusion Matrix")  # Plot CM depending on the classifier

    # classification tree
    # predict outcomes for test data and calculate the test scores
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    # calculate the scores
    DT_acc_train = accuracy_score(y_train, y_pred_train)
    DT_acc_test = accuracy_score(y_test, y_pred_test)
    DT_pre_train = precision_score(y_train, y_pred_train, average='macro')
    DT_pre_test = precision_score(y_test, y_pred_test, average='macro')
    DT_rec_train = recall_score(y_train, y_pred_train, average='macro')
    DT_rec_test = recall_score(y_test, y_pred_test, average='macro')
    DT_f1_train = f1_score(y_train, y_pred_train, average='macro')
    DT_f1_test = f1_score(y_test, y_pred_test, average='macro')

    print('Accuracy scores of Decision Tree classifier are:',
          'train: {:.2f}'.format(DT_acc_train), 'and test: {:.2f}.'.format(DT_acc_test))
    print('Precision scores of Decision Tree classifier are:',
          'train: {:.2f}'.format(DT_pre_train), 'and test: {:.2f}.'.format(DT_pre_test))
    print('Recall scores of Decision Tree classifier are:',
          'train: {:.2f}'.format(DT_rec_train), 'and test: {:.2f}.'.format(DT_rec_test))
    print('F1 scores of Decision Tree classifier are:',
          'train: {:.2f}'.format(DT_f1_train), 'and test: {:.2f}.'.format(DT_f1_test))
    print('')

    # EXTRA REPORTS!
    images_report(testImages, y_pred_test, "DT")  # Plot images with their classification depending on the classifier

    CM_report(y_test, y_pred_test, "DT Confusion Matrix")  # Plot CM depending on the classifier

    if choice_detector == 1:
        fd = "SIFT"
    if choice_detector == 2:
        fd = "ORB"
    if choice_detector == 3:
        fd = "BRIEF"
    if choice_clusterer == 1:
        cc = "MB K-Means"
    if choice_clusterer == 2:
        cc = "Affinity"
    # Automatic export to excel
    scores = {
        'Feature Extraction': [fd, fd, fd, fd],
        'Clustering Detection': [cc, cc, cc, cc],
        'Classifier': ['Decision Tree', 'KNN', 'Naive Bayes', 'SVM'],
        'Train Ratio': [train_ratio * 100, train_ratio * 100, train_ratio * 100, train_ratio * 100],
        'Accuracy (tr)': [DT_acc_train, KNN_acc_train, NB_acc_train, SVM_acc_train],
        'Precision (tr)': [DT_pre_train, KNN_pre_train, NB_pre_train, SVM_pre_train],
        'Recall (tr)': [DT_rec_train, KNN_rec_train, NB_rec_train, SVM_rec_train],
        'F1 Score (tr)': [DT_f1_train, KNN_f1_train, NB_f1_train, SVM_f1_train],
        'Accuracy (te)': [DT_acc_test, KNN_acc_test, NB_acc_test, SVM_acc_test],
        'Precision (te)': [DT_pre_test, KNN_pre_test, NB_pre_test, SVM_pre_test],
        'Recall (te)': [DT_rec_test, KNN_rec_test, NB_rec_test, SVM_rec_test],
        'F1 Score (te)': [DT_f1_test, KNN_f1_test, NB_f1_test, SVM_f1_test]
    }
    df = pd.DataFrame(scores, columns=['Feature Extraction', 'Clustering Detection', 'Classifier', 'Train Ratio',
                                       'Accuracy (tr)', 'Precision (tr)', 'Recall (tr)', 'F1 Score (tr)',
                                       'Accuracy (te)',
                                       'Precision (te)', 'Recall (te)', 'F1 Score (te)'])
    df.to_excel('BoVW_scores[BRISK&KMEANS]-[5-95].xls')
    print("Excel Exported.")


# TODO: PICK PARAMETERS FIRST!
# First Parameter specifies the feature detector:
#       [1] for SIFT,
#       [2] for ORB,
#       [3] for BRISK.
# Second Parameter specifies the clustering method
#       [1] for MiniBatch K-means,
#       [2] for Affinity Propagation -> Only works with a VERY small number (aprox: 5%:95% ratio)

classify(3, 1)