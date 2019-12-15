import os
import matplotlib.pyplot as plt
from LogisticRegression import logistic_regression
from LRM import logistic_regression_multiclass
from DataReader import *

# data_dir = "./data/"
data_dir = ""
train_filename = "training.npz"
test_filename = "test.npz"


def visualize_features(X, y):
    '''This function is used to plot a 2-D scatter plot of training features.

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.

    Returns:
        No return. Save the plot to 'train_features.*' and include it
        in submission.
    '''
    # YOUR CODE HERE

    pos_index = np.where(y == 1)

    neg_index = np.where(y == -1)

    plt.scatter(X[pos_index, 0], X[pos_index, 1], c='red', marker='o')
    plt.scatter(X[neg_index, 0], X[neg_index, 1], c='blue', marker='x')

    plt.xlabel('symmetry feature')
    plt.ylabel('intensity_feature')

    plt.title('train features')
    plt.savefig("train_features.png")

    plt.show()
    # END YOUR CODE


def visualize_result(X, y, W):
    '''This function is used to plot the sigmoid model after training.

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.
        W: An array of shape [n_features,].

    Returns:
        No return. Save the plot to 'train_result_sigmoid.*' and include it
        in submission.
    '''
    # YOUR CODE HERE
    pos_index = np.where(y == 1)
    neg_index = np.where(y == -1)
    plt.scatter(X[pos_index, 0], X[pos_index, 1], c='red', marker='o', s=15)
    plt.scatter(X[neg_index, 0], X[neg_index, 1], c='blue', marker='x', s=15)
    plt.plot([0, (W[2] - W[0]) / W[1]], [-W[0] / W[2], -1], 'k')
    plt.xlabel('symmetry feature')
    plt.ylabel('intensity_feature')
    plt.title('train result sigmoid')
    plt.savefig("../train_result_sigmoid.jpeg")
    plt.show()
    # END YOUR CODE


def visualize_result_multi(X, y, W):
    '''This function is used to plot the softmax model after training.

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 0,1,2.
        W: An array of shape [n_features, 3].

    Returns:
        No return. Save the plot to 'train_result_softmax.*' and include it
        in submission.
    '''
    # YOUR CODE HERE
    cls1 = np.where(y == 0)
    cls2 = np.where(y == 1)
    cls3 = np.where(y == 2)
    plt.scatter(X[cls1, 0], X[cls1, 1], c='red', marker='o')
    plt.scatter(X[cls2, 0], X[cls2, 1], c='blue', marker='x')
    plt.scatter(X[cls3, 0], X[cls3, 1], c='green', marker='^')
    new_W = np.vstack([W[:, 0]-W[:, 1], W[:, 0]-W[:, 2], W[:, 1]-W[:, 2]])
    plt.plot([0, -1], [-new_W[0][0]/new_W[0][2], (new_W[0][1]-new_W[0]
                                                  [0])/new_W[0][2]], color='k', linestyle='-', label='l01')
    plt.plot([0, -1], [-new_W[1][0]/new_W[1][2], (new_W[1][1]-new_W[1]
                                                  [0])/new_W[1][2]], color='k', linestyle='--', label='l02')
    plt.plot([0, (new_W[2][2]-new_W[2][0])/new_W[2][1]], [-new_W[2]
                                                          [0]/new_W[2][2], -1], color='k', linestyle='-.', label='l12')
    plt.legend()
    plt.xlabel('symmetry feature')
    plt.ylabel('intensity_feature')
    plt.title('train result softmax')
    plt.savefig("../train_result_softmax.jpeg")
    plt.show()
    # END YOUR CODE


def main():
    # ------------Data Preprocessing------------
    # Read data for training.

    raw_data, labels = load_data(os.path.join(data_dir, train_filename))
    raw_train, raw_valid, label_train, label_valid = train_valid_split(raw_data, labels, 2300)

    # Preprocess raw data to extract features
    train_X_all = prepare_X(raw_train)
    valid_X_all = prepare_X(raw_valid)
  
    # Preprocess labels for all data to 0,1,2 and return the idx for data from '1' and '2' class.
    train_y_all, train_idx = prepare_y(label_train)
    valid_y_all, val_idx = prepare_y(label_valid)

    # For binary case, only use data from '1' and '2'
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]

    # Only use the first 1350 data examples for binary training.
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx]

    # set lables to  1 and -1. Here convert label '2' to '-1' which means we treat data '1' as postitive class.
    train_y[np.where(train_y == 2)] = -1
    valid_y[np.where(valid_y == 2)] = -1

    data_shape = train_y.shape[0]

    print(train_X[1:3])
    print('train_X')
    print(train_X.shape)
    print(train_X[:, 1:3].shape)

   # Visualize training data.
  # visualize_features(train_X[:, 1:3], train_y)


   # ------------Logistic Regression Sigmoid Case------------

#    Check GD, SGD, BGD
    logisticR_classifier = logistic_regression(learning_rate=0.5, max_iter=100)

    logisticR_classifier.fit_GD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_BGD(train_X, train_y, data_shape)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_SGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_BGD(train_X, train_y, 1)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_BGD(train_X, train_y, 10)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))


    # Explore different hyper-parameters.
    # YOUR CODE HERE
    lr_list = [0.01, 0.05, 0.1, 0.5, 1]
    iter_list = [100, 300, 500, 700, 900]
    score_record = 0.0
    best_lr = 0.0
    best_iter = 0.0
    print('testing different learning rate and iteration numbers...')
    for lr in lr_list:
        for n_iter in iter_list:
            classifier = logistic_regression(learning_rate=lr, max_iter=n_iter)
            classifier.fit_SGD(train_X, train_y)
            score = classifier.score(valid_X, valid_y)
            if(score > score_record):
                score_record = score
                best_lr = lr
                best_iter = n_iter
            print("score = {}, lr = {}, n_iter = {}\n".format(score, lr, n_iter))

    print("best_lr = {}, best_iter = {}\n".format(best_lr, best_iter))

    print("testing different batch size...")
    bs_list = [4, 16, 64, 256]
    best_bs = 1
    for bs in bs_list:
        classifier = logistic_regression(learning_rate=best_lr, max_iter=best_iter)
        classifier.fit_BGD(train_X, train_y, bs)
        score = classifier.score(valid_X, valid_y)
        if(score > score_record):
            score_record = score
            best_bs = bs
        print("score = {}, bs = {}\n".format(score, bs))

    print("best_bs = {}\n".format(best_bs))

    best_logisticR = logistic_regression(learning_rate=best_lr, max_iter=best_iter)
    best_logisticR.fit_BGD(train_X, train_y, best_bs)
    # END YOUR CODE

    # Visualize the your 'best' model after training.
    visualize_result(train_X[:, 1:3], train_y, best_logisticR.get_params())

    # YOUR CODE HERE
    train_valid_X = np.concatenate((train_X, valid_X))
    train_valid_y = np.concatenate((train_y, valid_y))

    test_data, test_labels = load_data(os.path.join(data_dir, test_filename))
    test_y_all, test_idx = prepare_y(test_labels)
    test_X_all = prepare_X(test_data)
    test_y = test_y_all[test_idx]
    test_y[np.where(test_y == 2)] = -1
    test_X = test_X_all[test_idx]
    # END YOUR CODE

    # Use the 'best' model above to do testing. Note that the test data should be loaded and processed in the same way as the training data.
    # YOUR CODE HERE
    best_logisticR.fit_BGD(train_valid_X, train_valid_y, best_bs)
    score = best_logisticR.score(test_X, test_y)
    print("test score on the test set by best model is {}".format(score))
    # END YOUR CODE

    # ------------Logistic Regression Multiple-class case, let k= 3------------
    ###### Use all data from '0' '1' '2' for training
    train_X = train_X_all
    train_y = train_y_all
    valid_X = valid_X_all
    valid_y = valid_y_all

    #########  BGD for multiclass Logistic Regression
    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.5, max_iter=100,  k= 3)
    logisticR_classifier_multiclass.fit_BGD(train_X, train_y, 10)
    print(logisticR_classifier_multiclass.get_params())
    print(logisticR_classifier_multiclass.score(train_X, train_y))

    # Explore different hyper-parameters.
    ### YOUR CODE HERE
    lr_list = [0.01, 0.05, 0.1, 0.5, 1]
    iter_list = [100, 300, 500, 700, 900]
    score_record = 0.0
    best_lr = 0.0
    best_iter = 0.0
    print('testing different learning rate and iteration numbers...')
    for lr in lr_list:
        for n_iter in iter_list:
            classifier = logistic_regression_multiclass(learning_rate=lr, max_iter=n_iter, k=3)
            classifier.fit_BGD(train_X_all, train_y_all, 1)
            score = classifier.score(valid_X_all, valid_y_all)
            if(score > score_record):
                score_record = score
                best_lr = lr
                best_iter = n_iter
            print("score = {}, lr = {}, n_iter = {}\n".format(score, lr, n_iter))

    print("best_lr = {}, best_iter = {}\n".format(best_lr, best_iter))

    print("testing different batch size...")
    bs_list = [4, 16, 64, 256]
    best_bs = 1
    for bs in bs_list:
        classifier = logistic_regression_multiclass(learning_rate=best_lr, max_iter=best_iter, k=3)
        classifier.fit_BGD(train_X, train_y, bs)
        score = classifier.score(valid_X, valid_y)
        if(score > score_record):
            score_record = score
            best_bs = bs
        print("score = {}, bs = {}\n".format(score, bs))

    print("best_bs = {}\n".format(best_bs))

    best_logistic_multi_R = logistic_regression_multiclass(learning_rate=best_lr, max_iter=best_iter, k=3)
    best_logistic_multi_R.fit_BGD(train_X_all, train_y_all, best_bs)
    ### END YOUR CODE

    # Visualize the your 'best' model after training.
    visualize_result_multi(train_X[:, 1:3], train_y, logisticR_classifier_multiclass.get_params())

    # Use the 'best' model above to do testing.
    ### YOUR CODE HERE
    train_valid_X = np.concatenate((train_X_all, valid_X_all))
    train_valid_y = np.concatenate((train_y_all, valid_y_all))

    best_logistic_multi_R.fit_BGD(train_valid_X, train_valid_y, best_bs)
    score = best_logistic_multi_R.score(test_X_all, test_y_all)
    print("test score on the test set by best model is {}".format(score))
    ### END YOUR CODE

    # ------------Connection between sigmoid and softmax------------
    # Now set k=2, only use data from '1' and '2'

    # set labels to 0,1 for softmax classifer
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx]
    train_y[np.where(train_y == 2)] = 0
    valid_y[np.where(valid_y == 2)] = 0

    # First, fit softmax classifer until convergence, and evaluate
    # Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
    # YOUR CODE HERE
    lr = 0.5
    max_iter = 1000
    n_samples = train_X.shape[0]
    sf_classifier = logistic_regression_multiclass(lr, max_iter, 2)
    sf_classifier.fit_BGD(train_X, train_y, n_samples)
    score = sf_classifier.score(valid_X, valid_y)
    print("The score of converged softmax classifier is {}\n".format(score))
    # END YOUR CODE

    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx]
    # set lables to -1 and 1 for sigmoid classifer
    train_y[np.where(train_y == 2)] = -1
    valid_y[np.where(valid_y == 2)] = -1

    # Next, fit sigmoid classifer until convergence, and evaluate
    # Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
    # YOUR CODE HERE
    lo_classifier = logistic_regression(lr, max_iter)
    lo_classifier.fit_BGD(train_X, train_y, n_samples)
    score = lo_classifier.score(valid_X, valid_y)
    print("The score of converged logistic classifier is {}\n".format(score))
    # END YOUR CODE

    # Compare and report the observations/prediction accuracy


    '''
    Explore the training of these two classifiers and monitor the graidents/weights for each step. 
    Hint: First, set two learning rates the same, check the graidents/weights for the first batch in the first epoch. What are the relationships between these two models? 
    Then, for what leaning rates, we can obtain w_1-w_2= w for all training steps so that these two models are equivalent for each training step. 
    '''
    # YOUR CODE HERE
    train_y_multi = train_y.copy()
    train_y_multi[np.where(train_y_multi==-1)] = 0
    print("When setting the same learning rate...")
    classifier1 = logistic_regression(lr, 1)
    classifier2 = logistic_regression_multiclass(lr, 1, 2)
    classifier1.fit_BGD(train_X, train_y, n_samples)
    classifier2.fit_BGD(train_X, train_y_multi, n_samples)
    print("After {} steps of training...".format(0))
    print("The weight w of logistic classifier is {}".format(classifier1.get_params()))
    print("The weight w1, w2 of softmax classifier is {}, {}".format(classifier2.get_params()[:, 0], classifier2.get_params()[:,1]))
    print("w2-w1={}\n".format(classifier2.get_params()[:, 1]-classifier2.get_params()[:,0]))
    
    print("When setting the learning rate of logistic regression classifier is two times of that in softmax classifier...")
    for i in range(1,10):
        classifier1 = logistic_regression(2 * lr, i)
        classifier2 = logistic_regression_multiclass(lr, i, 2)
        classifier1.fit_BGD(train_X, train_y, n_samples)
        classifier2.fit_BGD(train_X, train_y_multi, n_samples)
        print("After {} steps of training...".format(i))
        print("The weight w of logistic classifier is {}".format(classifier1.get_params()))
        print("The weight w1, w2 of softmax classifier is {}, {}".format(classifier2.get_params()[:, 0], classifier2.get_params()[:,1]))
        print("w2-w1={}\n".format(classifier2.get_params()[:, 1]-classifier2.get_params()[:,0]))
    # END YOUR CODE

# ------------End------------


if __name__ == '__main__':
    main()
