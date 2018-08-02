from sklearn.model_selection import KFold
import numpy as np

class Stacking:
    """Implements Stacking -- a kind of ensemble methods.

    Parameters
    ----------
    base_learners : list of Classifiers or Regressors
        The collection of base learners which stacking used to create new features.

    top_learner : object
        The top learner from which the stacking ensemble is built.

    Attributes
    ----------
    base_learners_ : list of Classifiers or Regressors
        The collection of base learners which stacking used to create new features.

    top_learner_ : object
        The top learner from which the stacking ensemble is built.

    meta_X_ : np.matrix
        The original training data.

    train_X_ : np.matrix
        The new training data created with predictions based on base learners.

    train_y_ : np.matrix
        The labels of training data.
    """
    def __init__(self,base_learners,top_learner):
        self.base_learners=base_learners
        self.top_learner=top_learner

    def fit(self,X,y,n_splits=5):
        """Fit estimator.

        :param X: np.matrix of shape = [n_samples, n_features]
            Training data.
        :param y: array-like of shape = [n_samples]
            The target values.
        :param n_splits: int, default=5
            Number of folds. Must be at least 2.
        :return: None
        """
        self.meta_X=X
        self.train_X=self.__get_train_X(X,y,n_splits=n_splits)
        self.train_y=np.array(y)
        self.top_learner.fit(self.train_X,self.train_y)

    def predict(self,X):
        """Predict classes for X.

        :param X: np.matrix of shape = [n_samples, n_features]
            Testing data.
        :return:
            y : array of shape = [n_samples]
            The predictions.
        """
        test_X=self.__get_train_X(X,None,n_splits=0)
        return self.top_learner.predict(test_X)

    def __get_train_X(self,X,y,n_splits):
        X = np.array(X)
        y = np.array(y)
        train_X = np.zeros((X.shape[0], len(self.base_learners)))
        if n_splits>0:
            kf = KFold(n_splits=n_splits, shuffle=True)
            learner_cnt = 0
            for learner in self.base_learners:
                for train_index, test_index in kf.split(X):
                    learner.fit(X[train_index], y[train_index])
                    test_y = learner.predict(X[test_index])
                    id_cnt = 0
                    for id in test_index:
                        train_X[id][learner_cnt] = test_y[id_cnt]
                        id_cnt = id_cnt + 1
                learner_cnt = learner_cnt + 1
        else:
            learner_cnt=0
            for learner in self.base_learners:
                learner.fit(self.meta_X,self.train_y)
                train_X[:,learner_cnt]=learner.predict(X)
                learner_cnt=learner_cnt+1
        return train_X
