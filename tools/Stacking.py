from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

class Stacking:
    """Implements Stacking -- a kind of ensemble methods.

    Parameters
    ----------
    base_learners : list of Classifiers or Regressors
        The collection of base learners which stacking used to create new features.

    top_learner : object
        The top learner from which the stacking ensemble is built.

    n_splits: int, default=5
        Number of folds. Must be at least 2.

    include_old_features : bool, default=False
        If consider predictions of base learners as an expansion of old features.

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

    n_splits_: int
        Number of folds. Must be at least 2.

    include_old_features_ : bool
        If consider predictions of base learners as an expansion of old features.
    """
    def __init__(self,base_learners,top_learner,n_splits=5,include_old_features=False,post_normalization=False):
        self.base_learners=base_learners
        self.top_learner=top_learner
        self.n_splits=n_splits
        self.include_old_features=include_old_features
        self.post_normalization=post_normalization

    def fit(self,X,y):
        """Fit estimator.
            If include_old_features_==True then combine new features and
            old features with min-max normalization, then fit.

        :param X: np.matrix of shape = [n_samples, n_features]
            Training data.
        :param y: array-like of shape = [n_samples]
            The target values.
        :return: None
        """
        self.meta_X=X
        self.train_X=self.__get_train_X(X,y)
        self.train_y=np.array(y)
        if self.include_old_features:
            self.train_X=np.hstack((self.meta_X,self.train_X))
            self.scaler=MinMaxScaler()
            self.train_X=self.scaler.fit_transform(self.train_X)
        if self.post_normalization:
            self.standard_scaler=StandardScaler()
            self.train_X=self.standard_scaler.fit_transform(self.train_X)
        self.top_learner.fit(self.train_X,self.train_y)

    def predict(self,X):
        """Predict classes for X.

        :param X: np.matrix of shape = [n_samples, n_features]
            Testing data.
        :return:
            y : array of shape = [n_samples]
            The predictions.
        """
        test_X=self.__get_train_X(X,pd.DataFrame())
        if self.include_old_features:
            test_X=np.hstack((X,test_X))
            test_X=self.scaler.transform(test_X)
        if self.post_normalization:
            test_X=self.standard_scaler.transform(test_X)
        return self.top_learner.predict(test_X)

    def get_new_features(self,X):
        """Get predictions of trained base learners.
            Notice : call fit() first!

        :param X: np.matrix of shape = [n_samples, n_features]
            Meta features without labels.
        :return: np.matrix of shape = [n_samples, n_base_learners]
            New features.
        """
        ret_X=np.zeros((X.shape[0],len(self.base_learners)))
        for i,learner in enumerate(self.base_learners):
            ret_X[:,i]=learner.predict(X)
        return ret_X

    def __get_train_X(self,X,y):
        train_X = np.zeros((X.shape[0], len(self.base_learners)))
        if len(y)>0:
            X = np.array(X)
            y = np.array(y)
            kf = KFold(n_splits=self.n_splits, shuffle=True)
            for learner_cnt,learner in enumerate(self.base_learners):
                for train_index, test_index in kf.split(X):
                    learner.fit(X[train_index], y[train_index])
                    test_y = learner.predict(X[test_index])
                    for id_cnt,id in enumerate(test_index):
                        train_X[id][learner_cnt] = test_y[id_cnt]
        else:
            for learner_cnt,learner in enumerate(self.base_learners):
                learner.fit(self.meta_X,self.train_y)
                train_X[:,learner_cnt]=learner.predict(X)
        return train_X
