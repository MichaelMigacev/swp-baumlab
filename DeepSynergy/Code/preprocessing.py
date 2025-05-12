import numpy as np
import pandas as pd
import pickle
import gzip

class Preprocessor:
    def __init__(self, test_fold=0, val_fold=1):
        self.test_fold = test_fold
        self.val_fold = val_fold
        self.means1 = None
        self.std1 = None
        self.means2 = None
        self.std2 = None
        self.feat_filt = None
    
    @staticmethod    
    def normalize(X, means1=None, std1=None, means2=None, std2=None, feat_filt=None, norm='tanh_norm'):
        if std1 is None:                                                      #If std1 not given: calculate
            std1 = np.nanstd(X, axis=0)
        if feat_filt is None:                                                 #If standard deviation is 0 (non informative) throw data away
            feat_filt = std1!=0
        X = X[:,feat_filt]
        X = np.ascontiguousarray(X)                                           #Data array needs to be continuous
        if means1 is None:                                                    #Calc mean and standardize
            means1 = np.mean(X, axis=0)
        X = (X-means1)/std1[feat_filt]
        if norm == 'norm':                                                    #Now we start the actual normalization (corresponds to the end of 2.2 in the paper):
            return(X, means1, std1, feat_filt)
        elif norm == 'tanh':
            return(np.tanh(X), means1, std1, feat_filt)
        elif norm == 'tanh_norm':
            X = np.tanh(X)
            if means2 is None:
                means2 = np.mean(X, axis=0)
            if std2 is None:
                std2 = np.std(X, axis=0)
            X = (X-means2)/std2
            X[:,std2==0]=0
            return(X, means1, std1, means2, std2, feat_filt)
    
    @staticmethod  
    def load_features(features_path='X.p.gz'):
        #contains the data in both feature ordering ways (drug A - drug B - cell line and drug B - drug A - cell line)
        #in the first half of the data the features are ordered (drug A - drug B - cell line)
        #in the second half of the data the features are ordered (drug B - drug A - cell line)
        file = gzip.open(features_path, 'rb')
        X = pickle.load(file)
        file.close()
        return X
    
    @staticmethod      
    def load_labels(labels_path='labels.csv'):         
        #contains synergy values and fold split (numbers 0-4)
        labels = pd.read_csv(labels_path, index_col=0)
        #labels are duplicated for the two different ways of ordering in the data
        labels = pd.concat([labels, labels])
        #In the end X contains both ordering ways, both labeled the same. This therefor prevents AB and BA from being in different folds and thereby prevents data leakage.
        return labels

    def create_splits(self, norm_method = 'tanh_norm', features_path='X.p.gz', labels_path='labels.csv'):
        """Create train/val/test splits with normalization."""
        labels = self.load_labels(labels_path)
        X = self.load_features(features_path)

        print(f"Creating folds using: \nnorm: {norm_method}\ntest_fold: {self.test_fold}\nval_fold: {self.val_fold}\n")
        # Get indices for different splits
        idx_tr = np.where(np.logical_and(labels['fold'] != self.test_fold, 
                         labels['fold'] != self.val_fold))[0]
        idx_val = np.where(labels['fold'] == self.val_fold)[0]
        idx_train = np.where(labels['fold'] != self.test_fold)[0]
        idx_test = np.where(labels['fold'] == self.test_fold)[0]

        # Split features
        X_tr = X[idx_tr]
        X_val = X[idx_val]
        X_train = X[idx_train]
        X_test = X[idx_test]

        # Split labels
        y_tr = labels.iloc[idx_tr]['synergy'].values
        y_val = labels.iloc[idx_val]['synergy'].values
        y_train = labels.iloc[idx_train]['synergy'].values
        y_test = labels.iloc[idx_test]['synergy'].values
        
        print("Normalizing Data..")
        # Normalize data
        if norm_method == "tanh_norm":
            X_tr, mean, std, mean2, std2, feat_filt = self.normalize(X_tr, norm=norm_method)
            X_val, mean, std, mean2, std2, feat_filt = self.normalize(X_val, mean, std, mean2, std2,
                                                            feat_filt=feat_filt, norm=norm_method)
            X_train, mean, std, mean2, std2, feat_filt = self.normalize(X_train, norm=norm_method)
            X_test, mean, std, mean2, std2, feat_filt = self.normalize(X_test, mean, std, mean2, std2,
                                                            feat_filt=feat_filt, norm=norm_method)
        else:
            X_tr, mean, std, feat_filt = self.normalize(X_tr, norm=norm_method)
            X_val, mean, std, feat_filt = self.normalize(X_val, mean, std, feat_filt=feat_filt, norm=norm_method)
            X_train, mean, std, feat_filt = self.normalize(X_train, norm=norm_method)
            X_test, mean, std, feat_filt = self.normalize(X_test, mean, std, feat_filt=feat_filt, norm=norm_method)

        print("Dumping result..")
        pickle.dump((X_tr, X_val, X_train, X_test, y_tr, y_val, y_train, y_test),
            open('Code/test%dval%dnorm%s.p'%(self.test_fold, self.val_fold, norm_method), 'wb'))
        
        return

# Initialisierung
preprocessor = Preprocessor(test_fold=0, val_fold=1)
# Daten laden und verarbeiten
data = preprocessor.create_splits(norm_method='tanh_norm')