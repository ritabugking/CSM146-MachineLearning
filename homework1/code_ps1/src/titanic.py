"""
Author      : Yi-Chieh Wu, Sriram Sankararaman
Description : Titanic
"""

# Use only the provided packages!
import math
import csv
from util import *
from collections import Counter

from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics

######################################################################
# classes
######################################################################

class Classifier(object) :
    """
    Classifier interface.
    """
    
    def fit(self, X, y):
        raise NotImplementedError()
        
    def predict(self, X):
        raise NotImplementedError()


class MajorityVoteClassifier(Classifier) :
    
    def __init__(self) :
        """
        A classifier that always predicts the majority class.
        
        Attributes
        --------------------
            prediction_ -- majority class
        """
        self.prediction_ = None
    
    def fit(self, X, y) :
        """
        Build a majority vote classifier from the training set (X, y).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes
        
        Returns
        --------------------
            self -- an instance of self
        """
        majority_val = Counter(y).most_common(1)[0][0]
        self.prediction_ = majority_val
        return self
    
    def predict(self, X) :
        """
        Predict class values.
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
        
        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.prediction_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        
        n,d = X.shape
        y = [self.prediction_] * n 
        return y


class RandomClassifier(Classifier) :
    
    def __init__(self) :
        """
        A classifier that predicts according to the distribution of the classes.
        
        Attributes
        --------------------
            probabilities_ -- class distribution dict (key = class, val = probability of class)
        """
        self.probabilities_ = None
    
    def fit(self, X, y) :
        """
        Build a random classifier from the training set (X, y).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes
        
        Returns
        --------------------
            self -- an instance of self
        """
        
        ### ========== TODO : START ========== ###
        # part b: set self.probabilities_ according to the training set
        if Counter(y).most_common(2)[0][0]==1:
            self.probabilities_ = float(Counter(y).most_common(2)[0][1])/float(n)
        else:
            self.probabilities_ = float(Counter(y).most_common(2)[1][1])/float(n)
        
        
        
        ### ========== TODO : END ========== ###
        
        return self
    
    def predict(self, X, seed=1234) :
        """
        Predict class values.
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            seed -- integer, random seed
        
        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.probabilities_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        np.random.seed(seed)
        
        ### ========== TODO : START ========== ###
        n,d = X.shape
        #y = (np.random.rand(n,1)<self.probabilities_)*1
        y = np.random.choice(2, n, p=[1-self.probabilities_, self.probabilities_]).tolist()
        return y
        # part b: predict the class for each test example
        # hint: use np.random.choice (be careful of the parameters)
        
        ### ========== TODO : END ========== ###
        
      


######################################################################
# functions
######################################################################
def plot_histograms(X, y, Xnames, yname) :
    n,d = X.shape  # n = number of examples, d =  number of features
    fig = plt.figure(figsize=(20,15))
    nrow = 3; ncol = 3
    for i in xrange(d) :
        fig.add_subplot (3,3,i)  
        data, bins, align, labels = plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname, show = False)
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xnames[i])
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
 
    plt.savefig ('histograms.pdf')


def plot_histogram(X, y, Xname, yname, show = True) :
    """
    Plots histogram of values in X grouped by y.
    
    Parameters
    --------------------
        X     -- numpy array of shape (n,d), feature values
        y     -- numpy array of shape (n,), target classes
        Xname -- string, name of feature
        yname -- string, name of target
    """
    
    # set up data for plotting
    targets = sorted(set(y))
    data = []; labels = []
    for target in targets :
        features = [X[i] for i in xrange(len(y)) if y[i] == target]
        data.append(features)
        labels.append('%s = %s' % (yname, target))
    
    # set up histogram bins
    features = set(X)
    nfeatures = len(features)
    test_range = range(int(math.floor(min(features))), int(math.ceil(max(features)))+1)
    if nfeatures < 10 and sorted(features) == test_range:
        bins = test_range + [test_range[-1] + 1] # add last bin
        align = 'left'
    else :
        bins = 10
        align = 'mid'
    
    # plot
    if show == True:
        plt.figure()
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xname)
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
        plt.show()

    return data, bins, align, labels


def error(clf, X, y, ntrials=100, test_size=0.2) :
    """
    Computes the classifier error over a random split of the data,
    averaged over ntrials runs.
    
    Parameters
    --------------------
        clf         -- classifier
        X           -- numpy array of shape (n,d), features values
        y           -- numpy array of shape (n,), target classes
        ntrials     -- integer, number of trials
    
    Returns
    --------------------
        train_error -- float, training error
        test_error  -- float, test error
    """
    
    ### ========== TODO : START ========== ###
    train_error = 0
    test_error = 0
    for x in range(0, ntrials):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size, random_state=ntrials)
    
    
        clf.fit(X_train, y_train)                  # fit training data using the classifier
        y_pred = clf.predict(X_test)        # take the classifier and run it on the training data
        test_error = test_error + 1 - metrics.accuracy_score(y_test, y_pred, normalize=True)
    
        y_pred2 = clf.predict(X_train)        # take the classifier and run it on the training data
        train_error = train_error + 1 - metrics.accuracy_score(y_train, y_pred2, normalize=True)
    # compute cross-validation error over ntrials
    # hint: use train_test_split (be careful of the parameters)
    train_error = train_error/float(ntrials)
    test_error = test_error/float(ntrials)
        
    ### ========== TODO : END ========== ###
    
    return train_error, test_error


def write_predictions(y_pred, filename, yname=None) :
    """Write out predictions to csv file."""
    out = open(filename, 'wb')
    f = csv.writer(out)
    if yname :
        f.writerow([yname])
    f.writerows(zip(y_pred))
    out.close()


######################################################################
# main
######################################################################

def main():
    # load Titanic dataset
    titanic = load_data("titanic_train.csv", header=1, predict_col=0)
    X = titanic.X; Xnames = titanic.Xnames
    y = titanic.y; yname = titanic.yname
    n,d = X.shape  # n = number of examples, d =  number of features
    
    
    
    #========================================
    # part a: plot histograms of each feature
    print 'Plotting...'
    for i in xrange(d) :
        plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname)

       
    #========================================
    # train Majority Vote classifier on data
    print 'Classifying using Majority Vote...'
    clf = MajorityVoteClassifier() # create MajorityVote classifier, which includes all model parameters
    clf.fit(X, y)                  # fit training data using the classifier
    y_pred = clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print '\t-- training error: %.3f' % train_error
    
    
    
    ### ========== TODO : START ========== ###
    # part b: evaluate training error of Random classifier
    print 'Classifying using Random...'
    clf = RandomClassifier() # create MajorityVote classifier, which includes all model parameters
    clf.fit(X, y)                  # fit training data using the classifier
    y_pred = clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print '\t-- training error: %.3f' % train_error
    
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part c: evaluate training error of Decision Tree classifier
    # use criterion of "entropy" for Information gain 
    print 'Classifying using Decision Tree...'
    clf = DecisionTreeClassifier(criterion="entropy")
    clf.fit(X, y)                  # fit training data using the classifier
    y_pred = clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print '\t-- training error: %.3f' % train_error
    ### ========== TODO : END ========== ###
    
    
    
    # note: uncomment out the following lines to output the Decision Tree graph
    """
    # save the classifier -- requires GraphViz and pydot
    import StringIO, pydot
    from sklearn import tree
    dot_data = StringIO.StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=Xnames)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("dtree.pdf") 
    """
    
    
    
    ### ========== TODO : START ========== ###
    # part d: use cross-validation to compute average training and test error of classifiers
    print 'Investigating various classifiers...'
    clf = MajorityVoteClassifier() # create MajorityVote classifier, which includes all model parameters   
    train_error, test_error = error(clf, X, y)
        
    print 'Classifying using Majority Vote... -- average training error: %.3f' % train_error
    print '                                   -- average testing error: %.3f' % test_error
    
    clf = RandomClassifier() # create MajorityVote classifier, which includes all model parameters
    train_error, test_error = error(clf, X, y)
    print 'Classifying using Random...       -- average training error: %.3f' % train_error
    print '                                  -- average testing error: %.3f' % test_error
    
    clf = DecisionTreeClassifier(criterion="entropy")
    train_error, test_error = error(clf, X, y)
    print 'Classifying using Decision Tree... -- average training error: %.3f' % train_error
    print '                                   -- average testing error: %.3f' % test_error
    
    
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part e: investigate decision tree classifier with various depths
    print 'Investigating depths...'
    train_array = np.array([])
    test_array = np.array([])
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    for i in range(0, 20):
        depth=i+1
        clf = DecisionTreeClassifier(criterion="entropy", max_depth=depth)
        
        train_error, test_error = error(clf, X, y)
            
    
        train_array = np.append(train_array,[train_error])
        test_array = np.append(test_array,[test_error])
    
    
    x1 = np.linspace(1, 20, num=20)
    plt.plot(x1,train_array, label='decision tree train')
    plt.plot(x1,test_array, label='decision tree test')
    plt.xlabel('Depth')
    plt.ylabel('Error')
    
    
    
    ###############      baseline classifier 1   ######################
    train_array = np.array([])
    test_array = np.array([])
    for i in range(0, 20):
        depth=i+1
        clf = MajorityVoteClassifier()
        
        train_error, test_error = error(clf, X, y)
            
    
        train_array = np.append(train_array,[train_error])
        test_array = np.append(test_array,[test_error])
    
    
    #x1 = np.linspace(1, 20, num=20)
    #plt.plot(x1,train_array)
    plt.plot(x1,test_array, label='majority vote test')
    
    
    ###############      baseline classifier 2   ######################
    train_array = np.array([])
    test_array = np.array([])
    for i in range(0, 20):
        depth=i+1
        clf = RandomClassifier()
        
        train_error, test_error = error(clf, X, y)          
    
        train_array = np.append(train_array,[train_error])
        test_array = np.append(test_array,[test_error])
    
    
    #x1 = np.linspace(1, 20, num=20)
    #plt.plot(x1,train_array)
    plt.plot(x1,test_array, label='random test')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    
    
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part f: investigate decision tree classifier with various training set sizes
    print 'Investigating training set sizes...'
    train_array = np.array([])
    test_array = np.array([])
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    #depth=1
    #depth=6
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=1)
    
        
       
    for i in range(0, 19):
        
        train_error, test_error = error(clf, X, y, 100, 0.95-0.05*i)
            
        train_array = np.append(train_array,[train_error])
        test_array = np.append(test_array,[test_error])
    
    
    
    
    x1 = np.linspace(0.05, 0.95, num=19)
    plt.plot(x1,train_array, label='decision tree train')
    plt.plot(x1,test_array, label='decision tree test')
    plt.xlabel('percentage of training data (%)')
    plt.ylabel('Error')
    
    
    ###############      baseline classifier 1   ######################
    train_array = np.array([])
    test_array = np.array([])
    clf = MajorityVoteClassifier()
    for i in range(0, 19):
        
        train_error, test_error = error(clf, X, y, ntrials=100, test_size=0.95-0.05*i)
         
    
        train_array = np.append(train_array,[train_error])
        test_array = np.append(test_array,[test_error])
    
    
    #x1 = np.linspace(0.05, 0.95, num=19)
    #plt.plot(x1,train_array)
    plt.plot(x1,test_array, label='majority vote test')
    
    ###############      baseline classifier 2   ######################
    train_array = np.array([])
    test_array = np.array([])
    clf = RandomClassifier()
    for i in range(0, 19):
        
        train_error, test_error = error(clf, X, y, ntrials=100, test_size=0.95-0.05*i)           
    
        train_array = np.append(train_array,[train_error])
        test_array = np.append(test_array,[test_error])
    
    
    #x1 = np.linspace(0.05, 0.95, num=19)
    #plt.plot(x1,train_array)
    plt.plot(x1,test_array, label='random test')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    ### ========== TODO : END ========== ###
    
       
    print 'Done'


if __name__ == "__main__":
    main()
