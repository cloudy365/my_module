# -*- coding: utf-8 -*-
"""
module name: zyz_ml
description: Contains the common used statistical codes (including ML) used in my research,
             Functions arranged by function types.
"""

from .. import np
import statsmodels.api as sm
from scipy.stats import linregress, norm
#from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score
from sklearn import svm


#####
##  常用统计函数  ##
#####
def rmse(x, y):
    """
    Calculate the RMSE of two arrays.
    Square-root of the variance of (x-y)/y
    """
    return np.sqrt(np.nanvar( (x-y)/y ))



#####
##  时间序列分析 ##
#####
def detnd(array_in, order):
    """
    This array should be a 1-D array.
    """
    x = np.arange(len(array_in))
    f = np.poly1d(np.polyfit(x, array_in, order))
    y = f(x)
    delta_y = array_in - y

    return delta_y, y


    
#####
##  回归分析  ##
#####
def reg_1var(x, y, summary=False):
    """
    Return the linear regression equation.
    x should have 1 dimensions.
    y should have 1 dimension.
    """
    # method 1, using scipy's linregress
    r = linregress(x, y)
    # method 2, using statsmodels's OLS
    X = sm.add_constant(x)
    res_ols = sm.OLS(y, X).fit()
    if summary:
        res_ols.summary()
    return res_ols.params


def reg_2var(x, y, summary=False):
    """
    Return the binary linear regression equation.
    x should have 2 dimensions.
    y should have 1 dimension.
    """
    ones = np.ones(len(x[0]))
    X = sm.add_constant(np.column_stack((x[0], ones)))
    for ele in x[1:]:
        X = sm.add_constant(np.column_stack((ele, X)))
    results = sm.OLS(y, X).fit()
    if summary:
        print results.summary()
    return results.params


def tf_reg_anyvar(x_, y_, n_var, batch_frac=0.05, iterations=10000):
    """
    This is test function that uses tensorflow to do the linear regression for
    both single and multivariables. 
    The input (x, y) also can be treated as a training dataset.
    x_ should be in [None, n] shape where n is the number of variables;
    y_ should be in [None, 1] shape;
    n_var defines the number of variables;
    batch_frac defines the fraction of the total data in one batch.
    The regression function is assumed as: y = W*x + b.
    """
    tf.reset_default_graph() # clear symbolic graph
    
    # check input x, y
    if y_.shape[1] != 1 or x_.shape[1] != n_var:
        print "Please check input x_ or y_."
        print "Function terminated..."
        return
    else:
        x = tf.placeholder(tf.float32, [None, n_var], name='x')
        y_true = tf.placeholder(tf.float32, [None, 1], name='y')
        
        W = tf.Variable(tf.zeros([n_var, 1]))
        b = tf.Variable(tf.zeros([1]))
        
        y_pred = tf.matmul(x, W) + b
        cost = tf.reduce_sum(tf.square(y_true-y_pred))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(cost)
        
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        
        for i in range(iterations):
            x_batch, y_batch = MLhelper_next_batch(x_, y_)
            feed = {x:x_batch, y_true:y_batch}
            sess.run(optimizer, feed_dict=feed)
    
    # calculate RMSE for the whole dataset
    feed = {x:x_}
    y_pred_all = sess.run(y_pred, feed_dict=feed)
    print "RMSE is {:.2f}%".format(rmse(y_pred_all[:, 0], y_[:, 0])*100.)
    return sess.run(W), sess.run(b)


def tf_reg_logistic(x_, y_, n_var, n_cls, batch_frac=0.05, iterations=10000):
    """
    This is test function that uses tensorflow to do the logistic regression. 
    The input (x, y) also can be treated as a training dataset.
    x_ should be in [None, n_var] shape where n_var is the number of variables;
    y_ should be in [None, n_cls] shape where n_cls is the number of classifications;
    batch_frac defines the fraction of the total data in one batch.
    The regression function is assumed as: y = W*x + b.
    """
    tf.reset_default_graph() # clear symbolic graph
    
    # check input x, y
    if y_.shape[1] != n_cls or x_.shape[1] != n_var:
        print "Please check input x_ or y_."
        print "Function terminated..."
        return
    else:
        x = tf.placeholder(tf.float32, [None, n_var], name='x')
        y_true = tf.placeholder(tf.float32, [None, n_cls], name='y_true')
        y_true_cls = tf.placeholder(tf.int64, [None])
        
        W = tf.Variable(tf.zeros([n_var, n_cls]))
        b = tf.Variable(tf.zeros([n_cls]))
        
        logits = tf.matmul(x, W) + b
        # y_pred = tf.matmul(x, W) + b
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)
        cost = tf.reduce_mean(cross_entropy)
        # cost = tf.reduce_sum(tf.square(y-y_pred))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)
        
        # y_pred = tf.nn.softmax(logits)
        # y_pred_cls = tf.argmax(y_pred, axis=1)
        # correct_prediction = tf.equal(y_pred_cls, y_true_cls)
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        
        for i in range(iterations):
            x_batch, y_batch = MLhelper_next_batch(x_, y_)
            feed = {x:x_batch, y_true:y_batch}
            sess.run(optimizer, feed_dict=feed)
    
    # calculate RMSE for the whole dataset
    # feed = {x:x_}
    # y_pred_all = sess.run(y_pred, feed_dict=feed)
    # print "RMSE is {:.2f}%".format(rmse(y_pred_all[:, 0], y_[:, 0])*100.)
    return sess.run(W), sess.run(b)
    
    
def reg_weightedvar(x, y, weights):
    """
    TBD
    """
    X = sm.add_constant(x)
    mod_wls = sm.WLS(y, X, weights=weights)
    res_wls = mod_wls.fit()
    print(res_wls.summary())    
    return res_wls.params


#####
##  逻辑回归分析  ##
#####
def logistic_regression(train_X, train_y, verbose=False):
    """
    LogisticRegressionCV, which refers to http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html
    Default setting (currently used):
        Cs=10, 
        fit_intercept=True, 
        cv=None, 
        dual=False, 
        penalty=’l2’, 
        scoring=None, 
        solver=’lbfgs’, 
        tol=0.0001, 
        max_iter=100, 
        class_weight=None, 
        n_jobs=1, 
        verbose=0, 
        refit=True, 
        intercept_scaling=1.0, 
        multi_class=’ovr’, 
        random_state=None)
    """
    lr = LogisticRegressionCV()
    lr.fit(train_X, train_y)
    if verbose:
        print "Use lr.score(test_X, test_y) to show the performance of training.\n\
                or lr.predict(X) to predict class labels for samples in X.\n\
                or lr.get_params() to get parameters for this estimator."
    return lr
    
    

#####
##  支持向量机（SVM）分析  ##
#####
def svm(train_X, train_y):
    clf = svm.SVC()
    clf.fit(train_X, train_y)
    print "Use clf.score(test_X, test_y) to show the performance of training."
    return clf
    
    
    
#####
##  机器学习辅助函数  ##
#####
def MLhelper_next_batch(x_, y_, batch_size=64):
    length_data = x_.shape[0]
    # batch_nod = int(length_data*0.05) # number of data (nod) in a batch
    batch_idx = np.random.randint(0, length_data, batch_size)
    x_batch = x_[batch_idx]
    y_batch = y_[batch_idx]
    return x_batch, y_batch
    

def MLhelper_one_hot_encode_object_array(arr, num_of_class=None):
    '''One hot encode a numpy array of objects (e.g. strings)'''
    
    def MLhelper_to_categorical(y, num_classes=None):
        """
        Copied from .../keras/utils/np_utils.py.
   	
        Converts a class vector (integers) to binary class matrix.
   	
   		E.g., for use with categorical_crossentropy.
    
        # Arguments
        y: class vector to be converted into a matrix
 			(integers from 0 to num_classes)
        num_classes: total number of classes
        # Returns
        A binary matrix representation of the input
        """
        y = np.array(y, dtype='int').ravel()
        if not num_classes:
            num_classes = np.max(y) + 1
        n = y.shape[0]
        categorical = np.zeros((n, num_classes))
        categorical[np.arange(n), y] = 1
        return categorical
    
    uniques, ids = np.unique(arr, return_inverse=True)
    return MLhelper_to_categorical(ids, num_of_class)


    
    
