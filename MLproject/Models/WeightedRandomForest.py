'''
Created on Jan 6, 2019

@author: mofir
'''

class WRF(object):
    '''
    classdocs
    '''

    def __init__(self, n_trees=100, max_depth=5, n_features=None, type="cat", weight_type="div"):
    '''
    init a WRF classifier with the following parameters:
    n_trees: the number of trees to use.
    max_depth: the depth of each tree (will be passed along to DecisionTreeClassifier/DecisionTreeRegressor).
    n_features: the number of features to use for every split. The number should be given to DecisionTreeClassifier/Regressor as max_features.
    type: "cat" for categorization and "reg" for regression.
    weight_type: the tree weighting technique. 'div' for 1/error and 'sub' for 1-error.
    '''
    self.n_trees = n_trees
    self.max_depth = max_depth
    self.n_features = n_features
    self.type = type
    self.weight_type = weight_type
    self.weights = []

  def fit(self, X, y):
    '''
      fit the classifier for the data X with response y.
    '''
    # <Your Code if needed>
    n_trees = self.n_trees
    self.trees = []
    self.weights = []
    weights_list = []
    for n in range(n_trees):
      tree = self.build_tree()
      self.trees.append(tree)
      X_tree, y_tree, X_oob, y_oob = self.bootstrap(X, y)
      tree.fit(X_tree, y_tree)
      weight = self.calculate_weight(tree, X_oob, y_oob)
      weights_list.append(weight)

      # Normalize the weights so they sum to 1
      # <Your code goes here>

    self.weights = weights_list
    self.weights = self.weights / np.sum(self.weights)

  def build_tree(self):
    tree = None
    if self.type == "cat":
      tree = DecisionTreeClassifier(max_depth=self.max_depth, max_features=self.n_features)
    else:
      tree = DecisionTreeRegressor(max_depth=self.max_depth, max_features=self.n_features)
    return tree

  def bootstrap(self, X, y):
    '''
      This method creates a bootstrap of the dataset (uniformly sample len(X) samples from X with repetitions).
      It returns X_tree, y_tree, X_oob, y_oob.
      X_tree, y_tree are the bootstrap collections for the given X and y.
      X_oob, y_oob are the out of bag remaining instances (the ones that were not sampled as part of the bootstrap)
    '''
    # <Your code goes here>
    x_tree_ids = []
    x_oob_idx = []

    x_len = len(X.index)
    x_idx_list = np.arange(x_len)
    x_tree_ids = np.random.choice(x_idx_list,x_len)

    i=0
    for i in range(x_len):
      if i not in x_tree_ids:
        x_oob_idx.append(i)

    def id2val(x_idx,X):
      x_idx_val = pd.concat([X.iloc[[id]] for id in x_idx],ignore_index=True)
      return x_idx_val

    X_tree = id2val(x_tree_ids,X)
    y_tree = id2val(x_tree_ids,y)
    X_oob  = id2val(x_oob_idx,X)
    y_oob  = id2val(x_oob_idx,y)

    return X_tree, y_tree, X_oob, y_oob

  def calculate_weight(self, tree, X_oob, y_oob):
    '''
      This method calculates a weight for the given tree, based on it's performance on
      the OOB instances. We support two different types:
      if self.weight_type == 'div', we should return 1/error and if it's 'sub' we should
      return 1-error. The error is the normalized error rate of the tree on OOB instances.
      For classification use 0/1 loss error (i.e., count 1 for every wrong OOB instance and divide by the numbner of OOB instances),
      and for regression use mean square error of the OOB instances.
    '''
    # < Your code goes here>
    y_pred = tree.predict(X_oob)
    epsilon = 10^-6

    y_oob = y_oob.values.T.tolist()
    y_oob = y_oob[0]

    if self.type == "cat":
      y_error = np.equal(y_pred,y_oob)
      error = y_error.sum()/len(y_oob)
    else:
      error = sklearn.metrics.mean_squared_error(y_oob,y_pred)

    if self.weight_type == 'div':
      return 1/(error+epsilon) #to avoid 0 division
    else:
        return 1 - error

  def predict(self, X):
    '''
      Predict the label/value of the given instances in the X matrix.
      For classification you should implement weighted voting, and for regression implement weighted mean.
      Return a list of predictions for the given instances.
    '''
    # <Your code goes here>
    total_weighted_prediction = []
    y_pred_list = []

    if self.type == "reg":
      total_weighted_prediction = numpy.zeros(len(X))
    for i in range(self.n_trees):
      curr_tree = self.trees[i]
      y_pred = curr_tree.predict(X)

      if self.type == "cat":
        y_pred_list.append(y_pred)
      else:
        weighted_pred = self.weights[i] * y_pred
        total_weighted_prediction = total_weighted_prediction + weighted_pred


    y_pred_matrix = np.transpose(y_pred_list)
    shape_matrix = np.shape(y_pred_matrix)

    if self.type == "cat":
      for i in range(len(X.index)):
        counts = {} #defaultdict(int)

        for idx,value in enumerate(y_pred_matrix[i]):
          if value in counts:
              counts[value] += self.weights[idx]
          else:
              counts[value] = self.weights[idx]

        max_vote_pred = max(counts.items(), key=operator.itemgetter(1))[0]
        total_weighted_prediction.append(max_vote_pred)

    return total_weighted_prediction

