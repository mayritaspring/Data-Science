XGBoost has a large number of advanced parameters, which can all affect the quality and speed of your model.

- max_depth : int
    Maximum tree depth for base learners.
- learning_rate : float
    Boosting learning rate (XGBoost's "eta")
- n_estimators : int
    Number of boosted trees to fit.
- silent : boolean
    Whether to print messages while running boosting.
- objective : string
    Specify the learning task and the corresponding learning objective.
- nthread : int
    Number of parallel threads used to run XGBoost.
- gamma : float
    Minimum loss reduction required to make a further partition
    on a leaf node of the tree.
- min_child_weight : int
    Minimum sum of instance weight(hessian) needed in a child.
- max_delta_step : int
    Maximum delta step we allow each tree's weight estimation to be.
- subsample : float
    Subsample ratio of the training instance.
- colsample_bytree : float
    Subsample ratio of columns when constructing each tree.
- base_score:
    The initial prediction score of all instances, global bias.
- seed : int
    Random number seed.
- missing : float, optional
    Value in the data which needs to be present as a missing value.
    If None, defaults to np.nan.