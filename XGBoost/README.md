References for Tuning Parameter:
> 1. https://www.dataiku.com/learn/guide/code/python/advanced-xgboost-tuning.html
> 2. https://blog.csdn.net/u010657489/article/details/51952785

XGBoost has a large number of advanced parameters, which can all affect the quality and speed of your model.
- booster[default = gbtree]：選擇每次迭代的模型，有兩種選擇，gbtree：基於樹的模型，gbliner：線性模型
- max_depth : int,
    Maximum tree depth for base learners.
- learning_rate : float,
    Boosting learning rate (XGBoost's "eta")
- n_estimators : int,
    Number of boosted trees to fit.
- silent : boolean,
    Whether to print messages while running boosting.
- objective : string,
    Specify the learning task and the corresponding learning objective.
- nthread : int,
    Number of parallel threads used to run XGBoost.
- gamma : float,
    Minimum loss reduction required to make a further partition
    on a leaf node of the tree.
- min_child_weight : int,
    Minimum sum of instance weight(hessian) needed in a child.這個參數用於避免過擬合。當它的值較大時，可以避免模型學習到局部的特殊樣本。
- max_delta_step : int,
    Maximum delta step we allow each tree's weight estimation to be.
- subsample : float,
    Subsample ratio of the training instance.
- colsample_bytree : float,
    Subsample ratio of columns when constructing each tree.
- base_score:
    The initial prediction score of all instances, global bias.
- seed : int,
    Random number seed.
- missing : float, optional,
    Value in the data which needs to be present as a missing value.
    If None, defaults to np.nan.
	
XGBoost can take a sparse matrix as input. This allows you to convert categorical variables with high cardinality into a dummy matrix, then build a model without getting an out of memory error!
