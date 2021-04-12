import numpy as np

class EnsembleRegressor():  
    """
    This module implements a custom ensemble model.
    The training procedure on train set is as follows:
        * splits the train set into D1 and D2.(50-50)
        * now from this D1 sampling is done with replacement 
          to create d1,d2,d3....dk(k samples)
        * k DecisionTree models are now trained on each of these k samples
        (k can be considered as a hyperparameter)
        * now the set aside D2 is passed to the k trained models to obtain a k-dimensional feature set
        * with the help of these feature set along with D2 targets, a metalearner is trained
          which is also a decision tree. This metalearner is our actual model and rest of the base just
          baselearner can be considered as feature extractors
    """
    
    def __init__(self, n_learners = 10, meta_learner = None, oob_size=0.5, max_sample_ratio=None, meta_rs=False, meta_params=None):
        self.n_learners = n_learners
        self.oob_size = oob_size
        self.max_samples = max_samples_ratio if max_sample_ratio is not None else 0.2
        self.tree_list = [DecisionTreeRegressor() for i in range(self.n_learners)]
        
        self.meta_rs = meta_rs
        
        
        
        if meta_learner is None or meta_learner == 'decision_tree':
            self.meta_learner = DecisionTreeRegressor()
        elif meta_learner == 'random_forest':
            self.meta_learner = RandomForestRegressor()
        elif meta_learner == 'xgboost':
            self.meta_learner = XGBRegressor()
        elif meta_learner == 'svr':
            self.meta_learner = SVR()
        elif meta_learner == 'kernel_ridge':
            self.meta_learner = KernelRidge()
        elif meta_learner == 'bayesian_ridge':
            self.meta_learner = BayesianRidge()
            
        if self.meta_rs:
            if not isinstance(meta_params, dict):
                raise ValueError("Hyperparameter Search Mode requires a dictionary of parameters")
            else:
                self.meta_params = meta_params
                self.rs_obj = RandomizedSearchCV(self.meta_learner, self.meta_params, cv=5, n_iter=3, n_jobs=3)
            
        return None
    
    
    def _create_sample(self,X,y,fraction):
        X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        indices = random.sample(range(len(X)), int(fraction*len(X)))
        return X[indices], y[indices]
    
    def fit(self, X, y):
        
        X_D1, X_D2, y_D1, y_D2 = train_test_split(X,y,test_size=self.oob_size)
        
        D2_predlist = []
        
        for i in range(self.n_learners):
            X_temp, y_temp = self._create_sample(X_D1, y_D1, self.max_samples)
            self.tree_list[i].fit(X_temp, y_temp)
            preds = self.tree_list[i].predict(X_D2)
            
            D2_predlist.append(preds)
        
        new_feature_set = np.stack(D2_predlist, axis=1)
        
        if self.meta_rs:
            self.rs_obj.fit(new_feature_set, y_D2)
            self.meta_learner = self.rs_obj.best_estimator_
        
        self.meta_learner.fit(new_feature_set, y_D2)
        return self
        
    def predict(self, X):
        
        D2_predlist = []
        
        for i in range(self.n_learners):
            preds = self.tree_list[i].predict(X)    
            D2_predlist.append(preds)
        
        new_feature_set = np.stack(D2_predlist, axis=1)
        return self.meta_learner.predict(new_feature_set)