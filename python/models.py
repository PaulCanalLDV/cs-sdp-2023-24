import pickle
from abc import abstractmethod
from sklearn.cluster import KMeans
import numpy as np
from gurobipy import *


class BaseModel(object):
    """
    Base class for models, to be used as coding pattern skeleton.
    Can be used for a model on a single cluster or on multiple clusters"""

    def __init__(self):
        """Initialization of your model and its hyper-parameters"""
        pass

    @abstractmethod
    def fit(self, X, Y):
        """Fit function to find the parameters according to (X, Y) data.
        (X, Y) formatting must be so that X[i] is preferred to Y[i] for all i.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        # Customize what happens in the fit function
        return

    @abstractmethod
    def predict_utility(self, X):
        """Method to call the decision function of your model

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements

        Returns
        -------
        np.ndarray:
            (n_samples, n_clusters) array of decision function value for each cluster.
        """
        # Customize what happens in the predict utility function
        return

    def predict_preference(self, X, Y):
        """Method to predict which pair is preferred between X[i] and Y[i] for all i.
        Returns a preference for each cluster.

        Parameters
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements to compare with Y elements of same index
        Y: np.ndarray
            (n_samples, n_features) list of features of elements to compare with X elements of same index

        Returns
        -------
        np.ndarray:
            (n_samples, n_clusters) array of preferences for each cluster. 1 if X is preferred to Y, 0 otherwise
        """
        X_u = self.predict_utility(X)
        Y_u = self.predict_utility(Y)

        return (X_u - Y_u > 0).astype(int)

    def predict_cluster(self, X, Y):
        """Predict which cluster prefers X over Y THE MOST, meaning that if several cluster prefer X over Y, it will
        be assigned to the cluster showing the highest utility difference). The reversal is True if none of the clusters
        prefer X over Y.
        Compared to predict_preference, it indicates a cluster index.

        Parameters
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements to compare with Y elements of same index
        Y: np.ndarray
            (n_samples, n_features) list of features of elements to compare with X elements of same index

        Returns
        -------
        np.ndarray:
            (n_samples, ) index of cluster with highest preference difference between X and Y.
        """
        X_u = self.predict_utility(X)
        Y_u = self.predict_utility(Y)

        return np.argmax(X_u - Y_u, axis=1)

    def save_model(self, path):
        """Save the model in a pickle file. Don't hesitate to change it in the child class if needed

        Parameters
        ----------
        path: str
            path indicating the file in which the model will be saved
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(clf, path):
        """Load a model saved in a pickle file. Don't hesitate to change it in the child class if needed

        Parameters
        ----------
        path: str
            path indicating the path to the file to load
        """
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model


class RandomExampleModel(BaseModel):
    """Example of a model on two clusters, drawing random coefficients.
    You can use it to understand how to write your own model and the data format that we are waiting for.
    This model does not work well but you should have the same data formatting with TwoClustersMIP.
    """

    def __init__(self):
        self.seed = 444
        self.weights = self.instantiate()

    def instantiate(self):
        """No particular instantiation"""
        return []

    def fit(self, X, Y):
        """fit function, sets random weights for each cluster. Totally independant from X & Y.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        np.random.seed(self.seed)
        num_features = X.shape[1]
        weights_1 = np.random.rand(num_features) # Weights cluster 1
        weights_2 = np.random.rand(num_features) # Weights cluster 2

        weights_1 = weights_1 / np.sum(weights_1)
        weights_2 = weights_2 / np.sum(weights_2)
        self.weights = [weights_1, weights_2]
        return self

    def predict_utility(self, X):
        """Simple utility function from random weights.

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements

        Returns
        -------
        np.ndarray:
            (n_samples, n_clusters) array of decision function value for each cluster.
        """
        u_1 = np.dot(X, self.weights[0]) # Utility for cluster 1 = X^T.w_1
        u_2 = np.dot(X, self.weights[1]) # Utility for cluster 2 = X^T.w_2
        return np.stack([u_1, u_2], axis=1) # Stacking utilities over cluster on axis 1


class TwoClustersMIP(BaseModel):
    """Skeleton of MIP you have to write as the first exercise.
    You have to encapsulate your code within this class that will be called for evaluation.
    """

    def __init__(self, n_pieces, n_clusters, n_criteria, n_pairs):
        """Initialization of the MIP Variables

        Parameters
        ----------
        n_pieces: int
            Number of pieces for the utility function of each feature.
        n_clusters: int
            Number of clusters to implement in the MIP.
        n_criteria: int
            Number of criteria for each product
        n_pairs: int
            Number of pairs of comparison
        """
        self.seed = 123
        self.n_pieces = n_pieces # L
        self.n_clusters = n_clusters # K
        self.n_criteria = n_criteria # n
        self.n_pairs = n_pairs # P
        self.epsilon = 10**-10
        self.M = 5
        self.model = self.instantiate()

    def instantiate(self):
        """Instantiation of the MIP Variables - Completed!"""

        model = Model("TwoClustersMIP")

        # The values of each breakpoint in our function ((L + 1) * n * K)
        self.weights = [[[model.addVar(name=f"s_{k}_{i}_{l}", vtype=GRB.CONTINUOUS, lb=0, ub=1) for l in range(self.n_pieces+1)] for i in range(self.n_criteria)] for k in range(self.n_clusters)]# liste de poids, par clusters, par critères, par morceaux

        # The overestimation and underestimation errors sigma_x+/sigma_y+ and sigma_x-/sigma_y-, made for every pair
        self.sigma_x_plus = [model.addVar(name=f"sigma_x+_{j}", vtype=GRB.CONTINUOUS) for j in range(self.n_pairs)]
        self.sigma_x_minus = [model.addVar(name=f"sigma_x-_{j}", vtype=GRB.CONTINUOUS) for j in range(self.n_pairs)]
        self.sigma_y_plus = [model.addVar(name=f"sigma_y+_{j}", vtype=GRB.CONTINUOUS) for j in range(self.n_pairs)]
        self.sigma_y_minus = [model.addVar(name=f"sigma_y-_{j}", vtype=GRB.CONTINUOUS) for j in range(self.n_pairs)]

        # The binary variables, to caracterise the fact that x > y
        self.preferences = [[model.addVar(name=f"preferences_{j}_{k}", vtype=GRB.BINARY) for k in range(self.n_clusters)] for j in range(self.n_pairs)]

        model.update()

        return model

    def score(self, x, k, constraint=True):
        value = (lambda x: x) if constraint else (lambda x: x.X) # to get constraint or to get the value after the model have been optimised
        score = 0
        max = 1 # we make the assumption that the maximum value for x is 1

        for i in range(self.n_criteria):
            if x[i] == max:
                score += self.weights[k][i][-1] # if the value of x for criteria i is the maximum, we get the last value directly
            else:
                l = int(x[i] * self.n_pieces)
                x_l = l / self.n_pieces
                x_l_plus_1 = (l + 1) / self.n_pieces
                score += value(self.weights[k][i][l]) + ((x[i] - x_l) / (x_l_plus_1 - x_l)) * (value(self.weights[k][i][l+1]) - value(self.weights[k][i][l]))

        return score

    def fit(self, X, Y):
        """Estimation of the parameters - Completed!

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """

        # Constraint n°1: Constraint on the errors for each pair

        for j in range(self.n_pairs):
            x = X[j]
            y = Y[j]
            for k in range(self.n_clusters):
                score_x = self.score(x, k)
                score_y = self.score(y, k)
                self.model.addConstr((1 - self.preferences[j][k]) * self.M + score_x - self.sigma_x_plus[j] + self.sigma_x_minus[j] >= score_y - self.sigma_y_plus[j] + self.sigma_y_minus[j] + self.epsilon)

        # Constraint n°2: Monotony of functions
                
        for k in range(self.n_clusters):
            for i in range(self.n_criteria):
                for l in range(self.n_pieces):
                    self.model.addConstr(self.weights[k][i][l+1] >= self.weights[k][i][l])

        # Constraint n°3: Each function begins at 0
                    
        for k in range(self.n_clusters):
            for i in range(self.n_criteria):
                self.model.addConstr(self.weights[k][i][0] == 0)

        # Constraint n°4: Normalisation
                
        for k in range(self.n_clusters):
            self.model.addConstr(quicksum([self.weights[k][i][self.n_pieces] for i in range(self.n_criteria)]) == 1)

        # Constraint n°5: At least one cluster have the preference
        
        for j in range(self.n_pairs):
            self.model.addConstr(quicksum(self.preferences[j]) >= 1)


        # Objective function
            
        self.model.setObjective(quicksum(self.sigma_x_plus) + quicksum(self.sigma_x_minus) + quicksum(self.sigma_y_plus) + quicksum(self.sigma_y_minus), GRB.MINIMIZE)

        # Let's optimise!

        self.model.optimize()

        if self.model.Status == GRB.INFEASIBLE:
            print("Pas de solution...")
        elif self.model.Status == GRB.UNBOUNDED:
            print("Non borné")
        else:
            print("Solution trouvée !")

        return self

    def predict_utility(self, X):
        """Return Decision Function of the MIP for X. - Completed!

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements
        
        Returns
        -------
        np.ndarray:
            (n_samples, n_clusters) array of decision function value for each cluster.
        """
        # Do not forget that this method is called in predict_preference (line 42) and therefor should return well-organized data for it to work.
        results = [[self.score(x, k, constraint=False) for k in range(self.n_clusters)] for x in X] # For each value of X we compute the score according to each cluster

        return np.array(results)


class HeuristicModel(BaseModel):
    """Skeleton of MIP you have to write as the first exercise.
    You have to encapsulate your code within this class that will be called for evaluation.
    """

    
    def __init__(self, n_pieces, n_clusters, n_criteria, P=20000):
        """Initialization of the Heuristic Model.
        """
        self.seed = 123
        self.L = n_pieces
        self.K = n_clusters
        self.n = n_criteria
        self.P = P
        self.kmeans, self.models = self.instantiate()

    
    def instantiate(self):
        """Instantiation of the MIP Variables"""
        # To be completed
        kmeans = KMeans(n_clusters=self.K, random_state=self.seed)
        models = [TwoClustersMIP(self.L, n_clusters=1,n_criteria=self.n,n_pairs=2000) for k in range(self.K)]
        return kmeans, models

    def fit(self, X, Y):
        """Estimation of the parameters - To be completed.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        # To be completed
        pairs=X-Y
        self.kmeans.fit(pairs)
        self.clusters = self.kmeans.cluster_centers_
        self.labels = self.kmeans.labels_
        self.models = [TwoClustersMIP(self.L, n_clusters=1,n_criteria=self.n,n_pairs=2000) for i in range(self.K)]
        for k in range(self.K):
            indexes = np.where(self.labels == k)[0]
            X_k = X[indexes]
            Y_k = Y[indexes]
            self.models[k].fit(X_k, Y_k)
        return self

    def predict_utility(self, X):
        """Return Decision Function of the MIP for X. - To be completed.

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements
        
        Returns
        -------
        np.ndarray:
            (n_samples, n_clusters) array of decision function value for each cluster.
        """
        # To be completed
        # Do not forget that this method is called in predict_preference (line 42) and therefor should return well-organized data for it to work.
        U = [self.models[k].predict_utility(X) for k in range(self.K)]
        U = np.concatenate(U, axis=1)
        return U