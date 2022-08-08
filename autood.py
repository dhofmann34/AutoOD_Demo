from typing import NamedTuple, List, Optional, final
from enum import Enum, auto
import numpy as np
from scipy.io import arff
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn import metrics
import scipy as sp
from logging import Logger
from sklearn.neighbors import NearestNeighbors
from warnings import simplefilter
from dataclasses import dataclass, field
from autood_parameters import get_default_parameters
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
import time
from loguru import logger
from psycopg2.extensions import register_adapter, AsIs
import scipy.linalg
import sys
import csv

database = "y"  # DHDB "y" if we want to connect to database for demo

debug_small_n = 'n'  # "y" if we want smaller dataset for debugging

def addapt_numpy_float64(numpy_float64):
    return AsIs(numpy_float64)
def addapt_numpy_int64(numpy_int64):
    return AsIs(numpy_int64)
register_adapter(np.float64, addapt_numpy_float64)
register_adapter(np.int64, addapt_numpy_int64)

if database == "y":
    from connect import insert_input  # DHDB
    from connect import insert_tsne
    from connect import create_input_table
 
simplefilter(action='ignore', category=FutureWarning)

class OutlierDetectionMethod(Enum):
    LOF = auto()
    KNN = auto()
    IsolationForest = auto()
    Manalanobis = auto()


@dataclass
class AutoODParameters:
    filepath: str
    detection_methods: List[OutlierDetectionMethod]
    k_range: List[int]
    if_range: List[float]
    N_range: List[int]
    index_col_name: Optional[str] = None
    label_col_name: Optional[str] = None


def run_lof(X, k=60):
    clf = LocalOutlierFactor(n_neighbors=k)
    clf.fit(X)
    lof_scores = -clf.negative_outlier_factor_
    return lof_scores


def run_knn(X, k=60):
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(X)
    knn_dists = neigh.kneighbors(X)[0][:, -1]
    return knn_dists


def run_isolation_forest(X, max_features=1.0):
    # training the model
    clf = IsolationForest(random_state=42, max_features=max_features)
    clf.fit(X)
    # predictions
    sklearn_score_anomalies = clf.decision_function(X)
    if_scores = [-1 * s + 0.5 for s in sklearn_score_anomalies]
    return if_scores


def mahalanobis(x):
    """Compute the Mahalanobis Distance between each row of x and the data
    """
    x_minus_mu = x - np.mean(x)
    cov = np.cov(x.T)  # creates a square matrix 
    try:  # if matrix matrix is not invertible use pinv
        inv_covmat = sp.linalg.inv(cov)   # DH old line: inv_covmat = sp.linalg.inv(cov) ||| new line: inv_covmat = sp.linalg.pinv(cov)
    except:
        inv_covmat = sp.linalg.pinv(cov)
    results = []
    x_minus_mu = np.array(x_minus_mu)
    for i in range(np.shape(x)[0]):
        cur_data = x_minus_mu[i, :]
        results.append(np.dot(np.dot(x_minus_mu[i, :], inv_covmat), x_minus_mu[i, :].T))
    return np.array(results)


def run_mahalanobis(X):
    # training the model
    dist = mahalanobis(x=X)
    return dist


def load_dataset(filename, index_col_name = None, label_col_name=None):
    if filename.endswith('arff'):
        with open(filename, 'r') as f:
            data, meta = arff.loadarff(f)
        data = pd.DataFrame(data)
        if debug_small_n == 'y':
            data = data.sample(n=500, random_state = 15)  # DH: added to debug code locally
        data = data.rename(columns={index_col_name :'id', label_col_name : 'label'})  # DH set all data to the same label and id col names
        index_col_name = 'id'
        label_col_name = 'label'
        data[label_col_name] = data[label_col_name].map(lambda x: 1 if x == b'yes' else 0).values if label_col_name else None
        if database == "y":
            import psycopg2
            from config import config
            params = config()  # get DB info from config.py
            conn = psycopg2.connect(**params)
            cur = conn.cursor()
            cur.execute(""" TRUNCATE TABLE input, tsne, detectors, predictions, reliable, temp_if, temp_knn, temp_lof, temp_mahalanobis; """)  # empty tables for demo
            conn.commit()
            cur.close()
            conn.close()

            # SQL does not allow numeric values to be column names
            columnNames = data.columns[pd.to_numeric(data.columns, errors='coerce').to_series().notnull()]
            for name in columnNames:
                data.rename(columns={name : f'_{name}_'}, inplace=True)                     

            create_input_table(data)
            insert_input("input", data)  # DHDB
            insert_tsne("tsne", data, label_col_name, index_col_name)
        data = data.set_index(index_col_name)
        X = data.drop(columns=[label_col_name])
        y = data[label_col_name].values
        # Map dataframe to encode values and put values into a numpy array
        return X, y
    elif filename.endswith('csv'):
        data = pd.read_csv(filename)
        data = data.rename(columns={index_col_name :'id', label_col_name : 'label'})  # DH set all data to the same label and id col names
        index_col_name = 'id'
        label_col_name = 'label'
        if debug_small_n == 'y':
            data = data.sample(n=500, random_state = 15)  # DH: added to debug code locally
        if database == "y":
            
            import psycopg2
            from config import config
            params = config()  # get DB info from config.py
            conn = psycopg2.connect(**params)
            cur = conn.cursor()
            cur.execute(""" TRUNCATE TABLE input, tsne, detectors, predictions, reliable; """)  # empty tables for demo
            conn.commit()
            cur.close()
            conn.close()

            # SQL does not allow numeric values to be column names
            columnNames = data.columns[pd.to_numeric(data.columns, errors='coerce').to_series().notnull()]
            for name in columnNames:
                data.rename(columns={name : f'_{name}_'}, inplace=True)  
        
            create_input_table(data)
            insert_input("input", data)  # DHDB
            insert_tsne("tsne", data, label_col_name, index_col_name)
        data = data.set_index(index_col_name)
        X = data.drop(columns=[label_col_name]) if label_col_name else data
        y = data[label_col_name].values if label_col_name else None
        return X, y
    else: 
        return None


def get_predictions(scores, num_outliers=400):
    threshold = np.sort(scores)[::-1][num_outliers]
    predictions = np.array(scores > threshold)
    predictions = np.array([int(i) for i in predictions])
    return predictions, scores

def get_f1_scores(predictions, y):
    return metrics.f1_score(y, predictions)

@dataclass
class AutoODResults:
    best_unsupervised_f1_score: float = 0.0
    best_unsupervised_methods: List[str] = field(default_factory=list)
    mv_f1_score: float = 0.0
    autood_f1_score: float = 0.0
    results_file_name: str = ""
    error_message: str = ""


@dataclass(init=False)
class AutoOD:
    params: AutoODParameters
    logger: Logger

    def __init__(self, params, logger):
        self.params = params
        self.logger = logger
        self.best_unsupervised_f1_score = 0
        self.best_unsupervised_methods = []
        self.mv_f1_score = 0
        self.autood_f1_score = 0

    def _load_dataset(self):
        file_name = self.params.filepath
        data = load_dataset(file_name, self.params.index_col_name, self.params.label_col_name)
        return data

    def _run_unsupervised_outlier_detection_methods(self, X, y):
        all_results = []
        all_scores = []
        methods_to_best_f1 = {}
        f1s = []
        num_detectors = 0
        instance_index_ranges = []
        detector_index_ranges = []

        if OutlierDetectionMethod.LOF in self.params.detection_methods:
            f1_list_start_index = len(f1s)
            N_size = len(self.params.N_range)
            N_range = [int(np.shape(X)[0] * percent) for percent in self.params.N_range]
            krange_list = self.params.k_range * N_size
            knn_N_range = np.sort(N_range * len(self.params.k_range))

            self.logger.info(f'Start running LOF with k={self.params.k_range}')
            temp_lof_results = dict()
            for k in self.params.k_range:
                lof_scores = run_lof(X, k=k)
                temp_lof_results[k] = lof_scores
            if database == "y":  #DHDB
                col_names = ["id", "detector", "k", "n", "prediction", "score"]
                lof_df = pd.DataFrame(columns = col_names)
            for i in range(len(krange_list)):
                lof_predictions, lof_scores = get_predictions(temp_lof_results[krange_list[i]], num_outliers=knn_N_range[i])  # DH: get predictions takes in num outliers
                all_results.append(lof_predictions)
                all_scores.append(lof_scores)
                if database == "y":  #DHDB
                    temp_lof_data = pd.DataFrame()
                    temp_lof_data["id"] = X.index  # WITH ARFF FILE INDEX STARTS AT 0 WITH CSV IT STARTS WITH 1
                    temp_lof_data["detector"] = "LOF"
                    temp_lof_data["k"] = krange_list[i]
                    temp_lof_data["n"] = knn_N_range[i]
                    temp_lof_data["prediction"] = lof_predictions
                    temp_lof_data["score"] = lof_scores
                    lof_df = lof_df.append(temp_lof_data)
                if y is not None:
                    f1 = get_f1_scores(predictions=lof_predictions,y = y)  # f1 score for each of the detectors? 
                    f1s.append(f1)

            if database == "y":  # DHDB
                insert_input("detectors", lof_df)
            f1_list_end_index = len(f1s)
            instance_index_ranges.append([f1_list_start_index, f1_list_end_index])
            detector_index_ranges.append([num_detectors, num_detectors+len(self.params.k_range)])
            num_detectors = num_detectors+len(self.params.k_range)
            if y is not None:
                best_lof_f1 = 0
                for i in np.sort(self.params.k_range):
                    temp_f1 = max(np.array(f1s[f1_list_start_index: f1_list_end_index])[np.where(np.array(krange_list) == i)[0]])  # DH self.params.k_range is our paramiters. aka len(self.params.k_range) = num of detectors
                    best_lof_f1 = max(best_lof_f1, temp_f1)  # DH here we can set up a dict of detector id and the f1 score?
                methods_to_best_f1["LOF"] = best_lof_f1
                self.logger.info('Best LOF F-1 = {}'.format(best_lof_f1))

        if OutlierDetectionMethod.KNN in self.params.detection_methods:
            f1_list_start_index = len(f1s)
            N_size = len(self.params.N_range)
            N_range = [int(np.shape(X)[0] * percent) for percent in self.params.N_range]
            krange_list = self.params.k_range * N_size
            knn_N_range = np.sort(N_range * len(self.params.k_range))

            self.logger.info(f'Start running KNN with k={self.params.k_range}')
            temp_knn_results = dict()
            for k in self.params.k_range:
                knn_scores = run_knn(X, k=k)
                temp_knn_results[k] = knn_scores
            if database == "y":  #DHDB
                col_names = ["id", "detector", "k", "n", "prediction", "score"]
                knn_df = pd.DataFrame(columns = col_names)
            for i in range(len(krange_list)):
                knn_predictions, knn_scores = get_predictions(temp_knn_results[krange_list[i]],num_outliers=knn_N_range[i])
                all_results.append(knn_predictions)
                all_scores.append(knn_scores)
                if database == "y":  #DHDB
                    temp_knn_data = pd.DataFrame()
                    temp_knn_data["id"] = X.index
                    temp_knn_data["detector"] = "KNN"
                    temp_knn_data["k"] = krange_list[i]
                    temp_knn_data["n"] = knn_N_range[i]
                    temp_knn_data["prediction"] = knn_predictions
                    temp_knn_data["score"] = knn_scores
                    knn_df = knn_df.append(temp_knn_data)
                if y is not None:
                    f1 = get_f1_scores(predictions=knn_predictions,y = y)
                    f1s.append(f1)

            if database == "y":  # DHDB
                insert_input("detectors", knn_df)
            f1_list_end_index = len(f1s)
            instance_index_ranges.append([f1_list_start_index, f1_list_end_index])
            detector_index_ranges.append([num_detectors, num_detectors + len(self.params.k_range)])
            num_detectors = num_detectors + len(self.params.k_range)
            if y is not None:
                best_knn_f1 = 0
                for i in np.sort(self.params.k_range):
                    temp_f1 = max(np.array(f1s[f1_list_start_index:f1_list_end_index])[np.where(np.array(krange_list) == i)[0]])
                    best_knn_f1 = max(best_knn_f1, temp_f1)
                methods_to_best_f1["KNN"] = best_knn_f1
                self.logger.info('Best KNN F-1 = {}'.format(best_knn_f1))

        if OutlierDetectionMethod.IsolationForest in self.params.detection_methods:
            f1_list_start_index = len(f1s)
            N_size = len(self.params.N_range)
            N_range = [int(np.shape(X)[0] * percent) for percent in self.params.N_range]
            knn_N_range = np.sort(N_range * len(self.params.k_range))

            self.logger.info(f'Start running Isolation Forest with max feature = {self.params.if_range}')
            temp_if_results = dict()
            for k in self.params.if_range:
                if_scores = run_isolation_forest(X, max_features=k)
                temp_if_results[k] = if_scores
            if_range_list = self.params.if_range * N_size
            if_N_range = np.sort(self.params.N_range * len(self.params.if_range))
            if database == "y":  #DHDB
                col_names = ["id", "detector", "k", "n", "prediction", "score"]
                if_df = pd.DataFrame(columns = col_names)
            for i in range(len(if_range_list)):
                if_predictions, if_scores = get_predictions(temp_if_results[if_range_list[i]], num_outliers=knn_N_range[i])
                all_results.append(if_predictions)
                all_scores.append(if_scores)
                if database == "y":  #DHDB
                    temp_if_data = pd.DataFrame()
                    temp_if_data["id"] = X.index
                    temp_if_data["detector"] = "IF"
                    temp_if_data["k"] = if_range_list[i]
                    temp_if_data["n"] = knn_N_range[i]
                    temp_if_data["prediction"] = if_predictions
                    temp_if_data["score"] = if_scores
                    if_df = if_df.append(temp_if_data)
                if y is not None:
                    f1 = get_f1_scores(predictions=if_predictions,y = y)
                    f1s.append(f1)

            if database == "y":  # DHDB
                insert_input("detectors", if_df)
            f1_list_end_index = len(f1s)
            instance_index_ranges.append([f1_list_start_index, f1_list_end_index])
            detector_index_ranges.append([num_detectors, num_detectors + len(self.params.if_range)])
            num_detectors = num_detectors + len(self.params.if_range)
            if y is not None:
                best_if_f1 = 0
                for i in np.sort(self.params.if_range):
                    temp_f1 = max(np.array(f1s[f1_list_start_index:f1_list_end_index])[np.where(np.array(if_range_list) == i)[0]])
                    best_if_f1 = max(best_if_f1, temp_f1)
                methods_to_best_f1["Isolation_Forest"] = best_if_f1
                self.logger.info('Best IF F-1 = {}'.format(best_if_f1))

        if OutlierDetectionMethod.Manalanobis in self.params.detection_methods:
            self.logger.info(f'Start running Mahalanobis..')
            f1_list_start_index = len(f1s)
            N_range = [int(np.shape(X)[0] * percent) for percent in self.params.N_range]
            mahalanobis_scores = run_mahalanobis(X)
            best_mahala_f1 = 0
            if database == "y":  #DHDB
                col_names = ["id", "detector", "n", "prediction", "score"]  # no k
                mahala_df = pd.DataFrame(columns = col_names)
            for i in range(len(N_range)):
                mahalanobis_predictions, mahalanobis_scores = get_predictions(mahalanobis_scores,num_outliers=N_range[i])
                all_results.append(mahalanobis_predictions)
                all_scores.append(mahalanobis_scores)
                if database == "y":  #DHDB
                    temp_mahala_data = pd.DataFrame()
                    temp_mahala_data["id"] = X.index
                    temp_mahala_data["detector"] = "mahalanobis"
                    #temp_mahala_data["k"] = if_range_list[i]
                    temp_mahala_data["n"] = N_range[i]
                    temp_mahala_data["prediction"] = mahalanobis_predictions
                    temp_mahala_data["score"] = mahalanobis_scores
                    mahala_df = mahala_df.append(temp_mahala_data)
                if y is not None:
                    f1 = get_f1_scores(mahalanobis_predictions, y)
                    best_mahala_f1 = max(best_mahala_f1, f1)
                    f1s.append(f1)

            if database == "y":  # DHDB
                insert_input("detectors", mahala_df)
            f1_list_end_index = len(f1s)
            if y is not None:
                methods_to_best_f1["malalanobis"] = best_mahala_f1
                self.logger.info('Best Mahala F-1 = {}'.format(best_mahala_f1))
            instance_index_ranges.append([f1_list_start_index, f1_list_end_index])
            detector_index_ranges.append([num_detectors, num_detectors + 1])
            num_detectors = num_detectors + 1
        if y is not None and methods_to_best_f1:
            self.best_unsupervised_f1_score = max(methods_to_best_f1.values())
            self.best_unsupervised_methods = [method for method, f1 in methods_to_best_f1.items() if
                                              f1 >= self.best_unsupervised_f1_score]
            self.logger.info(f"Best Unsupervised F-1 Score = {self.best_unsupervised_f1_score}")
            self.logger.info(f"Best Unsupervised Outlier Detection Method = {self.best_unsupervised_methods}")
        L = np.stack(all_results).T
        scores = np.stack(all_scores).T
        self.logger.info(f"Instance Index Ranges: {instance_index_ranges}")
        self.logger.info(f"Detector Index Ranges: {detector_index_ranges}")
        return L, scores, instance_index_ranges, detector_index_ranges

    def run_mv(self, L, y):
        mid = np.shape(L)[1] / 2
        predictions = np.full((len(y)), 0)
        predictions[np.sum(L, axis=1) > mid] = 1
        self.mv_f1_score = metrics.f1_score(y, predictions)
        self.logger.info(f'F1 for Majority Vote:{self.mv_f1_score}')

    def autood_training(self, L, scores, X, y, instance_index_ranges, detector_index_ranges):  # add param here that is none on defult: takes in human labels
        L_prev = L
        scores_prev = scores
        prediction_result_list = []
        classifier_result_list = []
        prediction_list = []
        cur_f1_scores = []
        prediction_high_conf_outliers = np.array([])
        prediction_high_conf_inliers = np.array([])
        prediction_classifier_disagree = np.array([])
        index_range = np.array(instance_index_ranges)
        coef_index_range = np.array(detector_index_ranges)
        scores_for_training_indexes = []
        for i in range(len(index_range)):
            start = index_range[i][0]
            temp_range = coef_index_range[i][1] - coef_index_range[i][0]
            scores_for_training_indexes = scores_for_training_indexes + list(range(start, start + temp_range))
        scores_for_training = scores[:, np.array(scores_for_training_indexes)]  # outlier scores from unsup detectors

        # stable version
        high_confidence_threshold = 0.99
        low_confidence_threshold = 0.01
        max_iter = 200 # DH was 200 
        remain_params_tracking = np.array(range(0, np.max(coef_index_range)))
        training_data_F1 = []
        two_prediction_corr = []

        N_size = len(self.params.N_range)

        last_training_data_indexes = []
        counter = 0
        self.logger.info("##################################################################")
        self.logger.info("Start First-round AutoOD training...")

        if database == "y":  #DHDB
            col_names = ["id", "iteration", "reliable"]
            reliable_df = pd.DataFrame(columns = col_names)
        for i_range in range(0, 50):
            num_methods = np.shape(L)[1]  # L is matrix: data points x unsup methods, L[0] gives us the number of detectors
            agree_outlier_indexes = np.sum(L, axis=1) == np.shape(L)[1]  # index of all datapoints where they all agree on outlier
            agree_inlier_indexes = np.sum(L, axis=1) == 0
            disagree_indexes = np.where(np.logical_or(np.sum(L, axis=1) == 0, np.sum(L, axis=1) == num_methods) == 0)[0]
            all_inlier_indexes = np.setdiff1d(np.where(agree_inlier_indexes)[0], prediction_high_conf_outliers)
            if len(prediction_high_conf_inliers) > 0:
                all_inlier_indexes = np.intersect1d(
                    np.setdiff1d(np.where(agree_inlier_indexes)[0], prediction_high_conf_outliers),
                    prediction_high_conf_inliers)
            all_outlier_indexes = np.union1d(np.where(agree_outlier_indexes)[0], prediction_high_conf_outliers)  # set with all outliers that agree
            all_inlier_indexes = np.setdiff1d(all_inlier_indexes, prediction_classifier_disagree)  #  set with all inliers that agree 

            self_agree_index_list = []
            if ((len(all_outlier_indexes) == 0) or (len(all_inlier_indexes) / len(all_outlier_indexes) > 1000)):  # if no outliers in reliable set: loosen requirments
                for i in range(0, len(index_range)):
                    if (index_range[i, 1] - index_range[i, 0] <= 6):
                        continue
                    temp_index = disagree_indexes[np.where(
                        np.sum(L[disagree_indexes][:, index_range[i, 0]: index_range[i, 1]], axis=1) == (
                                index_range[i, 1] - index_range[i, 0]))[0]]
                    self_agree_index_list = np.union1d(self_agree_index_list, temp_index)
                self_agree_index_list = [int(i) for i in self_agree_index_list]

            all_outlier_indexes = np.union1d(all_outlier_indexes, self_agree_index_list)  # add empty set if outliers are in reliable set, else add self agree
            if(len(np.setdiff1d(all_outlier_indexes, prediction_classifier_disagree)) != 0):  # DH: remove reliable only if set will not be 0
                all_outlier_indexes = np.setdiff1d(all_outlier_indexes, prediction_classifier_disagree)  

            #### DH
            self_agree_index_list_inlier = []
            if ((len(all_inlier_indexes) == 0)):  # DH: requirements are relaxed 
                for i in range(0, len(index_range)):
                    if (index_range[i, 1] - index_range[i, 0] <= 6):
                        continue
                    temp_index = disagree_indexes[np.where(
                        np.sum(L[disagree_indexes][:, index_range[i, 0]: index_range[i, 1]], axis=1) == (
                                index_range[i, 1] - index_range[i, 0]))[0]]
                    self_agree_index_list_inlier = np.union1d(self_agree_index_list_inlier, temp_index)
                self_agree_index_list_inlier = [int(i) for i in self_agree_index_list_inlier]
            all_inlier_indexes = np.union1d(all_inlier_indexes, self_agree_index_list_inlier)  
            if(len(np.setdiff1d(all_inlier_indexes, prediction_classifier_disagree)) != 0): 
                all_inlier_indexes = np.setdiff1d(all_inlier_indexes, prediction_classifier_disagree)  

            if(len(all_inlier_indexes) == 0):  # still no reliable inliers: add most reliable inlier
                min_score = sys.maxsize
                index_method = 0
                counter = -1
                for scores_method in scores:
                    counter = counter + 1
                    if np.average(scores_method) < min_score:
                        min_score = np.average(scores_method)
                        index_method = counter
                all_inlier_indexes = np.union1d(all_inlier_indexes, index_method) 

            #### DH

            data_indexes = np.concatenate((all_inlier_indexes, all_outlier_indexes), axis=0)  # all reliable indices
            data_indexes = np.array([int(i) for i in data_indexes])
            labels = np.concatenate((np.zeros(len(all_inlier_indexes)), np.ones(len(all_outlier_indexes))), axis=0)
            transformer = RobustScaler().fit(scores_for_training)
            scores_transformed = transformer.transform(scores_for_training)
            training_data = scores_transformed[data_indexes]
            if y is not None:
                training_data_F1.append(metrics.f1_score(y[data_indexes], labels))

            clf = LogisticRegression(random_state=0, penalty='l2', max_iter=max_iter).fit(training_data, labels)
            clf_predictions = clf.predict(scores_transformed)
            clf_predict_proba = clf.predict_proba(scores_transformed)[:, 1]

            transformer = RobustScaler().fit(X)
            X_transformed = transformer.transform(X)
            X_training_data = X_transformed[data_indexes]

            clf_X = SVC(gamma='auto', probability=True, random_state=0)
            clf_X.fit(X_training_data, labels)
            clf_predict_proba_X = clf_X.predict_proba(X_transformed)[:, 1]
            if y is not None:
                self.logger.info(
                    f'Iteration = {i_range}, F-1 score = {metrics.f1_score(y, np.array([int(i) for i in clf_predict_proba_X > 0.5]))}')
                #f1_at_iter(f"1_{i_range}", len(X_training_data), len(L[0]), len(all_inlier_indexes), len(all_outlier_indexes), metrics.f1_score(y, np.array([int(i) for i in clf_predict_proba_X > 0.5])))  # DH
            else:
                self.logger.info(f"Iteration = {i_range}")
            prediction_result_list.append(clf_predict_proba)
            classifier_result_list.append(clf_predict_proba_X)

            prediction_list.append(np.array([int(i) for i in clf_predictions]))

            prediction_high_conf_outliers = np.intersect1d(  # DH select high conf outliers
                np.where(prediction_result_list[-1] > high_confidence_threshold)[0],  # results from logistic regression
                np.where(classifier_result_list[-1] > high_confidence_threshold)[0])  # results from SVM

            prediction_high_conf_inliers = np.intersect1d(  # DH select high conf inliers
                np.where(prediction_result_list[-1] < low_confidence_threshold)[0],
                np.where(classifier_result_list[-1] < low_confidence_threshold)[0])

            temp_prediction = np.array([int(i) for i in prediction_result_list[-1] > 0.5])  # logistic
            temp_classifier = np.array([int(i) for i in classifier_result_list[-1] > 0.5])  # SVM
            prediction_classifier_disagree = np.where(temp_prediction != temp_classifier)[0]  # gets index of where do SVM and log. disagree

            two_prediction_corr.append(np.corrcoef(clf_predict_proba, clf_predict_proba_X)[0, 1])  # Return Pearson product-moment correlation coefficients

            if np.max(coef_index_range) >= 2:
                if (len(prediction_high_conf_outliers) > 0 and len(prediction_high_conf_inliers) > 0):
                    # reliabel object set, SVM and log. agree
                    new_data_indexes = np.concatenate((prediction_high_conf_outliers, prediction_high_conf_inliers),
                                                      axis=0)
                    new_data_indexes = np.array([int(i) for i in new_data_indexes])
                    # update new labels based on SVM and log.
                    new_labels = np.concatenate(
                        (np.ones(len(prediction_high_conf_outliers)), np.zeros(len(prediction_high_conf_inliers))),
                        axis=0)
                    clf_prune_2 = LogisticRegression(random_state=0, penalty='l2', max_iter=max_iter).fit(
                        scores_transformed[new_data_indexes], new_labels)  # train log with updated labels: this is to prune poor detectors
                    combined_coef = clf_prune_2.coef_[0]
                else:
                    combined_coef = clf.coef_[0]

                if (np.max(coef_index_range) >= 2):
                    if (len(set(combined_coef)) > 1):
                        cur_clf_coef = combined_coef
                        cutoff = max(max(0, np.mean(combined_coef) - np.std(combined_coef)), min(combined_coef))  # cutoff to purne poor detectors

                        remain_indexes_after_cond = (
                                cur_clf_coef > cutoff)  # np.logical_and(cur_clf_coef > cutoff, abs(cur_clf_coef) > 0.01) 
                        remain_params_tracking = remain_params_tracking[remain_indexes_after_cond]  # prune poor detectors

                        remain_indexes_after_cond_expanded = []  # update index for reliable set: Can we use this to keep track of when detectors are removed?
                        for i in range(0, len(coef_index_range)):  
                            s_e_range = coef_index_range[i, 1] - coef_index_range[i, 0]
                            s1, e1 = coef_index_range[i, 0], coef_index_range[i, 1]
                            s2, e2 = index_range[i, 0], index_range[i, 1]
                            saved_indexes = np.where(cur_clf_coef[s1:e1] > cutoff)[0]
                            for j in range(N_size):
                                remain_indexes_after_cond_expanded.extend(np.array(saved_indexes) + j * s_e_range + s2)

                        new_coef_index_range_seq = []
                        for i in range(0, len(coef_index_range)): 
                            s, e = coef_index_range[i, 0], coef_index_range[i, 1]
                            new_coef_index_range_seq.append(sum((remain_indexes_after_cond)[s:e]))

                        coef_index_range = []
                        index_range = []
                        cur_sum = 0
                        for i in range(0, len(new_coef_index_range_seq)):
                            coef_index_range.append([cur_sum, cur_sum + new_coef_index_range_seq[i]])
                            index_range.append([cur_sum * 6, 6 * (cur_sum + new_coef_index_range_seq[i])])
                            cur_sum += new_coef_index_range_seq[i]

                        coef_index_range = np.array(coef_index_range)
                        index_range = np.array(index_range)

                        L = L[:, remain_indexes_after_cond_expanded]  # DH: here we are updating detectors
                        scores_for_training = scores_for_training[:, remain_indexes_after_cond]
            if ((len(last_training_data_indexes) == len(data_indexes)) and
                    (sum(last_training_data_indexes == data_indexes) == len(data_indexes)) and
                    (np.max(coef_index_range) < 2)):
                counter = counter + 1  # early stop statment: no changes for 3 iterations -> stop
            else:
                counter = 0
            if (counter > 3):
                break
            last_training_data_indexes = data_indexes
            if database == "y":  #DHDB
                full_data = np.zeros(shape=(len(X),))  # array length of input data
                np.put(full_data, prediction_high_conf_inliers, 1)  # set index of high conf values to 1
                np.put(full_data, prediction_high_conf_outliers, 1)
                temp_reliable_df = pd.DataFrame()
                temp_reliable_df["id"] = X.index
                #temp_reliable_df = temp_reliable_df.sort_values(by=["id"])
                temp_reliable_df["iteration"] = i_range
                temp_reliable_df["reliable"] = full_data
                reliable_df = reliable_df.append(temp_reliable_df)
        if database == "y":  # DHDB
            #reliable_df = reliable_df.convert_dtypes()
            insert_input("reliable", reliable_df)

        # Prepare second round AutoOD
        index_range = np.array(instance_index_ranges)
        coef_index_range = np.array(detector_index_ranges)

        scores_for_training_indexes = []
        for i in range(len(index_range)):
            start = index_range[i][0]
            temp_range = coef_index_range[i][1] - coef_index_range[i][0]
            scores_for_training_indexes = scores_for_training_indexes + list(range(start, start + temp_range))

        scores_for_training = scores[:, np.array(scores_for_training_indexes)]

        transformer = RobustScaler().fit(scores_for_training)
        scores_transformed = transformer.transform(scores_for_training)
        training_data = scores_transformed[data_indexes]
        if y is not None:
            training_data_F1.append(metrics.f1_score(y[data_indexes], labels))

        clf = LogisticRegression(random_state=0, penalty='l2', max_iter=max_iter).fit(training_data, labels)
        clf_predictions = clf.predict(scores_transformed)
        clf_predict_proba = clf.predict_proba(scores_transformed)[:, 1]

        transformer = RobustScaler().fit(X)
        X_transformed = transformer.transform(X)
        X_training_data = X_transformed[data_indexes]

        clf_X = SVC(gamma='auto', probability=True, random_state=0)
        clf_X.fit(X_training_data, labels)
        clf_predict_proba_X = clf_X.predict_proba(X_transformed)[:, 1]
        if y is not None:
            cur_f1_scores.append(metrics.f1_score(y, np.array([int(i) for i in clf_predict_proba_X > 0.5])))

        prediction_result_list.append(clf_predict_proba)
        classifier_result_list.append(clf_predict_proba_X)

        prediction_list.append(np.array([int(i) for i in clf_predictions]))

        prediction_high_conf_outliers = np.intersect1d(
            np.where(prediction_result_list[-1] > high_confidence_threshold)[0],
            np.where(classifier_result_list[-1] > high_confidence_threshold)[0])
        prediction_high_conf_inliers = np.intersect1d(
            np.where(prediction_result_list[-1] < low_confidence_threshold)[0],
            np.where(classifier_result_list[-1] < low_confidence_threshold)[0])

        temp_prediction = np.array([int(i) for i in prediction_result_list[-1] > 0.5])
        temp_classifier = np.array([int(i) for i in classifier_result_list[-1] > 0.5])
        prediction_classifier_disagree = np.where(temp_prediction != temp_classifier)[0]

        L = L_prev
        remain_params_tracking = np.array(range(0, np.max(coef_index_range)))

        if np.max(coef_index_range) >= 2:
            if (len(prediction_high_conf_outliers) > 0 and len(prediction_high_conf_inliers) > 0):  # losen requirements 
                new_data_indexes = np.concatenate((prediction_high_conf_outliers, prediction_high_conf_inliers), axis=0)
                new_data_indexes = np.array([int(i) for i in new_data_indexes])
                new_labels = np.concatenate(
                    (np.ones(len(prediction_high_conf_outliers)), np.zeros(len(prediction_high_conf_inliers))), axis=0)
                clf_prune_2 = LogisticRegression(random_state=0, penalty='l2', max_iter=max_iter).fit(
                    scores_transformed[new_data_indexes], new_labels)
                combined_coef = clf_prune_2.coef_[0]
            else:
                combined_coef = clf.coef_[0]

            if (np.max(coef_index_range) > 2 or
                    ((np.max(combined_coef) / np.min(combined_coef) >= 1.1) and np.max(coef_index_range) >= 2)):
                if (len(set(combined_coef)) > 1):
                    cur_clf_coef = combined_coef
                    cutoff = max(max(0, np.mean(combined_coef) - np.std(combined_coef)), min(combined_coef))

                    remain_indexes_after_cond = (
                            cur_clf_coef > cutoff)  # np.logical_and(cur_clf_coef > cutoff, abs(cur_clf_coef) > 0.01) 
                    remain_params_tracking = remain_params_tracking[remain_indexes_after_cond]
                    remain_indexes_after_cond_expanded = []
                    for i in range(0, len(coef_index_range)):  #
                        s_e_range = coef_index_range[i, 1] - coef_index_range[i, 0]
                        s1, e1 = coef_index_range[i, 0], coef_index_range[i, 1]
                        s2, e2 = index_range[i, 0], index_range[i, 1]
                        saved_indexes = np.where(cur_clf_coef[s1:e1] > cutoff)[0]
                        for j in range(N_size):
                            remain_indexes_after_cond_expanded.extend(np.array(saved_indexes) + j * s_e_range + s2)

                    new_coef_index_range_seq = []
                    for i in range(0, len(coef_index_range)):  #
                        s, e = coef_index_range[i, 0], coef_index_range[i, 1]
                        new_coef_index_range_seq.append(sum((remain_indexes_after_cond)[s:e]))

                    coef_index_range = []
                    index_range = []
                    cur_sum = 0
                    for i in range(0, len(new_coef_index_range_seq)):
                        coef_index_range.append([cur_sum, cur_sum + new_coef_index_range_seq[i]])
                        index_range.append([cur_sum * 6, 6 * (cur_sum + new_coef_index_range_seq[i])])
                        cur_sum += new_coef_index_range_seq[i]

                    coef_index_range = np.array(coef_index_range)
                    index_range = np.array(index_range)

                    L = L[:, remain_indexes_after_cond_expanded]
                    scores_for_training = scores_for_training[:, remain_indexes_after_cond]
        self.logger.info("##################################################################")
        self.logger.info("Start Second-round AutoOD training...")
        last_training_data_indexes = []
        counter = 0

        for i_range in range(0, 50):
            num_methods = np.shape(L)[1]

            agree_outlier_indexes = np.sum(L, axis=1) == np.shape(L)[1]
            agree_inlier_indexes = np.sum(L, axis=1) == 0

            disagree_indexes = np.where(np.logical_or(np.sum(L, axis=1) == 0, np.sum(L, axis=1) == num_methods) == 0)[0]

            all_inlier_indexes = np.setdiff1d(np.where(agree_inlier_indexes)[0], prediction_high_conf_outliers)
            if len(prediction_high_conf_inliers) > 0:
                all_inlier_indexes = np.intersect1d(
                    np.setdiff1d(np.where(agree_inlier_indexes)[0], prediction_high_conf_outliers),
                    prediction_high_conf_inliers)

            all_outlier_indexes = np.union1d(np.where(agree_outlier_indexes)[0], prediction_high_conf_outliers)
            all_inlier_indexes = np.setdiff1d(all_inlier_indexes, prediction_classifier_disagree)

            self_agree_index_list = []
            if ((len(all_outlier_indexes) == 0) or (len(all_inlier_indexes) / len(all_outlier_indexes) > 1000)):  # losen requirements 
                for i in range(0, len(index_range)):
                    if (index_range[i, 1] - index_range[i, 0] <= 6):
                        continue
                    temp_index = disagree_indexes[np.where(
                        np.sum(L[disagree_indexes][:, index_range[i, 0]: index_range[i, 1]], axis=1) == (
                                index_range[i, 1] - index_range[i, 0]))[0]]
                    self_agree_index_list = np.union1d(self_agree_index_list, temp_index)
                self_agree_index_list = [int(i) for i in self_agree_index_list]

            all_outlier_indexes = np.union1d(all_outlier_indexes, self_agree_index_list)
            if(len(np.setdiff1d(all_outlier_indexes, prediction_classifier_disagree)) != 0):  # remove reliable only if set will not be 0
                all_outlier_indexes = np.setdiff1d(all_outlier_indexes, prediction_classifier_disagree)

             #### DH
            self_agree_index_list_inlier = []
            if ((len(all_inlier_indexes) == 0)):  # here is where requirements are relaxed 
                for i in range(0, len(index_range)):
                    if (index_range[i, 1] - index_range[i, 0] <= 6):
                        continue
                    temp_index = disagree_indexes[np.where(
                        np.sum(L[disagree_indexes][:, index_range[i, 0]: index_range[i, 1]], axis=1) == (
                                index_range[i, 1] - index_range[i, 0]))[0]]
                    self_agree_index_list_inlier = np.union1d(self_agree_index_list_inlier, temp_index)
                self_agree_index_list_inlier = [int(i) for i in self_agree_index_list_inlier]
            all_inlier_indexes = np.union1d(all_inlier_indexes, self_agree_index_list_inlier)  
            if(len(np.setdiff1d(all_inlier_indexes, prediction_classifier_disagree)) != 0): 
                all_inlier_indexes = np.setdiff1d(all_inlier_indexes, prediction_classifier_disagree)  

            if(len(all_inlier_indexes) == 0):  # still no reliable labels: add most reliable inlier
                    min_score = sys.maxsize
                    index_method = 0
                    counter = -1
                    for scores_method in scores:
                        counter = counter + 1
                        if np.average(scores_method) < min_score:
                            min_score = np.average(scores_method)
                            index_method = counter
                    all_inlier_indexes = np.union1d(all_inlier_indexes, index_method) 

            #### DH

            data_indexes = np.concatenate((all_inlier_indexes, all_outlier_indexes), axis=0)
            data_indexes = np.array([int(i) for i in data_indexes])
            labels = np.concatenate((np.zeros(len(all_inlier_indexes)), np.ones(len(all_outlier_indexes))), axis=0)
            transformer = RobustScaler().fit(scores_for_training)
            scores_transformed = transformer.transform(scores_for_training)
            training_data = scores_transformed[data_indexes]
            if y is not None:
                training_data_F1.append(metrics.f1_score(y[data_indexes], labels))

            clf = LogisticRegression(random_state=0, penalty='l2', max_iter=max_iter).fit(training_data, labels)
            clf_predictions = clf.predict(scores_transformed)
            clf_predict_proba = clf.predict_proba(scores_transformed)[:, 1]

            transformer = RobustScaler().fit(X)
            X_transformed = transformer.transform(X)
            X_training_data = X_transformed[data_indexes]

            clf_X = SVC(gamma='auto', probability=True, random_state=0)
            clf_X.fit(X_training_data, labels)

            clf_predict_proba_X = clf_X.predict_proba(X_transformed)[:, 1]
            if y is not None:
                self.logger.info(
                    f'Iteration = {i_range}, F-1 score = {metrics.f1_score(y, np.array([int(i) for i in clf_predict_proba_X > 0.5]))}')
                cur_f1_scores.append(metrics.f1_score(y, np.array([int(i) for i in clf_predict_proba_X > 0.5])))
                #f1_at_iter(f"2_{i_range}", len(X_training_data), len(L[0]), len(all_inlier_indexes), len(all_outlier_indexes), metrics.f1_score(y, np.array([int(i) for i in clf_predict_proba_X > 0.5])))  # DH
            else:
                self.logger.info(f"Iteration = {i_range}")

            prediction_result_list.append(clf_predict_proba)
            classifier_result_list.append(clf_predict_proba_X)

            prediction_list.append(np.array([int(i) for i in clf_predictions]))

            prediction_high_conf_outliers = np.intersect1d(
                np.where(prediction_result_list[-1] > high_confidence_threshold)[0],
                np.where(classifier_result_list[-1] > high_confidence_threshold)[0])

            prediction_high_conf_inliers = np.intersect1d(
                np.where(prediction_result_list[-1] < low_confidence_threshold)[0],
                np.where(classifier_result_list[-1] < low_confidence_threshold)[0])

            temp_prediction = np.array([int(i) for i in prediction_result_list[-1] > 0.5])
            temp_classifier = np.array([int(i) for i in classifier_result_list[-1] > 0.5])
            prediction_classifier_disagree = np.where(temp_prediction != temp_classifier)[0]

            if np.max(coef_index_range) >= 2:
                if (len(prediction_high_conf_outliers) > 0 and len(prediction_high_conf_inliers) > 0):
                    new_data_indexes = np.concatenate((prediction_high_conf_outliers, prediction_high_conf_inliers),
                                                      axis=0)
                    new_data_indexes = np.array([int(i) for i in new_data_indexes])
                    new_labels = np.concatenate(
                        (np.ones(len(prediction_high_conf_outliers)), np.zeros(len(prediction_high_conf_inliers))),
                        axis=0)
                    clf_prune_2 = LogisticRegression(random_state=0, penalty='l2', max_iter=max_iter).fit(
                        scores_transformed[new_data_indexes], new_labels)
                    combined_coef = clf_prune_2.coef_[0]
                else:
                    combined_coef = clf.coef_[0]

                if (np.max(coef_index_range) > 2 or
                        ((np.max(combined_coef) / np.min(combined_coef) >= 1.1) and np.max(coef_index_range) >= 2)):
                    if (len(set(combined_coef)) > 1):
                        cur_clf_coef = combined_coef
                        cutoff = max(max(0, np.mean(combined_coef) - np.std(combined_coef)), min(combined_coef))

                        remain_indexes_after_cond = (
                                cur_clf_coef > cutoff)  # np.logical_and(cur_clf_coef > cutoff, abs(cur_clf_coef) > 0.01) # #
                        remain_params_tracking = remain_params_tracking[remain_indexes_after_cond]
                        remain_indexes_after_cond_expanded = []
                        for i in range(0, len(coef_index_range)):  #
                            s_e_range = coef_index_range[i, 1] - coef_index_range[i, 0]
                            s1, e1 = coef_index_range[i, 0], coef_index_range[i, 1]
                            s2, e2 = index_range[i, 0], index_range[i, 1]
                            saved_indexes = np.where(cur_clf_coef[s1:e1] > cutoff)[0]
                            for j in range(N_size):
                                remain_indexes_after_cond_expanded.extend(np.array(saved_indexes) + j * s_e_range + s2)

                        new_coef_index_range_seq = []
                        for i in range(0, len(coef_index_range)):  #
                            s, e = coef_index_range[i, 0], coef_index_range[i, 1]
                            new_coef_index_range_seq.append(sum((remain_indexes_after_cond)[s:e]))

                        coef_index_range = []
                        index_range = []
                        cur_sum = 0
                        for i in range(0, len(new_coef_index_range_seq)):
                            coef_index_range.append([cur_sum, cur_sum + new_coef_index_range_seq[i]])
                            index_range.append([cur_sum * 6, 6 * (cur_sum + new_coef_index_range_seq[i])])
                            cur_sum += new_coef_index_range_seq[i]

                        coef_index_range = np.array(coef_index_range)
                        index_range = np.array(index_range)

                        L = L[:, remain_indexes_after_cond_expanded]
                        scores_for_training = scores_for_training[:, remain_indexes_after_cond]
            if ((len(last_training_data_indexes) == len(data_indexes)) and
                    (sum(last_training_data_indexes == data_indexes) == len(data_indexes)) and
                    (np.max(coef_index_range) < 2)):
                counter = counter + 1
            else:
                counter = 0
            if (counter > 3):
                break
            last_training_data_indexes = data_indexes
        return np.array([int(i) for i in clf_predict_proba_X > 0.5])

    def run_autood(self, dataset):
        data = self._load_dataset()
        if data is None:
            return AutoODResults(error_message=f"Cannot load data from file {self.params.filepath}")
        X,y = data
        self.logger.info(f"Dataset size = {np.shape(X)}, dataset label size = {np.shape(y) if y is not None else None}")
        L, scores,instance_index_ranges, detector_index_ranges = self._run_unsupervised_outlier_detection_methods(X, y)
        if y is not None:
            self.run_mv(L, y)
        prediction_results = self.autood_training(L, scores, X, y, instance_index_ranges, detector_index_ranges)
        if database == "y":  # # DHDB
            final_df = pd.DataFrame()
            final_df["id"] = X.index
            final_df["prediction"] = prediction_results
            final_df["correct"] = np.where(prediction_results == y, 1, 0)
            final_df = final_df.convert_dtypes()
            insert_input("predictions", final_df)
        result_filename = f"results_{dataset}_{int(time.time())}.csv"
        pd.DataFrame(prediction_results).to_csv(f"results/{result_filename}")
        self.logger.info(f"Length of prediction results = {len(prediction_results)}")
        if y is not None:
            self.autood_f1_score = metrics.f1_score(y, prediction_results)
            self.logger.info(f"Final AutoOD F-1 Score= {self.autood_f1_score}")
        return AutoODResults(best_unsupervised_f1_score=self.best_unsupervised_f1_score,
                             best_unsupervised_methods=self.best_unsupervised_methods,
                             mv_f1_score=self.mv_f1_score,
                             autood_f1_score=self.autood_f1_score,
                             results_file_name=result_filename)


def get_default_detection_method_list():
    return [OutlierDetectionMethod.LOF, OutlierDetectionMethod.KNN,
            OutlierDetectionMethod.Manalanobis]


def run_autood(filepath, logger, outlier_min, outlier_max, detection_methods, index_col_name, label_col_name):
    dataset = Path(filepath).stem
    logger.info(f"Dataset Name = {dataset}")
    default_parameters = get_default_parameters(dataset)
    if outlier_min and outlier_max:
        logger.info(f"Outlier Range defined as [{outlier_min}%, {outlier_max}%]")
        outlier_min_percent = outlier_min * 0.01
        outlier_max_percent = outlier_max * 0.01
        interval = (outlier_max_percent - outlier_min_percent) / 5
        new_N_range = [round(x, 5) for x in np.arange(outlier_min_percent, outlier_max_percent + interval, interval)]
        default_parameters.N_range = new_N_range
    return AutoOD(AutoODParameters(
        filepath,
        detection_methods,
        k_range=default_parameters.k_range,
        if_range=default_parameters.if_range,
        N_range=default_parameters.N_range,
        index_col_name=index_col_name,
        label_col_name=label_col_name
    ), logger).run_autood(dataset)
