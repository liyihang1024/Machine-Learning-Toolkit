import sys
import io
import os
import time
import pprint
from PIL import Image
import pandas as pd
import numpy as np
import multiprocessing
from functools import partial
from prettytable import PrettyTable
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableView, QTableWidgetItem, qApp, QAction, QMenu, QFileDialog, QMessageBox, QGraphicsEllipseItem
from Ui_ML_GUI import Ui_MainWindow
from PyQt5.QtGui import QStandardItemModel, QStandardItem

import warnings
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, LeaveOneOut, LeavePOut, cross_validate, cross_val_score
import matplotlib.pyplot as plt
import joblib

from PyQt5.QtWidgets import QDialog, QSystemTrayIcon, QGraphicsScene, QGraphicsView, QTreeWidget, QTreeWidgetItem
from PyQt5.QtGui import QBrush, QColor, QPainter, QPen, QPixmap, QIcon
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# 导入回归算法
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor, BaggingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# 导入分别类算法
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

######################## 导入数据子线程 ################################

class ImportDataThread(QThread):
    # 声明一个信号，通过 "信号名 = pyqtSignal(类型)" 实现
    finished = pyqtSignal(bool)
    data_loaded = pyqtSignal(pd.DataFrame)

    # 更新进度条信号
    progress_bar_update = pyqtSignal(int)

    def __init__(self, path): # 带一个参数path
        super().__init__()
        self.path = path

    def run(self):
        try:
            # 加载Excel文件
            data = pd.read_excel(self.path, index_col=0, header=0)

            # 发送数据到主线程,通过 "信号名.emit(信号内容)" 实现发射信号
            self.data_loaded.emit(data)
            self.finished.emit(True)
        except Exception as e:
            self.finished.emit(False)


######################## ML训练模型子线程 ################################

class MachineLearningThread(QThread):
    # 机器学习预测结果信号
    ml_results = pyqtSignal(list)
    # 更新进度条信号
    progress_bar_update = pyqtSignal(int)

    def __init__(self, data, 
                 algorithm_instance,
                 CV_method,
                 algorithm_category,
                 algorithm_name,
                 test_size,
                 shuffle,
                 random_state,
                 data_scaling,
                 n_jobs,
                 save_model,
                 n_splits,
                 leave_out
                 ):
        super().__init__()
        self.data = data
        self.algorithm_instance = algorithm_instance  # 算法实例
        self.CV_method = CV_method                    # 交叉验证方法
        self.algorithm_category = algorithm_category  # 算法类型
        self.algorithm_name = algorithm_name          # 算法名称
        self.test_size = test_size                    # 测试集大小
        self.shuffle = shuffle                        # 划分训练/测试集前是否打乱数据
        self.random_state = random_state              # 随机种子(42或者None)
        self.data_scaling = data_scaling              # 数据归一化方法
        self.n_jobs = n_jobs                           # CPU核心数
        self.save_model = save_model
        self.n_splits = n_splits                      # 交叉验证次数
        self.leave_out = leave_out
        # self.kwargs = kwargs

    def run(self):
        results = self.cross_validation(self.data,
                                        self.algorithm_instance,
                                        self.CV_method,
                                        self.algorithm_category,
                                        self.algorithm_name,
                                        self.test_size,
                                        self.shuffle,  
                                        self.random_state,
                                        self.data_scaling,
                                        self.n_jobs,
                                        self.save_model,    
                                        self.n_splits,
                                        self.leave_out
                                        )      
        self.ml_results.emit(results)
    
    def data_preprocessing(self, X, data_scaling):
        # 数据归一化处理,将归一化之后的X转换为DataFrame格式,加上index和columns
        if data_scaling == "Z-score":
            X_scalered = StandardScaler().fit_transform(X)

        elif data_scaling == "MinMax":
            X_scalered = MinMaxScaler().fit_transform(X)

        elif data_scaling == "MaxAbs":
            X_scalered = MaxAbsScaler().fit_transform(X)

        else:
            X_scalered = X
            # print('没有归一化')

        X_scalered = pd.DataFrame(X_scalered, index=X.index, columns=X.columns)

        return X_scalered

    def save_models(self, num, model, algorithm_name):
        # 保存模型
        # 判断文件夹是否存在,如果不存在，则创建文件夹
        folder_path = 'model'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        localtime = time.localtime()
        time_ = time.strftime('%Y%m%d_%H%M%S', localtime)
        filename = f'{folder_path}/{num}_{time_}_{str(algorithm_name)[:-2]}.kpi'  # 定义文件名
        joblib.dump(model, filename)

    def cross_validation(self, data, 
                         algorithm_instance,  # 算法实例 
                         CV_method,           # 交叉验证方法
                         algorithm_category,  # 算法类型
                         algorithm_name,      # 算法名称
                         test_size,           # 测试集大小
                         shuffle,             # 划分训练/测试集前是否打乱数据
                         random_state,        # 随机种子(42或者None)
                         data_scaling,        # 数据归一化方法
                         n_jobs,              # CPU核心数
                         save_model,
                         n_splits,
                         leave_out):
        # print(f'save_model:{save_model}')
        """
        实现不同类型的交叉验证

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            训练数据集的特征矩阵
        y : array-like, shape (n_samples,)
            训练数据集的目标变量向量
        algorithm : 
            机器学习算法实例
        CV_method : str
            交叉验证方法，可以是'simple'（简单交叉验证）、'kfold'（k折交叉验证）、
            'leave_one_out'（留一法交叉验证）或'stratified'（分层交叉验证）。
        test_size : float, optional (default=0.2)
            简单交叉验证中测试集的比例
        k : int, default=None
            当 method='k_fold' 时，指定 k 的值。默认为 None，表示 5 折交叉验证
        shuffle : bool, default=False
            当 method='k_fold' 或 method='stratified' 时，指定是否打乱数据集
        random_state : int, default=None
            当 shuffle=True 时，指定随机数生成器的种子值

        Returns
        -------
        scores : list of float
            交叉验证的准确率列表
        """

        # 划分为features和targets
        X, y = data.iloc[:, :-1], data.iloc[:, -1]

        # 数据归一化处理
        X_scalered = self.data_preprocessing(X, data_scaling)

        # 创建一个模型实例
        model = algorithm_instance

        # 数据划分
        X_train, X_test, y_train, y_test = train_test_split(X_scalered, y,
                                                            test_size=test_size, 
                                                            shuffle=shuffle, 
                                                            random_state=random_state)

        if CV_method == 'simple':
            # 简单交叉验证
            # print(f"K fold:{n_splits}")

            # 模型训练
            model.fit(X_train, y_train)

            # 保存模型
            if save_model:
                self.save_models(0, model, algorithm_instance)

            if algorithm_category == '回归':
                # 在训练集上进行预测
                y_train_pred = model.predict(X_train)
                train_r2 = r2_score(y_train, y_train_pred)
                train_mae = mean_absolute_error(y_train, y_train_pred)
                train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

                # 在测试集上进行预测
                y_test_pred = model.predict(X_test)
                test_r2 = r2_score(y_test, y_test_pred)
                test_mae = mean_absolute_error(y_test, y_test_pred)
                test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
               
                # 训练及预测数据
                data_pred = {'TrainSet':X_train.index,
                             'y_train':y_train.values,
                             'y_train_pred':y_train_pred,
                             'TestSet':X_test.index,
                             'y_test':y_test.values,
                             'y_test_pred':y_test_pred}
                
                # 训练评估指标
                scores = {'train_r2':train_r2,
                          'train_mae':train_mae,
                          'train_rmse':train_rmse,
                          'test_r2':test_r2,
                          'test_mae':test_mae,
                          'test_rmse':test_rmse}
    
                # 将训练及预测数据转换为DataFrame,不同长度的缺失值用nan填充
                data_pred = pd.DataFrame({k: pd.Series(v) for k, v in data_pred.items()})
                scores = pd.DataFrame({k: pd.Series(v) for k, v in scores.items()})

                # 将预测得分指标保留三位小数
                scores = scores.round(3)
            
            elif algorithm_category == '分类':
                # 在训练集上进行预测
                y_train_pred = model.predict(X_train)
                train_acc = accuracy_score(y_train, y_train_pred)
                train_f1 = f1_score(y_train, y_train_pred)
                train_auc = roc_auc_score(y_train, y_train_pred)

                # 在测试集上进行预测
                y_test_pred = model.predict(X_test)
                test_acc = accuracy_score(y_test, y_test_pred)
                test_f1 = f1_score(y_test, y_test_pred)
                test_auc = roc_auc_score(y_test, y_test_pred)
               
                # 训练及预测数据
                data_pred = {'TrainSet':X_train.index,
                             'y_train':y_train.values,
                             'y_train_pred':y_train_pred,
                             'TestSet':X_test.index,
                             'y_test':y_test.values,
                             'y_test_pred':y_test_pred}
                
                # 训练评估指标
                scores = {'train_acc':train_acc,
                          'train_f1':train_f1,
                          'train_auc':train_auc,
                          'test_acc':test_acc,
                          'test_f1':test_f1,
                          'test_auc':test_auc}
    
                # 将训练及预测数据转换为DataFrame,不同长度的缺失值用nan填充
                data_pred = pd.DataFrame({k: pd.Series(v) for k, v in data_pred.items()})
                scores = pd.DataFrame({k: pd.Series(v) for k, v in scores.items()})

                # 将预测得分指标保留三位小数
                scores = scores.round(3)
                
            elif algorithm_category == '聚类':
                pass
            elif algorithm_category == '降维':
                pass
            else:
                pass

        elif CV_method == 'k_fold':
            # k 折交叉验证
            # n_splits = kwargs.get('n_splits')
            
            if algorithm_category == '回归':
                # 指定评分函数列表
                scoring = {'r2': make_scorer(r2_score),
                        'mae': make_scorer(mean_absolute_error),
                        'rmse': make_scorer(mean_squared_error, squared=False)}
                
                # 构造交叉验证迭代器对象
                # print(f"K fold:{n_splits}")
                kf = KFold(n_splits=n_splits, shuffle=False)
                # 进行交叉验证并返回每一次交叉验证的训练数据得分和评估器
                cv_results = cross_validate(model, X_train, y_train, cv=kf, scoring=scoring, 
                                            return_train_score=True, return_estimator=True, n_jobs=n_jobs)
                
                # 遍历每个拆分的估计器列表,在独立测试集上预测并保存模型
                test_r2 = []
                test_mae = []
                test_rmse = []
                # 保存每一折的预测数据
                data_pred = {}
                # 根据每一次交叉验证的训练数据和验证数据分别得到其预测值
                for i, estimator, index in zip(range(n_splits), cv_results['estimator'], kf.split(X_train, y_train)):
                    i += 1 # 从1开始计数

                    # 每一折的训练集及验证集索引
                    train_index = index[0]
                    val_index = index[1]
                    # 每一折的训练集
                    X_train_fold = X_train.iloc[train_index]
                    y_train_fold = y_train.iloc[train_index]
                    # 每一折的验证集
                    X_val_fold = X_train.iloc[val_index]
                    y_val_fold = y_train.iloc[val_index]

                    # 每一折在训练集/验证集/测试集上的预测结果
                    y_train_pred = estimator.predict(X_train_fold)
                    y_val_pred = estimator.predict(X_val_fold)
                    y_test_pred = estimator.predict(X_test)

                    # 将每一折的训练集/验证集/测试集数据及其索引依次保存在字典中
                    data_pred[f'y_train_index_F{i}'] = y_train_fold.index
                    data_pred[f'y_train_F{i}']       = y_train_fold.values
                    data_pred[f'y_train_pred_F{i}']  = y_train_pred
                    data_pred[f'y_val_index_F{i}']   = y_val_fold.index
                    data_pred[f'y_val_F{i}']         = y_val_fold.values
                    data_pred[f'y_val_pred_F{i}']    = y_val_pred
                    data_pred[f'y_test_index_F{i}']  = y_test.index
                    data_pred[f'y_test_F{i}']        = y_test.values
                    data_pred[f'y_test_pred_F{i}']   = y_test_pred

                    # 每一折的模型在独立测试集上的预测得分
                    test_r2_ = r2_score(y_test, y_test_pred)
                    test_mae_ = mean_absolute_error(y_test, y_test_pred)
                    test_rmse_ = mean_squared_error(y_test, y_test_pred, squared=False)
                    
                    test_r2.append(test_r2_)
                    test_mae.append(test_mae_)
                    test_rmse.append(test_rmse_)

                    # 保存模型
                    if save_model:
                        self.save_models(i, model, algorithm_name)

                # 训练评估指标
                scores = {'train_r2':cv_results['train_r2'],
                          'train_mae':cv_results['train_mae'],
                          'train_rmse':cv_results['train_rmse'],
                          'val_r2':cv_results['test_r2'],
                          'val_mae':cv_results['test_mae'],
                          'val_rmse':cv_results['test_rmse'],
                          'test_r2':np.array(test_r2),
                          'test_mae':np.array(test_mae),
                          'test_rmse':np.array(test_rmse)}

         

                # # 训练及预测数据
                # data_pred = {'TrainSet':X_train.index,
                #             'y_train':y_train.values,
                #             'y_train_pred':y_train_pred,
                #             'TestSet':X_test.index,
                #             'y_test':y_test.values,
                #             'y_test_pred':y_test_pred}
    
                # 将训练及预测数据转换为DataFrame,不同长度的缺失值用nan填充
                data_pred = pd.DataFrame({k: pd.Series(v) for k, v in data_pred.items()})
                scores = pd.DataFrame({k: pd.Series(v) for k, v in scores.items()})

                # 为scores添加一列索引，求每一列的均值并添加在最后一行
                scores.index = [i+1 for i in range(scores.shape[0])]
                scores.index.name = 'fold'

                # 计算每一列的均值并添加到最后一行
                scores.loc['mean'] = scores.mean() 

                # 将预测得分指标保留三位小数并将索引添加为新列
                scores = scores.round(3).reset_index()

        elif CV_method == 'leave_one_or_P_out':
            # 留一法或者留P法交叉验证

            # 留一法每次只有一个测试样本，少于2个样本将无法计算test_R2，可用以下方法将报出的警告信息忽略
            # 忽略UndefinedMetricWarning警告
            warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
            # 如果希望再次打印出已经被忽略的警告信息，可以使用以下代码将警告设置为默认模式：
            # warnings.filterwarnings('default', category=UndefinedMetricWarning)
            
            def leave_one_or_P_out_regression(leave_out, n_splits):
                # 定义留一/P法的回归模型

                # 指定评分函数列表
                scoring = {'r2': make_scorer(r2_score),
                        'mae': make_scorer(mean_absolute_error),
                        'rmse': make_scorer(mean_squared_error, squared=False)}
                
                if leave_out == 1:
                    # 进入了留一法

                    loo = LeaveOneOut()

                    # 留一法交叉验证
                    cv_results = cross_validate(model, X_train, y_train, cv=loo, scoring=scoring, 
                                                return_train_score=True, return_estimator=True, n_jobs=n_jobs)
                    
                    # 遍历每个拆分的估计器列表,在独立测试集上预测并保存模型
                    test_r2 = []
                    test_mae = []
                    test_rmse = []

                    # 保存验证集r2，实际上包含只有一个值
                    val_r2 = []
                    
                    # 得到每个val样本的预测值以计算测试集R2
                    y_val_pred_all = np.zeros(y_train.shape)
                    
                    # 保存每一折的预测数据
                    data_pred = {}
                    
                    # 根据每一次交叉验证的训练数据和验证数据分别得到其预测值
                    for i, estimator, index in zip(range(n_splits), cv_results['estimator'], loo.split(X_train, y_train)):
                        i += 1 # 从1开始计数

                        # 每一折的训练集及验证集索引
                        train_index = index[0]
                        val_index = index[1]
                        # 每一折的训练集
                        X_train_fold = X_train.iloc[train_index]
                        y_train_fold = y_train.iloc[train_index]
                        # 每一折的验证集
                        X_val_fold = X_train.iloc[val_index]
                        y_val_fold = y_train.iloc[val_index]

                        # 每一折在训练集/验证集/测试集上的预测结果
                        y_train_pred = estimator.predict(X_train_fold)
                        y_val_pred = estimator.predict(X_val_fold)
                        y_test_pred = estimator.predict(X_test)
                        
                        # 将预测结果保存下来，即验证集预测结果
                        y_val_pred_all[i-1] = y_val_pred

                        # 将每一折的训练集/验证集/测试集数据及其索引依次保存在字典中
                        data_pred[f'y_train_index_F{i}'] = y_train_fold.index
                        data_pred[f'y_train_F{i}']       = y_train_fold.values
                        data_pred[f'y_train_pred_F{i}']  = y_train_pred
                        data_pred[f'y_val_index_F{i}']   = y_val_fold.index
                        data_pred[f'y_val_F{i}']         = y_val_fold.values
                        data_pred[f'y_val_pred_F{i}']    = y_val_pred
                        data_pred[f'y_test_index_F{i}']  = y_test.index
                        data_pred[f'y_test_F{i}']        = y_test.values
                        data_pred[f'y_test_pred_F{i}']   = y_test_pred

                        # 每一折的模型在独立测试集上的预测得分
                        test_r2_ = r2_score(y_test, y_test_pred)
                        test_mae_ = mean_absolute_error(y_test, y_test_pred)
                        test_rmse_ = mean_squared_error(y_test, y_test_pred, squared=False)
                        
                        test_r2.append(test_r2_)
                        test_mae.append(test_mae_)
                        test_rmse.append(test_rmse_)

                        # 保存模型
                        if save_model:
                            self.save_models(i, model, algorithm_instance)

                    # 训练评估指标
                    # 验证集R2
                    val_r2_ = r2_score(y_train, y_val_pred_all)
                    val_r2.append(round(val_r2_, 3))

                    scores = {'train_r2':cv_results['train_r2'],
                                'train_mae':cv_results['train_mae'],
                                'train_rmse':cv_results['train_rmse'],
                                'val_r2':np.array(val_r2), # 只有一个值
                                'val_mae':cv_results['test_mae'],
                                'val_rmse':cv_results['test_rmse'],
                                'test_r2':np.array(test_r2),
                                'test_mae':np.array(test_mae),
                                'test_rmse':np.array(test_rmse)}
                    
                    # 将训练及预测数据转换为DataFrame,不同长度的缺失值用nan填充
                    data_pred = pd.DataFrame({k: pd.Series(v) for k, v in data_pred.items()})
                    scores = pd.DataFrame({k: pd.Series(v) for k, v in scores.items()})

                    # 为scores添加一列索引，求每一列的均值并添加在最后一行
                    scores.index = [i+1 for i in range(scores.shape[0])]
                    scores.index.name = 'fold'

                    # 计算每一列的均值并添加到最后一行.有nan值的DataFrame调用mean()方法时，
                    # 默认情况下会自动忽略缺失值（NaN），因此可以计算其平均值。
                    scores.loc['mean'] = scores.mean()

                    # 将预测得分指标保留三位小数并将索引添加为新列
                    scores = scores.round(3).reset_index()
                
                elif leave_out > 1:
                    # 进入了留P法

                    # 留P法交叉验证
                    lpo = LeavePOut(leave_out)
                    # n_splits = lpo.get_n_splits(X)
                    # algorithm_category = kwargs.get('algorithm_category')
                    
                    if algorithm_category == '回归':
                        # 指定评分函数列表
                        scoring = {'r2': make_scorer(r2_score),
                                'mae': make_scorer(mean_absolute_error),
                                'rmse': make_scorer(mean_squared_error, squared=False)}
                        
                        # 构造交叉验证迭代器对象
                        # kf = KFold(n_splits=n_splits, shuffle=False)
                        # 进行交叉验证并返回每一次交叉验证的训练数据得分和评估器
                        cv_results = cross_validate(model, X_train, y_train, cv=lpo, scoring=scoring, 
                                                    return_train_score=True, return_estimator=True, n_jobs=n_jobs)
                        
                        # 遍历每个拆分的估计器列表,在独立测试集上预测并保存模型
                        test_r2 = []
                        test_mae = []
                        test_rmse = []
                        # 保存每一折的预测数据
                        data_pred = {}
                        # 根据每一次交叉验证的训练数据和验证数据分别得到其预测值
                        for i, estimator, index in zip(range(n_splits), cv_results['estimator'], lpo.split(X_train, y_train)):
                            i += 1 # 从1开始计数

                            # 每一折的训练集及验证集索引
                            train_index = index[0]
                            val_index = index[1]
                            # 每一折的训练集
                            X_train_fold = X_train.iloc[train_index]
                            y_train_fold = y_train.iloc[train_index]
                            # 每一折的验证集
                            X_val_fold = X_train.iloc[val_index]
                            y_val_fold = y_train.iloc[val_index]

                            # 每一折在训练集/验证集/测试集上的预测结果
                            y_train_pred = estimator.predict(X_train_fold)
                            y_val_pred = estimator.predict(X_val_fold)
                            y_test_pred = estimator.predict(X_test)

                            # 将每一折的训练集/验证集/测试集数据及其索引依次保存在字典中
                            data_pred[f'y_train_index_F{i}'] = y_train_fold.index
                            data_pred[f'y_train_F{i}']       = y_train_fold.values
                            data_pred[f'y_train_pred_F{i}']  = y_train_pred
                            data_pred[f'y_val_index_F{i}']   = y_val_fold.index
                            data_pred[f'y_val_F{i}']         = y_val_fold.values
                            data_pred[f'y_val_pred_F{i}']    = y_val_pred
                            data_pred[f'y_test_index_F{i}']  = y_test.index
                            data_pred[f'y_test_F{i}']        = y_test.values
                            data_pred[f'y_test_pred_F{i}']   = y_test_pred

                            # 每一折的模型在独立测试集上的预测得分
                            test_r2_ = r2_score(y_test, y_test_pred)
                            test_mae_ = mean_absolute_error(y_test, y_test_pred)
                            test_rmse_ = mean_squared_error(y_test, y_test_pred, squared=False)
                            
                            test_r2.append(test_r2_)
                            test_mae.append(test_mae_)
                            test_rmse.append(test_rmse_)

                            # 保存模型
                            if save_model:
                                self.save_models(i, model, algorithm_instance)

                        # 训练评估指标
                        scores = {'train_r2':cv_results['train_r2'],
                                'train_mae':cv_results['train_mae'],
                                'train_rmse':cv_results['train_rmse'],
                                'val_r2':cv_results['test_r2'],
                                'val_mae':cv_results['test_mae'],
                                'val_rmse':cv_results['test_rmse'],
                                'test_r2':np.array(test_r2),
                                'test_mae':np.array(test_mae),
                                'test_rmse':np.array(test_rmse)}
            
                        # 将训练及预测数据转换为DataFrame,不同长度的缺失值用nan填充
                        data_pred = pd.DataFrame({k: pd.Series(v) for k, v in data_pred.items()})
                        scores = pd.DataFrame({k: pd.Series(v) for k, v in scores.items()})

                        # 为scores添加一列索引，求每一列的均值并添加在最后一行
                        scores.index = [i+1 for i in range(scores.shape[0])]
                        scores.index.name = 'fold'

                        # 计算每一列的均值并添加到最后一行
                        scores.loc['mean'] = scores.mean() 

                        # 将预测得分指标保留三位小数并将索引添加为新列
                        scores = scores.round(3).reset_index()

                else:
                    pass

                return data_pred, scores


            # 获取留出的样本数
            # leave_out = self.spinBox_4.value()
            
            # 计算需要训练的模型数目，超过100个时需要用确认是否继续执行
            n_splits = LeaveOneOut().get_n_splits(X_train) if leave_out==1 else LeavePOut(leave_out).get_n_splits(X_train)
            
            if algorithm_category == '回归':
                # 进入了回归模型

                data_pred, scores = leave_one_or_P_out_regression(leave_out, n_splits)

                # if n_splits < 100:
                #     data_pred, scores = leave_one_or_P_out_regression(leave_out, n_splits)
                
                # elif n_splits >= 100:
                #     reply = self.confirm_message(f'总共需要训练{n_splits}个模型,可能需要一万年,是否继续执行!')
                #     # reply = QMessageBox.question(self, '确认', f'总共需要训练{n_splits}个模型,可能需要大量的时间,是否继续执行!', 
                #     #                             QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                #     if reply == QMessageBox.Yes:
                #         # 继续执行
                #         data_pred, scores = leave_one_or_P_out_regression(leave_out, n_splits)
            
                #     else:
                #         # '取消执行
                #         data_pred, scores = pd.DataFrame(), pd.DataFrame()

        elif CV_method == 'stratified':
            # 分层交叉验证
            # n_splits = kwargs.get('n_splits')
            # shuffle = kwargs.get('shuffle')
            # random_state = kwargs.get('random_state')
            skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
            # 指定评分函数列表
            scoring = {'r2': make_scorer(r2_score),
                    'mae': make_scorer(mean_absolute_error),
                    'rmse': make_scorer(mean_squared_error, squared=False)}
            
            cv_results = cross_validate(model, X_train, y_train, cv=skf, scoring=scoring, 
                                        return_train_score=True, return_estimator=True, n_jobs=n_jobs)

            # scores = cross_val_score(model, X, y, cv=skf, scoring='r2')
            
            # scores = []
            # for train_index, test_index in skf.split(X, y):
            #     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            #     y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            #     model.fit(X_train, y_train)
            #     y_pred = model.predict(X_test)
            # score = r2_score(y_test, y_pred)
            # scores.append(score)
        else:
            raise ValueError("Invalid method parameter.")

        return [data_pred, scores]


######################## GUI主线程 ################################

class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MyMainWindow, self).__init__()
        self.setupUi(self)
        self.setWindowIcon(QIcon('icon/icon.jpg'))
        self.setWindowTitle('Machine Learning Toolkit')

        self.pushButton_2.setEnabled(False)
        self.pushButton_3.setEnabled(False)

        # 禁用交叉验证的所有单选框
        for button in self.buttonGroup.buttons():
            button.setEnabled(False)
        self.spinBox.setEnabled(False)
        self.spinBox_2.setEnabled(False)
        self.spinBox_4.setEnabled(False)

        self.treeWidget.itemClicked.connect(self.select_algorithm)
        # self.pushButton.clicked.connect(self.import_data)
        self.pushButton.clicked.connect(self.load_data)
        self.pushButton_2.clicked.connect(self.run_machine_learning)
        # self.pushButton_3.clicked.connect(lambda: self.confirm_message('你想中断模型训练？'))
        self.pushButton_3.clicked.connect(partial(self.confirm_message, '根本停不下来~'))

        # 设置CPU核心数的提示信息
        self.num_cores = os.cpu_count()
        self.spinBox_3.setToolTip(f'此电脑总共有{self.num_cores}个CPU核心！')
        self.spinBox_3.setMaximum(self.num_cores)  # 将CPU核心数设置为该spinBox的最大取值范围

        # About信息
        self.about = """
        <html><head/><body><p align="center"><span style=" font-size:12pt; 
        font-weight:600;">Machine Learning Toolkit</span></p><p align="center">
        <span style=" font-size:12pt;">当前版本：1.1.5</span></p><p align="center">
        <span style=" font-size:12pt;">更新时间：2023年03月23日 18:42:19</span></p>
        <p align="center"><span style=" font-size:12pt;">开发者：Yihang Li</span>
        <br/></p><p align="justify"><span style=" font-size:12pt;">   该项目在上海
        大学材料基因组工程研究院FLY教授的支持下完成，旨在设计一个用于实现各种机器学习算法
        预测的GUI框架，用于帮助课题组内想了解机器学习但是没有编程背景的同学将机器学习应
        用到自己的科研当中。</span></p></body></html>
        """
        self.action_2.triggered.connect(self.showAbout)
    def showAbout(self):
        QMessageBox.about(self, '关于本软件', self.about)

##############################################
        # 设置菜单
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.showContextMenu)
        self.menu = QMenu(self)
        self.export_action = QAction(QIcon("icon/icon_no_bg.png"), "Export to Excel", self)
        self.export_action.triggered.connect(self.export_to_excel)
        self.menu.addAction(self.export_action)

        # self.setCentralWidget(self.table_view)

    def showContextMenu(self, pos):
        self.menu.exec_(self.mapToGlobal(pos))

    def export_to_excel(self):
        table_view = self.sender()
        if not isinstance(table_view, QTableView):
            return
        model = table_view.model()
        if not isinstance(model, QStandardItemModel):
            return
        data = []
        for row in range(model.rowCount()):
            data.append([model.index(row, col).data() for col in range(model.columnCount())])
        df = pd.DataFrame(data=data, columns=[model.headerData(col, Qt.Horizontal) for col in range(model.columnCount())])
        file_path, _ = QFileDialog.getSaveFileName(self, "Save file", "", "Excel files (*.xlsx)")
        if file_path:
            writer = pd.ExcelWriter(file_path, engine="xlsxwriter")
            df.to_excel(writer, index=False)
            writer.save()
##############################################

    def load_data(self):
        # 打开一个子线程导入数据
        path, _ = QFileDialog.getOpenFileName(self, "Open Excel File", "", "Excel Files (*.xlsx *.xls)")
        if path:
            # 实例化一个线程,名为thread,参数path设置为文件路径path
            self.thread = ImportDataThread(path)
            self.thread.finished.connect(self.load_finished)
            # 将线程thread的信号data_loaded和UI主线程中的槽函数display_data进行连接
            # self.thread.data_loaded.connect(self.display_data)
            self.thread.data_loaded.connect(lambda data: self.display_data(data, "tableView"))
            """
            在上述代码中,将display_data函数的参数data改为了data和tableView两个参数,同时在连接信号和槽函数时,使用
            了lambda函数将self.tableView作为额外的参数传递给了display_data函数。在display_data函数中,我们可以
            使用传递进来的tableView参数来将数据显示在QTableWidget中.
            """
            # 启动线程，执行线程类中run函数
            self.thread.start()

            # 导入数据后在单行文本框中显示文件名
            self.lineEdit.setText(path)

    def load_finished(self, success):
        if not success:
            QMessageBox.warning(self, "错误", "加载Excel文件失败！", QMessageBox.Ok)

    def display_data(self, data, tableView_name):
        # 将导入的数据显示在tableWdget表格中
        self.data = data  # 将导入的数据设置为全局变量

        # 将索引列转换为普通列以便显示在数据预览表格中
        data = data.reset_index()
        model = QStandardItemModel(len(data), len(data.columns))
        model.setHorizontalHeaderLabels(list(data.columns))
        for i in range(len(data)):
            # 导入数据的进度条
            progress = int((i + 1) * 100 / data.shape[0])
            self.progressBar.setValue(progress)

            for j in range(len(data.columns)):
                item = QStandardItem(str(data.iloc[i, j]))
                model.setItem(i, j, item)
            
        table_view = getattr(self, tableView_name)
        # 添加数据到model.可以直接使用传递进来的tableView参数来将数据显示在QTableWidget中.
        table_view.setModel(model)
        
        # 自动调整列宽
        table_view.resizeColumnsToContents()
        
        # 弹窗提示
        self.show_message('数据导入成功,请选择算法！')
            
    def show_data_in_table(self, data, tableView_name):
        # 将预测结果数据data显示在tableView中，data为pandas.DataFrame格式
        # tableView_name为要显示数据的QTableView名
        model = QStandardItemModel(len(data), len(data.columns))
        model.setHorizontalHeaderLabels(list(data.columns))
        # print(data.shape)
        for i in range(len(data)):
            # print('行',i)
            # 显示预测数据的进度条
            progress = int((i + 1) * 100 / data.shape[0])
            # print(progress)
            self.progressBar.setValue(progress)

            for j in range(len(data.columns)):
                item = QStandardItem(str(data.iloc[i, j]))
                model.setItem(i, j, item)
            
        table_view = getattr(self, tableView_name)
        # 添加数据到model
        table_view.setModel(model)
        # 自动调整列宽
        table_view.resizeColumnsToContents()

    def convertPlainTextToDict(self):
        # 读取PlainTextEdit中的文本
        # plainTextEdit = self.findChild(QPlainTextEdit, "plainTextEdit")
        text = self.plainTextEdit.toPlainText()

        # 将文本转换为字典格式
        data = {}
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            key, value = line.split(": ")
            data[key] = value
        # print(data)  

    def show_message(self, message):
        # 弹出信息提示窗
        QMessageBox.information(self, '提示', message)
    
    def confirm_message(self, message):
        # 使用QMessageBox.question获取用户选择,弹出选择提示窗，让用户决定是否继续执行当前函数
        reply = QMessageBox.question(self, '确认', message, 
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        return reply

    def select_algorithm(self):
        # check if the clicked item is a leaf node (i.e. no child items)
        # 获取选择的算法及其类型(回归or分类)
        current_item = self.treeWidget.currentItem()  # 鼠标当前选中的项目
        # print('当前项目', current_item)
        parent_item = current_item.parent()           # 鼠标当前选中项目的父项目
        if parent_item is not None:
            algorithm_category = parent_item.text(0)   # 算法类型
            algorithm_name = current_item.text(0)      # 算法名称
            # print(f"您选择了{algorithm_category}中的{algorithm_name}算法,实例为{algorithm_instance}")
        
        # print(f"当前算法的子项目有{current_item.childCount()}个")
        if current_item.childCount() == 0:
            # 当前选中算法的分类
            # parent_item = item.parent()
            # category_name = parent_item.text(column)

            if algorithm_category == "回归":
                # 控制启用的交叉验证方法
                self.radioButton.setEnabled(True)
                self.radioButton_2.setEnabled(True)
                self.radioButton_3.setEnabled(True)
                self.spinBox.setEnabled(True)
                self.spinBox_4.setEnabled(True)

            elif algorithm_category == "分类":
                # 控制启用的交叉验证方法
                self.radioButton.setEnabled(True)
                self.radioButton_2.setEnabled(True)
                self.radioButton_3.setEnabled(True)
                self.radioButton_4.setEnabled(True)
                self.spinBox.setEnabled(True)
                self.spinBox_2.setEnabled(True)
                self.spinBox_4.setEnabled(True)
            
            elif algorithm_category == "聚类":
                pass

            elif algorithm_category == "降维":
                pass

            else:
                pass


            # 所有定义的回归算法
            if algorithm_name == '线性回归':
                model = LinearRegression()
            elif algorithm_name == '岭回归':
                model = Ridge()
            elif algorithm_name == 'LASSO回归':
                model = Lasso()
            elif algorithm_name == 'ElasticNet回归':
                model = ElasticNet()
            elif algorithm_name == '多项式回归':
                model = PolynomialFeatures()
            elif algorithm_name == '支持向量机回归':
                model = SVR()
            elif algorithm_name == '决策树回归':
                model = DecisionTreeRegressor()
            elif algorithm_name == '随机森林回归':
                model = RandomForestRegressor()
            elif algorithm_name == '梯度提升回归':
                model = GradientBoostingRegressor()
            elif algorithm_name == 'ExtraTree回归':
                model = ExtraTreesRegressor()
            elif algorithm_name == 'AdaBoost回归':
                model = AdaBoostRegressor()
            elif algorithm_name == 'XGBoost回归':
                model = XGBRegressor()
            elif algorithm_name == 'LightGBM回归':
                model = LGBMRegressor()
            elif algorithm_name == 'CatBoost回归':
                model = CatBoostRegressor()
            elif algorithm_name == 'Bagging回归':
                model = BaggingRegressor()
            else:
                pass
            
            # 所有定义的分类算法
            if algorithm_name == '逻辑回归':
                model = LogisticRegression()
            elif algorithm_name == 'K近邻分类':
                model = KNeighborsClassifier()
            elif algorithm_name == '决策树分类':
                model = DecisionTreeClassifier()
            elif algorithm_name == '随机森林分类':
                model = RandomForestClassifier()
            elif algorithm_name == 'AdaBoost分类':
                model = AdaBoostClassifier()
            elif algorithm_name == 'XGBoost分类':
                model = xgb.XGBClassifier()
            elif algorithm_name == 'CatBoost分类':
                model = cb.CatBoostClassifier()
            elif algorithm_name == 'Bagging分类':
                model = BaggingClassifier(base_estimator=DecisionTreeClassifier())
            elif algorithm_name == '支持向量分类':
                model = SVC()
            elif algorithm_name == '多层感知器分类':
                model = MLPClassifier()
            elif algorithm_name == '高斯朴素贝叶斯分类':
                model = GaussianNB()
            else:
                pass
            
            # 将模型参数打印在“参数设置”文本框中
            algorithm_params = model.get_params()
            # print(algorithm_params)
            text = ""
            for key, value in algorithm_params.items():
                text += f"{key}: {value}\n"
            # 打印参数
            self.plainTextEdit.setPlainText(text)
            # 放开"开始"按钮
            self.pushButton_2.setEnabled(True)

            # 返回算法实例
            return model

    
    def show_metrics(self, metrics_mean):
        # 设置显示分类算法的metric名
        metric_lineEdit = ['lineEdit_7', 'lineEdit_8', 'lineEdit_9', 'lineEdit_10', 'lineEdit_11', 'lineEdit_12', 'lineEdit_13', 'lineEdit_14', 'lineEdit_15']
        for lineEdit_name, score_mean in zip(metric_lineEdit, metrics_mean):
            lineEdit_name = getattr(self, lineEdit_name)
            lineEdit_name.setText(score_mean)
       
    
    def run_machine_learning(self):
        # 获取所选算法实例并进行机器学习

        # 释放stop按钮
        self.pushButton_3.setEnabled(True)
        # self.pushButton_3.clicked.connect(self.confirm_message(self, 'nnn'))
        # self.pushButton_3.clicked.connect(lambda: self.confirm_message('Hello World'))

        # 获取选择的CPU核心数
        n_jobs = self.spinBox_3.value()

        # 获取训练集大小,是否打乱数据,随机种子,以及是否保存模型
        test_size = round(self.doubleSpinBox.value(), 2)
        shuffle = self.checkBox.isChecked()
        random_state = 42 if self.checkBox_2.isChecked() else None
        save_model = self.checkBox_3.isChecked()

        # 获取数据归一化方法
        if self.checkBox_4.isChecked():
            data_scaling = "Z-score"
        elif self.checkBox_5.isChecked():
            data_scaling = "MinMax"
        elif self.checkBox_6.isChecked():
            data_scaling = "MaxAbs"
        else:
            data_scaling = None
            # print("没有数据归一化！")
        
        # 获取选择的算法及其类型(回归or分类)
        current_item = self.treeWidget.currentItem()  # 鼠标当前选中的项目
        parent_item = current_item.parent()           # 鼠标当前选中项目的父项目

        if parent_item is not None:
            algorithm_category = parent_item.text(0)   # 算法类型
            algorithm_name = current_item.text(0)      # 算法名称
            # print(f"您选择了{algorithm_category}中的{algorithm_name}算法,实例为{algorithm_instance}")
        
            
            # 获取算法实例名，列入传入("回归","随机森林回归"），得到"RandomForestRegressor()"
            # algorithm_instance = self.select_algorithm(current_item, algorithm_category, algorithm_name)
            algorithm_instance = self.select_algorithm()
            
            # print(f"您选择了{algorithm_category}中的{algorithm_name}算法,实例为{algorithm_instance}")
        else:
            print("请选择一个算法")

        # 请空metrics显示框内容
        self.show_metrics([""] * 9)

        # 设置显示算法的metric名
        if algorithm_category == "回归":
            metric_label = ['label_10', 'label_11', 'label_12']
            metric_text = ['R2', 'MAE', 'RMSE']
            for label_name, text in zip(metric_label, metric_text):
                label_name = getattr(self, label_name)
                label_name.setText(text)

        elif algorithm_category == "分类":
            metric_label = ['label_10', 'label_11', 'label_12']
            metric_text = ['Accuracy', 'F1-Score', 'AUC']
            for label_name, text in zip(metric_label, metric_text):
                label_name = getattr(self, label_name)
                label_name.setText(text)
                
        elif algorithm_category == "聚类":
            pass

        elif algorithm_category == "降维":
            pass

        else:
            pass
        

###########################################################
        # 交叉验证方法选择
        if self.radioButton.isChecked():
            # 简单交叉验证
            CV_method ='simple'

            # 建立ML子线程
            self.ml_thread = MachineLearningThread(self.data,
                                                   algorithm_instance,  # 算法实例
                                                   CV_method,           # 交叉验证方法
                                                   algorithm_category,  # 算法类型
                                                   algorithm_name,      # 算法名称
                                                   test_size,           # 测试集大小
                                                   shuffle,             # 划分训练/测试集前是否打乱数据
                                                   random_state,        # 随机种子(42或者None)
                                                   data_scaling,        # 数据归一化方法
                                                   n_jobs,              # CPU核心数
                                                   save_model,
                                                   n_splits=1,
                                                   leave_out=1
                                                   )
            self.ml_thread.ml_results.connect(self.process_ml_results)
            self.ml_thread.start()

            
                
# self.thread.data_loaded.connect(lambda data: self.display_data(data, self.tableView))
        elif self.radioButton_2.isChecked():
            # 留一/P法交叉验证
            CV_method ='leave_one_or_P_out'

            # 获取留出的样本数
            leave_out = self.spinBox_4.value()

            # 此处划分数据仅仅是为了计算留一/P法的n_splits
            X, y = self.data.iloc[:, :-1], self.data.iloc[:, -1]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
 
            # 计算需要训练的模型数目，超过100个时需要用确认是否继续执行
            n_splits = LeaveOneOut().get_n_splits(X_train) if leave_out==1 else LeavePOut(leave_out).get_n_splits(X_train)

            if n_splits < 100:
                # 建立ML子线程
                self.ml_thread = MachineLearningThread(self.data,
                                                       algorithm_instance,  # 算法实例
                                                       CV_method,           # 交叉验证方法
                                                       algorithm_category,  # 算法类型
                                                       algorithm_name,      # 算法名称
                                                       test_size,           # 测试集大小
                                                       shuffle,             # 划分训练/测试集前是否打乱数据
                                                       random_state,        # 随机种子(42或者None)
                                                       data_scaling,        # 数据归一化方法
                                                       n_jobs,              # CPU核心数
                                                       save_model,
                                                       n_splits,
                                                       leave_out
                                                       )
                self.ml_thread.ml_results.connect(self.process_ml_results)
                self.ml_thread.start()
            elif n_splits >= 100:
                reply = self.confirm_message(f'总共需要训练{n_splits}个模型,可能需要一万年,是否继续执行!')

                if reply == QMessageBox.Yes:
                    # 继续执行
                    # 建立ML子线程
                    self.ml_thread = MachineLearningThread(self.data,
                                                           algorithm_instance,  # 算法实例
                                                           CV_method,           # 交叉验证方法
                                                           algorithm_category,  # 算法类型
                                                           algorithm_name,      # 算法名称
                                                           test_size,           # 测试集大小
                                                           shuffle,             # 划分训练/测试集前是否打乱数据
                                                           random_state,        # 随机种子(42或者None)
                                                           data_scaling,        # 数据归一化方法
                                                           n_jobs,              # CPU核心数
                                                           save_model,
                                                           n_splits,
                                                           leave_out
                                                           )
                    self.ml_thread.ml_results.connect(self.process_ml_results)
                    self.ml_thread.start()
                else:
                    # '取消执行
                    data_pred, scores = pd.DataFrame(), pd.DataFrame()

        elif self.radioButton_3.isChecked():
            # K折交叉验证
            CV_method ='k_fold'
            n_splits = self.spinBox.value()

            # 建立ML子线程
            self.ml_thread = MachineLearningThread(self.data,
                                                   algorithm_instance,  # 算法实例
                                                   CV_method,           # 交叉验证方法
                                                   algorithm_category,  # 算法类型
                                                   algorithm_name,      # 算法名称
                                                   test_size,           # 测试集大小
                                                   shuffle,             # 划分训练/测试集前是否打乱数据
                                                   random_state,        # 随机种子(42或者None)
                                                   data_scaling,        # 数据归一化方法
                                                   n_jobs,              # CPU核心数
                                                   save_model,
                                                   n_splits,
                                                   leave_out=1
                                                   )
            self.ml_thread.ml_results.connect(self.process_ml_results)
            self.ml_thread.start()
        
        elif self.radioButton_4.isChecked():
            # 分层交叉验证
            CV_method ='stratified'
            n_splits = self.spinBox_2.value()
        
            # 建立ML子线程
            self.ml_thread = MachineLearningThread(self.data,
                                                   algorithm_instance,  # 算法实例
                                                   CV_method,           # 交叉验证方法
                                                   algorithm_category,  # 算法类型
                                                   algorithm_name,      # 算法名称
                                                   test_size,           # 测试集大小
                                                   shuffle,             # 划分训练/测试集前是否打乱数据
                                                   random_state,        # 随机种子(42或者None)
                                                   data_scaling,        # 数据归一化方法
                                                   n_jobs,              # CPU核心数
                                                   save_model,
                                                   n_splits,
                                                   leave_out=1
                                                   )
            self.ml_thread.ml_results.connect(self.process_ml_results)
            self.ml_thread.start()
        
        else:
            pass


    def create_ml_thread(self):
        # 建立ML子线程
        self.ml_thread = MachineLearningThread(self.data)
        self.ml_thread.ml_results.connect(self.process_ml_results)
        self.ml_thread.start()

    def process_ml_results(self, ml_results):
        data_pred = ml_results[0]
        scores = ml_results[1]

        # 调用show_data_in_table函数显示预测数据和预测得分
        for data, tableView_name in zip([data_pred, scores], ['tableView_3', 'tableView_4']):
            self.show_data_in_table(data, tableView_name)
            
        # 调用show_metrics函数显示平均预测指标
        if scores.shape[1] < 9:
            metrics_mean = scores.iloc[-1,:]
            metrics_mean = pd.concat([metrics_mean.iloc[:3], pd.Series(['']*3), metrics_mean.iloc[3:]]).astype(str)
            self.show_metrics(metrics_mean)
        else:
            metrics_mean = scores.iloc[-1,1:].astype(str)
            self.show_metrics(metrics_mean)

        self.show_message('模型训练成功,请查看预测结果！')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = MyMainWindow()
    MainWindow.show()
    sys.exit(app.exec_())