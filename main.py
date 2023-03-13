import sys
import io
import os
import time
import pprint
from PIL import Image
import pandas as pd
import numpy as np
from prettytable import PrettyTable
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QGraphicsEllipseItem
from Ui_ML_GUI import Ui_MainWindow
from PyQt5.QtGui import QStandardItemModel,QStandardItem

from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, LeaveOneOut, cross_validate, cross_val_score
import matplotlib.pyplot as plt
import joblib

from PyQt5.QtWidgets import QDialog, QGraphicsScene, QGraphicsView, QTreeWidget, QTreeWidgetItem
from PyQt5.QtGui import QBrush, QColor, QPainter, QPen
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


class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MyMainWindow, self).__init__()
        self.setupUi(self)
        self.pushButton_2.setEnabled(False)
        self.pushButton_3.setEnabled(False)
        # 禁用交叉验证的所有单选框
        for button in self.buttonGroup.buttons():
            button.setEnabled(False)
        self.spinBox.setEnabled(False)
        self.spinBox_2.setEnabled(False)

        self.treeWidget.itemClicked.connect(self.select_algorithm)
        self.pushButton.clicked.connect(self.import_data)
        self.pushButton_2.clicked.connect(self.run_machine_learning)
        self.pushButton_3.clicked.connect(self.convertPlainTextToDict)
        
        # self.fig = Figure()
        # self.canvas = FigureCanvas(self.fig)
        # self.graphicsView.layout().addWidget(self.canvas)

    def import_data(self):
        # 导入数据
        file, _ = QFileDialog.getOpenFileName(self, 'Open file', '', 'Excel files (*.xlsx *.xls)')
        if file:
            # 读取数据
            data = pd.read_excel(file, index_col=0,header=0)
            # 划分为features和targets
            self.X, self.y = data.iloc[:, :-1], data.iloc[:, -1]

                       
            # 预览数据
            print("Data imported successfully!")
            model = QStandardItemModel(len(data), len(data.columns))
            model.setHorizontalHeaderLabels(list(data.columns))
            for i in range(len(data)):
                for j in range(len(data.columns)):
                    item = QStandardItem(str(data.iloc[i, j]))
                    model.setItem(i, j, item)
            self.tableView.setModel(model)
            # 显示文件名
            self.lineEdit.setText(file)
            # self.line_edit.show()
            # 弹窗提示
            self.show_message('数据导入成功,请选择算法！')
            

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
        print(data)  

    def show_message(self, message):
        # 弹出信息提示窗
        QMessageBox.information(self, '提示', message)

    def printSelectedItem(self, item, column):
        print(item.text(column))

    def select_algorithm(self, item, column):
        # check if the clicked item is a leaf node (i.e. no child items)
        if item.childCount() == 0:
            # 当前选中算法的分类
            parent_item = item.parent()
            category_name = parent_item.text(column)
            if category_name == "回归":
                self.radioButton.setEnabled(True)
                self.radioButton_2.setEnabled(True)
                self.radioButton_3.setEnabled(True)
                self.spinBox.setEnabled(True)
            elif category_name == "分类":
                self.radioButton.setEnabled(True)
                self.radioButton_2.setEnabled(True)
                self.radioButton_3.setEnabled(True)
                self.radioButton_4.setEnabled(True)
                self.spinBox.setEnabled(True)
                self.spinBox_2.setEnabled(True)
            print(category_name)
            # 当前选中算法的名称
            regressor_name = item.text(column)
            print(item.text(column))

            # 所有定义的回归算法
            if regressor_name == '线性回归':
                model = LinearRegression()
            elif regressor_name == '岭回归':
                model = Ridge()
            elif regressor_name == 'LASSO回归':
                model = Lasso()
            elif regressor_name == 'ElasticNet回归':
                model = ElasticNet()
            elif regressor_name == '多项式回归':
                model = PolynomialFeatures()
            elif regressor_name == '支持向量机回归':
                model = SVR()
            elif regressor_name == '决策树回归':
                model = DecisionTreeRegressor()
            elif regressor_name == '随机森林回归':
                model = RandomForestRegressor()
            elif regressor_name == '梯度提升回归':
                model = GradientBoostingRegressor()
            elif regressor_name == 'ExtraTree回归':
                model = ExtraTreesRegressor()
            elif regressor_name == 'AdaBoost回归':
                model = AdaBoostRegressor()
            elif regressor_name == 'XGBoost回归':
                model = XGBRegressor()
            elif regressor_name == 'LightGBM回归':
                model = LGBMRegressor()
            elif regressor_name == 'CatBoost回归':
                model = CatBoostRegressor()
            elif regressor_name == 'Bagging回归':
                model = BaggingRegressor()
            else:
                pass
            
            # 所有定义的分类算法
            if regressor_name == '逻辑回归':
                model = LogisticRegression()
            elif regressor_name == 'K近邻分类':
                model = KNeighborsClassifier()
            elif regressor_name == '决策树分类':
                model = DecisionTreeClassifier()
            elif regressor_name == '随机森林分类':
                model = RandomForestClassifier()
            elif regressor_name == 'AdaBoost分类':
                model = AdaBoostClassifier()
            elif regressor_name == 'XGBoost分类':
                model = xgb.XGBClassifier()
            elif regressor_name == 'CatBoost分类':
                model = cb.CatBoostClassifier()
            elif regressor_name == 'Bagging分类':
                model = BaggingClassifier(base_estimator=DecisionTreeClassifier())
            elif regressor_name == '支持向量分类':
                model = SVC()
            elif regressor_name == '多层感知器分类':
                model = MLPClassifier()
            elif regressor_name == '高斯朴素贝叶斯分类':
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

    def data_preprocessing(self, data):
        # 数据归一化处理,将归一化之后的X转换为DataFrame格式,加上index和columns
        if self.checkBox_4.isChecked():
            print('StandardScaler()')
            X_scalered = StandardScaler().fit_transform(data)
            data = pd.DataFrame(X_scalered, 
                                index=data.index, 
                                columns=data.columns)
        elif self.checkBox_5.isChecked():
            print('MinMaxScaler()')
            X_scalered = MinMaxScaler().fit_transform(data)
            data = pd.DataFrame(X_scalered, 
                                index=data.index, 
                                columns=data.columns)
        elif self.checkBox_6.isChecked():
            print('分类暂时不可用！')
            pass
        else:
            print('没有数据归一化！')
            pass

        return data

    def save_model(self, num, model, algorithm_name):
        # 保存模型
        # 判断文件夹是否存在,如果不存在，则创建文件夹
        folder_path = 'model'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        localtime = time.localtime()
        time_ = time.strftime('%Y%m%d_%H%M%S', localtime)
        filename = f'{folder_path}/{num}_{time_}_{str(algorithm_name)[:-2]}.kpi'  # 定义文件名
        joblib.dump(model, filename)
       
    def cross_validation(self, X, y, algorithm, CV_method, **kwargs):
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

        # 创建一个模型实例
        model = algorithm

        # 数据划分
        test_size = round(self.doubleSpinBox.value(), 2)
        shuffle = self.checkBox.isChecked()
        random_state = 42 if self.checkBox_2.isChecked() else None
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                                shuffle=shuffle, 
                                                                random_state=random_state)

        if CV_method == 'simple':
            # 简单交叉验证

            # 算法类别
            algorithm_category = kwargs.get('algorithm_category')
            
            # 模型训练
            model.fit(X_train, y_train)

            # 保存模型
            if self.checkBox_3.isChecked():
                self.save_model(0, model, algorithm)

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

                scores = {'train_r2':train_r2,
                          'train_mae':train_mae,
                          'train_rmse':train_rmse,
                          'test_r2':test_r2,
                          'test_mae':test_mae,
                          'test_rmse':test_rmse}

        elif CV_method == 'k_fold':
            # k 折交叉验证
            # if k is None:
            #     k = 5
            # n_splits = kwargs.get('n_splits')
            # shuffle = kwargs.get('shuffle')
            # random_state = kwargs.get('random_state')
            n_splits = kwargs.get('n_splits')
            algorithm_category = kwargs.get('algorithm_category')
            
            if algorithm_category == '回归':
                # 指定评分函数列表
                scoring = {'r2': make_scorer(r2_score),
                        'mae': make_scorer(mean_absolute_error),
                        'rmse': make_scorer(mean_squared_error, squared=False)}
                
                cv_results = cross_validate(model, X_train, y_train, cv=n_splits, scoring=scoring, 
                                            return_train_score=True, return_estimator=True)

                # 遍历每个拆分的估计器列表,在独立测试集上预测并保存模型
                test_r2 = []
                test_mae = []
                test_rmse = []

                for i, estimator in enumerate(cv_results['estimator']):
                    y_test_pred = estimator.predict(X_test)
                    
                    test_r2_ = r2_score(y_test, y_test_pred)
                    test_mae_ = mean_absolute_error(y_test, y_test_pred)
                    test_rmse_ = mean_squared_error(y_test, y_test_pred, squared=False)
                    
                    test_r2.append(test_r2_)
                    test_mae.append(test_mae_)
                    test_rmse.append(test_rmse_)

                    # 保存模型
                    if self.checkBox_3.isChecked():
                        self.save_model(i, model, algorithm)

                scores = {'train_r2':cv_results['train_r2'],
                          'train_mae':cv_results['train_mae'],
                          'train_rmse':cv_results['train_rmse'],
                          'val_r2':cv_results['test_r2'],
                          'val_mae':cv_results['test_mae'],
                          'val_rmse':cv_results['test_rmse'],
                          'test_r2':np.array(test_r2),
                          'test_mae':np.array(test_mae),
                          'test_rmse':np.array(test_rmse)}

            # scores = cross_val_score(model, X, y, cv=n_splits, scoring='r2')
            
            
            # kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
            # scores = []
            # for train_index, test_index in kf.split(X):
            #     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            #     y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            #     model.fit(X_train, y_train)
            #     y_pred = model.predict(X_test)
            #     score = r2_score(y_test, y_pred)
            #     scores.append(score)

        elif CV_method == 'leave_one_out':
            # 留一法交叉验证

            # 留一法每次只有一个测试样本，少于2个样本将无法计算test_R2，可用以下方法将报出的警告信息忽略
            import warnings
            from sklearn.exceptions import UndefinedMetricWarning
            # 忽略UndefinedMetricWarning警告
            warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
            # 如果希望再次打印出已经被忽略的警告信息，可以使用以下代码将警告设置为默认模式：
            # warnings.filterwarnings('default', category=UndefinedMetricWarning)

            loo = LeaveOneOut()

            algorithm_category = kwargs.get('algorithm_category')

            if algorithm_category == '回归':
                # 指定评分函数列表
                scoring = {'r2': make_scorer(r2_score),
                        'mae': make_scorer(mean_absolute_error),
                        'rmse': make_scorer(mean_squared_error, squared=False)}
                # 留一法交叉验证
                cv_results = cross_validate(model, X_train, y_train, cv=loo, scoring=scoring, 
                                            return_train_score=True, return_estimator=True)
                
                # 得到每个样本的预测值以计算测试集R2
                estimators = cv_results['estimator']
                y_val_pred = np.zeros(y_train.shape)

                # 遍历每个拆分的评估器列表,在独立测试集上预测并保存模型
                test_r2 = []
                test_mae = []
                test_rmse = []

                for i, estimator in enumerate(estimators):
                    # 拿出训练该评估器时用到的的唯一一个训练样本
                    x_val = pd.DataFrame(X_train.iloc[i]).T
                    # x_val = np.array(X_train.iloc[i]).reshape(1, -1)
                    y_val_pred_ = estimator.predict(x_val)
                    # 将预测结果保存下来，即验证集预测结果
                    y_val_pred[i] = y_val_pred_

                    # 每个LOO模型对独立测试集的预测结果
                    y_test_pred = estimator.predict(X_test)
                    
                    test_r2_ = r2_score(y_test, y_test_pred)
                    test_mae_ = mean_absolute_error(y_test, y_test_pred)
                    test_rmse_ = mean_squared_error(y_test, y_test_pred, squared=False)
                    
                    test_r2.append(test_r2_)
                    test_mae.append(test_mae_)
                    test_rmse.append(test_rmse_)

                    # 保存模型
                    if self.checkBox_3.isChecked():
                        self.save_model(i, model, algorithm)
                
                # 验证集R2
                val_r2 = r2_score(y_train, y_val_pred)

                # 预测评估指标                
                scores = {'train_r2':cv_results['train_r2'],
                          'train_mae':cv_results['train_mae'],
                          'train_rmse':cv_results['train_rmse'],
                          'val_r2':np.array(val_r2), # 只有一个值
                          'val_mae':cv_results['test_mae'],
                          'val_rmse':cv_results['test_rmse'],
                          'test_r2':np.array(test_r2),
                          'test_mae':np.array(test_mae),
                          'test_rmse':np.array(test_rmse)}
            # scores = cross_val_score(model, X, y, cv=loo, scoring='r2')
            
            # scores = []
            # for train_index, test_index in loo.split(X):
            #     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            #     y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            #     model.fit(X_train, y_train)
            #     y_pred = model.predict(X_test)
            #     score = r2_score(y_test, y_pred)
            #     scores.append(score)

        elif CV_method == 'stratified':
            # 分层交叉验证
            n_splits = kwargs.get('n_splits')
            shuffle = kwargs.get('shuffle')
            random_state = kwargs.get('random_state')
            skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
            # 指定评分函数列表
            scoring = {'r2': make_scorer(r2_score),
                    'mae': make_scorer(mean_absolute_error),
                    'rmse': make_scorer(mean_squared_error, squared=False)}
            
            cv_results = cross_validate(model, X_train, y_train, cv=skf, scoring=scoring, 
                                        return_train_score=True, return_estimator=True)

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

        return (scores ,test_size, shuffle, random_state)
    
    def run_machine_learning(self):
        # print(self.X.shape, self.y.shape)
        # 获取所选算法实例并进行机器学习
        # selected_items = self.treeWidget.selectedItems()
        
        # 获取选择的算法及其分类(回归or分类)
        currentItem = self.treeWidget.currentItem()
        parentItem = currentItem.parent()
        if parentItem is not None:
            algorithmName = currentItem.text(0)
            categoryName = parentItem.text(0)
            algorithm = self.select_algorithm(currentItem, 0)
            
            print(f"您选择了{categoryName}中的{algorithmName}算法,实例为{algorithm}")
        else:
            print("请选择一个算法")

        # 数据预处理
        # self.X = self.data_preprocessing(self.X)    

        # algorithm_classification = selected_items[0].parent()
        # print(algorithm_classification)
        # print(selected_items)
        # if selected_items:
        #     selected_item = selected_items[0]
        #     algorithm = self.select_algorithm(selected_item, 0)
        #     # 进行机器学习，这里仅仅是演示，没有实际训练和预测的过程
        #     print(f"Running machine learning with algorithm: {algorithm}")
        # else:
        #     print("Please select an algorithm from the list.")
        
        # 交叉验证方法选择
        if self.radioButton.isChecked():
            # 简单交叉验证
            CV_method ='simple'
            # test_size = self.doubleSpinBox.value()
            # print(self.X.shape,
            #       self.y.shape,
            #       algorithm,
            #       CV_method,
            #       test_size)
            scores = self.cross_validation(self.X, 
                                          self.y, 
                                          algorithm, 
                                          CV_method, 
                                          algorithm_category=categoryName)
            # print('简单交叉验证，测试集大小：',round(test_size, 2))
            scores = scores[0]
            for key, value in scores.items():
                scores[key] = round(value, 2)
            pprint.pprint(f"预测scores为: {scores}")

        elif self.radioButton_2.isChecked():
            # 留一法交叉验证
            CV_method ='leave_one_out'
            scores = self.cross_validation(self.X, 
                                          self.y, 
                                          algorithm, 
                                          CV_method,
                                          algorithm_category=categoryName)
            print('留一法')
            pprint.pprint(f"预测scores为: {scores}")
        
        elif self.radioButton_3.isChecked():
            # K折交叉验证
            CV_method ='k_fold'
            n_splits = self.spinBox.value()
            scores = self.cross_validation(self.X, 
                                          self.y, 
                                          algorithm, 
                                          CV_method, 
                                          n_splits=n_splits,
                                          algorithm_category=categoryName 
                                          )
            print('k折交叉验证：',self.spinBox.value())
            pprint.pprint(f"预测scores为: {scores}")
        
        elif self.radioButton_4.isChecked():
            # 分层交叉验证
            CV_method ='stratified'
            n_splits = self.spinBox_2.value()
            scores = self.cross_validation(self.X, 
                                          self.y, 
                                          algorithm, 
                                          CV_method, 
                                          n_splits=n_splits, 
                                          algorithm_category=categoryName
                                          )
            print('分层交叉验证',self.spinBox_2.value())
            print(f"预测scores为: {scores}")
        else:
            pass

        '''
        # 训练模型
        algorithm.fit(self.X_train, self.y_train)

        # 保存模型
        localtime = time.localtime()
        time1 = time.strftime('%Y%m%d_%H%M%S', localtime)
        joblib.dump(algorithm, 'Model/%s_%s.pkl'%(str(algorithm)[:-2], time1))
        
        # 在训练集上进行预测
        y_train_pred = algorithm.predict(self.X_train)
        r2_train = r2_score(self.y_train, y_train_pred)
        mae_train = mean_absolute_error(self.y_train, y_train_pred)
        rmse_train = np.sqrt(mean_squared_error(self.y_train, y_train_pred))

        # 在测试集上进行预测
        y_test_pred = algorithm.predict(self.X_test)
        r2_test = r2_score(self.y_test, y_test_pred)
        mae_test = mean_absolute_error(self.y_test, y_test_pred)
        rmse_test = np.sqrt(mean_squared_error(self.y_test, y_test_pred))

        # 显示预测得分
        self.lineEdit_7.setText(str(round(r2_train, 2)))
        self.lineEdit_8.setText(str(round(mae_train, 2)))
        self.lineEdit_9.setText(str(round(rmse_train, 2)))
        self.lineEdit_10.setText(str(round(r2_test, 2)))
        self.lineEdit_11.setText(str(round(mae_test, 2)))
        self.lineEdit_12.setText(str(round(rmse_test, 2)))


        # 打印指标
        table = PrettyTable(['Metric', 'Training set', 'Test set'])
        table.add_row(['R2', round(r2_train, 2), round(r2_test, 2)])
        table.add_row(['MAE', round(mae_train, 2), round(mae_test, 2)])
        table.add_row(['RMSE', round(rmse_train, 2), round(rmse_test, 2)])
        print(table)
        
        # 显示预测结果
        data = {'TrainSet':self.y_train.index,
                'y_train':self.y_train.values,
                'y_train_pred':y_train_pred,
                'TestSet':self.y_test.index,
                'y_test':self.y_test.values,
                'y_test_pred':y_test_pred}
    
        data = pd.DataFrame({k: pd.Series(v) for k, v in data.items()})

        model = QStandardItemModel(len(data), len(data.columns))
        model.setHorizontalHeaderLabels(list(data.columns))
        for i in range(len(data)):
            for j in range(len(data.columns)):
                item = QStandardItem(str(data.iloc[i, j]))
                model.setItem(i, j, item)
        self.tableView_3.setModel(model)

        '''
        self.show_message('模型训练成功,请查看预测结果！')
        

    def machineLearning(self):
        # 获取选择的算法名称
        algorithm = self.select_algorithm()
        print(algorithm)
        currentItem = self.treeWidget.currentItem()
        parentItem = currentItem.parent()
        if parentItem is not None:
            algorithmName = currentItem.text(0)
            categoryName = parentItem.text(0)
            print(f"您选择了{categoryName}中的{algorithmName}算法")
        else:
            print("请选择一个算法")

        # 训练模型并打印预测结果
        if algorithmName == "随机森林回归":
            rf_params = RandomForestRegressor().get_params()
            print(rf_params)
            text = ""
            for key, value in rf_params.items():
                text += f"{key}: {value}\n"
            self.plainTextEdit.setPlainText(text)
            # model = LinearRegression(**self.model_params)
            print("随机森林回归")
            # 构建随机森林回归模型并进行预测
            # X = self.data.iloc[:, :-1].values
            # y = self.data.iloc[:, -1].values
            # regressor = RandomForestRegressor(n_estimators=100, random_state=0)
            # regressor.fit(X, y)
            # y_pred = regressor.predict(X)
            # print(f"随机森林回归预测结果：\n{y_pred}")
        elif algorithmName == "支持向量回归":
            print("支持向量回归")
            # 构建支持向量回归模型并进行预测
            # X = self.data.iloc[:, :-1].values
            # y = self.data.iloc[:, -1].values
            # regressor = SVR(kernel='rbf')
            # regressor.fit(X, y)
            # y_pred = regressor.predict(X)
            # print(f"支持向量回归预测结果：\n{y_pred}")
        elif algorithmName == "决策树":
            print("决策树")
            # 构建决策树分类器并进行预测
            pass
        elif algorithmName == "支持向量分类":
            print("支持向量分类")
            # 构建支持向量分类器并进行预测
            pass
        else:
            pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = MyMainWindow()
    MainWindow.show()
    sys.exit(app.exec_())