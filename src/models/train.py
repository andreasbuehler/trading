import os
import pickle

from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier, DecisionTreeClassifier, GradientBoostingClassifier
from sklearn.metrics import average_precision_score

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.features.build_features import DataConfig


class ModelTrainer():
    def __init__(self):
        pass


    def loadData(self, dataPath, shuffle=False, numDataPoints=None):

        if os.path.isfile(dataPath):
            print("Loading data from file: ", dataPath)
            with open(dataPath, 'rb') as f:
                data = pickle.load(f)
        else:
            print("Could not load data from: ", dataPath)

        if numDataPoints:
            data = data.tail(numDataPoints)

        X = data.drop(columns=['gt'])
        y = data['gt']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=shuffle)

        return X_train, y_train, X_test, y_test


    def trainModels(self, dataPath):
        X_train, y_train, X_test, y_test = self.loadData(dataPath, shuffle=False, numDataPoints=1000)

        names = [
            # "Nearest Neighbors",
            "Linear SVM",
            "RBF SVM",
            # "Gaussian Process",
            "Decision Tree",
            "Random Forest",
            "Neural Net",
            "AdaBoost",
            "Naive Bayes",
            "QDA",
        ]

        classifiers = [
            # KNeighborsClassifier(3),
            SVC(kernel="linear", C=0.025),
            SVC(gamma=2, C=1),
            # GaussianProcessClassifier(1.0 * RBF(1.0)),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            MLPClassifier(alpha=1, max_iter=1000),
            AdaBoostClassifier(),
            GaussianNB(),
            QuadraticDiscriminantAnalysis(),
        ]

        report = {}

        for name, clf in zip(names, classifiers):
            print("Training: ", name)
            clf = make_pipeline(StandardScaler(), clf)

            clf.fit(X_train, y_train)

            y_pred_train = clf.predict(X_train)
            y_pred_test = clf.predict(X_test)

            train_score = average_precision_score(y_train, y_pred_train)
            test_score = average_precision_score(y_test, y_pred_test)

            report[name] = {'train_score': train_score, 'test_score': test_score}
            print(report[name])

        # clf = RandomForestClassifier(class_weight='balanced', max_depth=20, n_estimators=50, random_state=42, n_jobs=-1)
        # clf.fit(X_train, y_train)

        # y_pred_train = clf.predict(X_train)
        # y_pred_test = clf.predict(X_test)

        # train_score = average_precision_score(y_train, y_pred_train)
        # test_score = average_precision_score(y_test, y_pred_test)

        return report


if __name__ == "__main__":
    modelTrainer = ModelTrainer()
    dataPath = os.path.join(os.getcwd(), DataConfig.processed_data_object_path)
    report = modelTrainer.trainModels(dataPath)
    print("Results: ", report)