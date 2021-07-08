"""
template class for prediction models
"""
from sklearn import model_selection, metrics
import numpy as np
import pickle


class ModelTemplate:
    def __init__(self, name, input_dim, output_dim):
        self.params_desc = []
        self.name = name
        self.model = None

        self._dim = (input_dim, output_dim)


    def __eq__(self, other):
        return self.name == other.name and self.params_desc == other.params_desc


    def new_model(self, params):
        return self.model(name=self.name, params=params, input_dim=self._dim[0], output_dim=self._dim[1])


    def __str__(self):
        objstr = self.name + "\n" + "Dimension: " + str(self._dim) + "\n" + "Parameters:\n"
        for i in self.params_desc:
            objstr += "param name: {}   mode: {}   type: {}   mod param: {}\n".format(i[0], i[1], i[2], i[3])
        return objstr


class ClassificationModel:
    def __init__(self, name, model, params, model_path=None, **kwargs):
        self.name = name
        self._model_init = model
        self.params = params

        self.model = self._model_init(**params) if model != None else None

        if model_path != None:
            self.load(model_path)


    def _classification_metrics(self, truth, pred):
        res = dict()
        if len(truth.shape) == 2:
            truth = np.argmax(truth, axis=1)
        if len(pred.shape) == 2:
            pred = np.argmax(pred, axis=1)
        else:
            pred = pred.round()

        """
        print(truth)
        print(truth.shape)
        print(pred)
        print(pred.shape)
        """

        res['acc'] = metrics.accuracy_score(truth, pred)
        res['recall'] = metrics.recall_score(truth, pred)
        res['precision'] = metrics.precision_score(truth, pred)
        res['f1'] = metrics.f1_score(truth, pred)
        res['log'] = metrics.log_loss(truth, pred)
        res['auc'] = metrics.roc_auc_score(truth, pred, average='weighted')

        return res


    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)


    def predict(self, X, **kwargs):
        return self.model.predict(X, **kwargs)


    def train(self, X, y, X_test=None, y_test=None, weight=False, **kwargs):
        X_train = X
        y_train = y

        try:
            len(X_test)
            len(y_test)
        except Exception as e:
            X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, shuffle=True)


        weights = dict(map(lambda x: [x[0], len(y_train) / (2*x[1])], enumerate(np.sum(y_train, axis=0)))) # for binary


        # fit model
        if weight==True:
            self.fit(X_train, y_train, class_weight=weights, **kwargs)
        else:
            self.fit(X_train, y_train, **kwargs)

        # calculate for performance
        y_train_pred = self.predict(X_train)
        y_test_pred = self.predict(X_test)

        performance = {'train': self._classification_metrics(y_train, y_train_pred), 'test': self._classification_metrics(y_test, y_test_pred)}

        return performance


    def save(self, filepath):
        with open(filepath, 'wb') as file:
            pickle.dump(self.model, file)


    def load(self, filepath):
        with open(filepath, 'rb') as file:
            self.model = pickle.load(file)


class MultiClassClassificationModel(ClassificationModel):
    def __init__(self, input_dim, output_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = [ self._model_init(**self.params) for i in range(output_dim) ]


    def fit(self, X, y, **kwargs):
        y = y.T
        for ind, i in enumerate(self.model):
            i.fit(X, y[ind], **kwargs)


    def predict(self, X, **kwargs):
        return np.array(list(map(lambda i: i.predict(X, **kwargs), self.model))).T
