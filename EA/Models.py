from .ModelTemplate import *
from sklearn import ensemble, gaussian_process, naive_bayes, neighbors, tree
import tensorflow as tf
from tensorflow import keras


"""
sklearn.ensemble
"""
class RandomForestClassifier(ClassificationModel):
    def __init__(self, name, params, **kwargs):
        super().__init__(name=name, model=ensemble.RandomForestClassifier, params=params, **kwargs)


class RandomForestClassifierTemplate(ModelTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__('Random Forest', *args, **kwargs)
        self.model = RandomForestClassifier
        self.params_desc = [['n_estimator', 'range', 'int', [3, 300]], ['criterion', 'select', 'string', ["entropy", "gini"]], ['max_depth', 'const', 'int', None], ['min_samples_split', 'range', 'float', [0.6, 0.99]], ['n_jobs', 'const', 'int', -1]]


"""
sklearn.naive_bayes
"""
class BernoulliNB(MultiClassClassificationModel):
    def __init__(self, name, params, **kwargs):
        super().__init__(name=name, model=naive_bayes.BernoulliNB, params=params, **kwargs)


class BernoulliNBTemplate(ModelTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__('BernoulliNB', *args, **kwargs)
        self.model = BernoulliNB
        self.params_desc = [['alpha', 'range', 'float', [0, 1]]]


class ComplementNB(MultiClassClassificationModel):
    def __init__(self, name, params, **kwargs):
        super().__init__(name=name, model=naive_bayes.ComplementNB, params=params, **kwargs)


class ComplementNBTemplate(ModelTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__('ComplementNB', *args, **kwargs)
        self.model = ComplementNB
        self.params_desc = [['alpha', 'range', 'float', [0, 1]], ['norm', 'range', 'bool', [False, True]]]


class CategoricalNB(MultiClassClassificationModel):
    def __init__(self, name, params, **kwargs):
        super().__init__(name=name, model=naive_bayes.CategoricalNB, params=params, **kwargs)


class CategoricalNBTemplate(ModelTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__('CategoricalNB', *args, **kwargs)
        self.mode = CategoricalNB
        self.params_desc = [['alpha', 'range', 'float', [0, 1]]]


class GaussianNB(MultiClassClassificationModel):
    def __init__(self, name, params, **kwargs):
        super().__init__(name=name, model=naive_bayes.GaussianNB, params=params, **kwargs)


class GaussianNBTemplate(ModelTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__('GaussianNB', *args, **kwargs)
        self.model = GaussianNB
        self.params_desc = [['var_smoothing', 'range', 'float', [1e-12, 1e-3]]]


class MultinomialNB(MultiClassClassificationModel):
    def __init__(self, name, params, **kwargs):
        super().__init__(name=name, model=naive_bayes.MultinomialNB, params=params, **kwargs)


class MultinomialNBTemplate(ModelTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__('MultinomialNB', *args, **kwargs)
        self.model = MultinomialNB
        self.params_desc = [['alpha', 'range', 'float', [0, 1]]]


"""
sklearn.neighbors
"""
class KNeighborsClassifier(ClassificationModel):
    def __init__(self, name, params, **kwargs):
        super().__init__(name=name, model=neighbors.KNeighborsClassifier, params=params, **kwargs)


class KNeighborsClassifierTemplate(ModelTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__('KNeighborsClassifier', *args, **kwargs)
        self.model = KNeighborsClassifier
        self.params_desc = [["n_neighbors", "range", "int", [1, 15]], ["weights", "const", "string", "uniform"], ["leaf_size", "range", "int", [15, 60]], ["p", "const", "int", 2], ["n_jobs", "const", "int", -1]]


class RadiusNeighborsClassifier(ClassificationModel):
    def __init__(self, name, params, **kwargs):
        super().__init__(name=name, model=neighbors.RadiusNeighborsClassifier, params=params, **kwargs)


class RadiusNeighborsClassifierTemplate(ModelTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__("RadiusNeighborsClassifier", *args, **kwargs)
        self.model = RadiusNeighborsClassifier
        self.params_desc = [["radius", "range", "float", [1.0, 30.0]], ["weights", "const", "string", "uniform"], ["leaf_size", "range", "int", [15, 60]], ["p", "const", "int", 2], ["n_jobs", "const", "int", -1]]


"""
sklearn.tree
"""
class DecisionTreeClassifier(ClassificationModel):
    def __init__(self, name, params, **kwargs):
        super().__init__(name=name, model=tree.DecisionTreeClassifier, params=params, **kwargs)


class DecisionTreeClassifierTemplate(ModelTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__("DecisionTreeClassifier", *args, **kwargs)
        self.model = DecisionTreeClassifier
        self.params_desc = [["criterion", "select", "string", ["gini", "entropy"]], ["splitter", "select", "string", ["best", "random"]], ["min_impurity_decrease", "range", "float", [0.0, 0.3]], ["ccp_alpha", "range", "float", [0.0, 0.3]]]


class ExtraTreeClassifier(ClassificationModel):
    def __init__(self, name, params, **kwargs):
        super().__init__(name=name, model=tree.ExtraTreeClassifier, params=params, **kwargs)


class ExtraTreeClassifierTemplate(ModelTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__("ExtraTreeClassifier", *args, **kwargs)
        self.model = ExtraTreeClassifier
        self.params_desc = [["criterion", "select", "string", ["gini", "entropy"]], ["splitter", "const", "string", "random"], ["min_impurity_decrease", "range", "float", [0.0, 0.3]], ["ccp_alpha", "range", "float", [0.0, 0.3]]]


"""
neural networks
"""
class NNClassifier(ClassificationModel):
    def __init__(self, name, params, **kwargs):
        super().__init__(name=name, model=None, params=params, **kwargs)
        #print(params)
        self.weights = params['weights'] if 'weights' in params else False
        self.model = keras.Sequential()
        self.model.add(keras.Input(shape=params['input']))
        try:
            for i in range(params['layers']):
                layer_params = params["layer_{}".format(i)]
                if layer_params["type"] == "dense":
                    self.model.add(keras.layers.Dense(layer_params["units"], activation=layer_params["activation"]))
                elif layer_params["type"] == "conv2d":
                    self.model.add(keras.layers.Conv2D(filters=layer_params["units"], kernel_size=layer_params["kernel"], strides=layer_params["strides"], padding=layer_params["padding"], activation=layer_params["activation"]))
                    self.model.add(keras.MaxPool2D(pool_size=layer_params["poolsize"], strides=layer_params["poolstride"], padding=layer_params["poolpadding"]))
                elif layer_params["type"] == "conv1d":
                    self.model.add(keras.layers.Conv1D(filters=layer_params["units"], kernel_size=layer_params["kernel"], strides=layer_params["strides"], padding=layer_params["padding"], activation=layer_params["activation"]))
                    self.model.add(keras.MaxPool1D(pool_size=layer_params["poolsize"], strides=layer_params["poolstride"], padding=layer_params["poolpadding"]))
                elif layer_params["type"] == "lstm":
                    self.model.add(keras.layers.LSTM(layer_params["units"], activation=layer_params["activation"], recurrent_activation=layer_params["recurrent_activation"]))
                elif layer_params["type"] == "gru":
                    self.model.add(keras.layers.GRU(layer_params["units"], activation=layer_params["activation"], recurrent_activation=layer_params["recurrent_activation"]))
                elif layer_params["type"] == "dropout":
                    self.model.add(keras.layers.Dropout(rate=layer_params["rate"]))
        except Exception as e:
            print(e)
            print(params)
            exit()
        self.model.add(keras.layers.Dense(params["output"]))

        optimizer = None
        if params['optimizer'] == "adam":
            optimizer = keras.optimizers.Adam
        elif params['optimizer'] == "adamax":
            optimizer = keras.optimizers.Adamax
        elif params['optimizer'] == "adagrad":
            optimizer = keras.optimizers.Adagrad
        elif params['optimizer'] == "adadelta":
            optimizer = keras.optimizers.Adadelta
        elif params['optimizer'] == "ftrl":
            optimizer = keras.optimizers.Ftrl
        elif params['optimizer'] == "nadam":
            optimizer = keras.optimizers.Nadam
        elif params['optimizer'] == 'rmsprop':
            optimizer = keras.optimizers.RMSprop
        elif params['optimizer'] == 'sgd':
            optimizer = keras.optimizers.SGD

        optimizer = optimizer(**params["optimizer_params"])

        self.model.compile(optimizer=optimizer, loss=params['loss'], metrics=params['metrics'])


    def train(self, *args, **kwargs):
        return super().train(*args, weight=self.weights, **kwargs)


    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, batch_size=self.params['batch_size'], epochs=self.params['epochs'], verbose=0, **kwargs)


    def save(self, filepath):
        self.model.save(filepath)


    def load(self, filepath):
        self.model = keras.models.load_model(filepath)


class BaseNNClassifierTemplate(ModelTemplate):
    def __init__(self, input_dim = 3, output_dim = 2):
        super().__init__("neural network", input_dim, output_dim)
        self.model = NNClassifier
        self.params_desc = [["layers", "range", "int", [1, 10]],
            ["input", "const", "int", input_dim],
            ["output", "const", "int", output_dim],
            ["optimizer", "select", "string", ["adam", "adamax", "ftrl", "nadam", "adagrad", "adadelta", "rmsprop", "sgd"]],
            ["loss", "select", "string", ["binary_crossentropy", "categorical_crossentropy", "cosine_similarity", "huber", "hinge", "mean_squared_error", "mean_squared_logarithmic_error", "poisson", "sparse_categorical_crossentropy"]],
            ["metrics", "const", "list", ["accuracy"]],
            ["batch_size", "const", "int", 64],
            ["epochs", "range", "int", [50, 500]]]
        self.layer_type = ["type", "select", "string", ["dense", "conv1d", "conv2d", "lstm", "gru", "dropout"]]
        self.layer_params = {"general": [["units", "range", "int", [4, 800]], ["activation", "select", "string", ["relu", "sigmoid", "selu", "exponential", "softmax", "softplus", "tanh", "linear", None]]], "conv": [["kernel", "range", "int", [2, 10]], ["strides", "range", "int", [1, 4]], ["padding", "select", "string", ["same", "valid"]], ["poolsize", "range", "int", [2, 9]], ["poolstride", "const", "none", None]], "rnn": [["recurrent_activation", "select", "string", ["tanh", "relu", "sigmoid"]]], "dropout": [["rate", "range", "float", [1e-6, 0.5]]]}
        self.optimizer_params = {"learning_rate": ["learning_rate", "range", "float", [1e-5, 1e-3]], "rho": ["rho", "range", "float", [0.8, 1.0]], "epsilon": ["epsilon", "range", "float", [1e-9, 1e-7]]}

        self.optimizer_param_list = {"adadelta": ["learning_rate", "rho", "epsilon"], "adagrad": ["learning_rate", "epsilon"], "adam": ["learning_rate", "epsilon"], "adamax": ["learning_rate", "epsilon"], "ftrl": ["learning_rate"], "nadam": ["learning_rate", "epsilon"], "rmsprop": ["learning_rate", "rho", "epsilon"], "sgd": ["learning_rate"]}


    def __eq__(self, other):
        return super().__eq__(other) and self.layer_params == other.layer_params


    def __str__(self):
        objstr = super().__str__()
        objstr += "layer types: {}\n".format(self.layer_type[3])
        for j in self.layer_params:
            objstr += "{} layer parameters\n".format(j)
            for i in self.layer_params[j]:
                objstr += "param name: {}   mode: {}   type: {}   mod param: {}\n".format(i[0], i[1], i[2], i[3])
        return objstr


    def rename(self, name):
        self.name = name


    def set_dim(self, input_dim=None, output_dim=None):
        if input_dim != None:
            self.params_desc[1][3] = input_dim
            self.params_desc[2][3] = output_dim


class EnhancedNNClassifierTemplate(BaseNNClassifierTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_params["general"][1][3] = ["relu", "sigmoid", "softmax", "tanh"]
        self.layer_params["conv"][0][3][1] = 8
        self.layer_params["conv"][2] = ["padding", "const", "string", "same"]

        self.params_desc[3] = ["optimizer", "select", "string", ["adam", "adamax", "adagrad", "rmsprop"]]
        self.params_desc[4] = ["loss", "const", "string", "categorical_crossentropy"]
        self.params_desc[7] = ["epochs", "const", "int", 300]


class DNNClassifierTemplate(EnhancedNNClassifierTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "deep neural network"
        self.layer_type[3] = ["dense", "dropout"]


class WeightedBinaryDNNClassifierTemplate(DNNClassifierTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "wbd neural network"
        self.params_desc.append(['weights', 'const', 'bool', True])


class CNNClassifierTemplate(DNNClassifierTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "convolutional neural network"
        self.layer_type[3].append("conv2d")


class RNNClassifierTemplate(DNNClassifierTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "recurrent neural network"
        self.layer_type[3].append("lstm")
