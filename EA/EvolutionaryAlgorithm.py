"""
evolutionary algorithm
"""
import numpy as np
from sklearn.model_selection import train_test_split
import gc

class ModelSelector:
    def __init__(self, models, acc=0.95, generations=7, pool_size=20, reproduce_probability=0.95, max_children=5, crossing_probability=0.5, mutate_probability=0.01, max_mutate=2, hybrid_probability=0, no_extinct=True, mutate_after_crossover=True, split_data=0.3, score_weightings={'train':{'auc':1}, 'test':{'auc':1}}):
        self._models = models
        self._gen = generations
        self._acc = acc
        self._ps = pool_size
        self._no_extinct = no_extinct
        self._split_data = split_data

        self._topick = pool_size
        self._pool = []
        self.picked = []

        self.p_r = reproduce_probability
        self.max_children = max_children
        self.p_mutate = mutate_probability
        self.max_mutate = max_mutate
        self.hybrid_p = hybrid_probability
        self.p_cross = crossing_probability

        self.best_acc = 0
        self.best = None

        self._mutate_after_crossover = mutate_after_crossover

        self._score_weightings = score_weightings # expected input: {'train':{...}, 'test':{...}}


    def _new_model(self, model, params=None, acc_passed=0):
        rng = np.random.default_rng()
        if params == None:
            params = dict()
            for i in model.params_desc:
                if i[1] == 'const':
                    params[i[0]] = i[3]
                elif i[1] == "select":
                    params[i[0]] = rng.choice(i[3])
                else:
                    rge = None
                    if i[2] == 'bool':
                        rge = {'low':2}
                    elif i[1] == 'range':
                        rge = {'low':i[3][0], 'high':i[3][1]}
                    elif i[1] == 'threshold':
                        rge = {'low':i[3][1]}

                    if i[2] in ('int', 'bool'):
                        params[i[0]] = rng.integers(**rge)
                        if i[2] == 'bool':
                            params[i[0]] = True if params[i[0]] == 0 else False
                    elif i[2] == 'float':
                        params[i[0]] = rng.uniform(**rge)

                    if i[3][0] == '>':
                        params[i[0]] += i[3][1]

            if "neural network" in model.name:
                front = True
                for i in range(params["layers"]):
                    choose_layer = True
                    layer_params = dict()
                    while choose_layer:
                        layer_params["type"] = rng.choice(model.layer_type[3])
                        choose_layer = "conv" in layer_params["type"] and front
                    front = "conv" in layer_params["type"] and front

                    if layer_params["type"] != "dropout":
                        for j in model.layer_params["general"]:
                            if j[1] == 'const':
                                layer_params[j[0]] = j[3]
                            elif j[1] == "select":
                                layer_params[j[0]] = rng.choice(j[3])
                            else:
                                rge = None
                                if j[2] == 'bool':
                                    rge = {'low':2}
                                elif j[1] == 'range':
                                    rge = {'low':j[3][0], 'high':j[3][1]}
                                elif j[1] == 'threshold':
                                    rge = {'low':j[3][1]}

                                if j[2] in ('int', 'bool'):
                                    layer_params[j[0]] = rng.integers(**rge)
                                    if j[2] == 'bool':
                                        layer_params[j[0]] = True if layer_params[j[0]] == 0 else False
                                elif j[2] == 'float':
                                    layer_params[j[0]] = rng.uniform(**rge)

                                if j[3][0] == '>':
                                    layer_params[j[0]] += j[3][1]

                        if "conv" in layer_params["type"]:
                            for j in model.layer_params["conv"]:
                                if j[1] == 'const':
                                    layer_params[j[0]] = j[3]
                                elif j[1] == "select":
                                    layer_params[j[0]] = rng.choice(j[3])
                                else:
                                    rge = None
                                    if j[2] == 'bool':
                                        rge = {'low':2}
                                    elif j[1] == 'range':
                                        rge = {'low':j[3][0], 'high':j[3][1]}
                                    elif j[1] == 'threshold':
                                        rge = {'low':j[3][1]}

                                    if j[2] in ('int', 'bool'):
                                        layer_params[j[0]] = rng.integers(**rge)
                                        if j[2] == 'bool':
                                            layer_params[j[0]] = True if layer_params[j[0]] == 0 else False
                                    elif j[2] == 'float':
                                        layer_params[j[0]] = rng.uniform(**rge)

                                    if j[3][0] == '>':
                                        layer_params[j[0]] += j[3][1]

                        elif layer_params["type"] in ("lstm", "gru"):
                            for j in model.layer_params["rnn"]:
                                if j[1] == 'const':
                                    layer_params[j[0]] = j[3]
                                elif j[1] == "select":
                                    layer_params[j[0]] = rng.choice(j[3])
                                else:
                                    rge = None
                                    if j[2] == 'bool':
                                        rge = {'low':2}
                                    elif j[1] == 'range':
                                        rge = {'low':j[3][0], 'high':j[3][1]}
                                    elif j[1] == 'threshold':
                                        rge = {'low':j[3][1]}

                                    if j[2] in ('int', 'bool'):
                                        layer_params[j[0]] = rng.integers(**rge)
                                        if j[2] == 'bool':
                                            layer_params[j[0]] = True if layer_params[j[0]] == 0 else False
                                    elif j[2] == 'float':
                                        layer_params[j[0]] = rng.uniform(**rge)

                                    if j[3][0] == '>':
                                        layer_params[j[0]] += j[3][1]

                    else:
                        for j in model.layer_params["dropout"]:
                            if j[1] == 'const':
                                layer_params[j[0]] = j[3]
                            elif j[1] == "select":
                                layer_params[j[0]] = rng.choice(j[3])
                            else:
                                rge = None
                                if j[2] == 'bool':
                                    rge = {'low':2}
                                elif j[1] == 'range':
                                    rge = {'low':j[3][0], 'high':j[3][1]}
                                elif j[1] == 'threshold':
                                    rge = {'low':j[3][1]}

                                if j[2] in ('int', 'bool'):
                                    layer_params[j[0]] = rng.integers(**rge)
                                    if j[2] == 'bool':
                                        layer_params[j[0]] = True if layer_params[j[0]] == 0 else False
                                elif j[2] == 'float':
                                    layer_params[j[0]] = rng.uniform(**rge)

                                if j[3][0] == '>':
                                    layer_params[j[0]] += j[3][1]

                    params["layer_{}".format(i)] = layer_params

                optimizer_params = dict()
                for j in model.optimizer_param_list[params["optimizer"]]:
                    i = model.optimizer_params[j]
                    if i[1] == 'const':
                        optimizer_params[i[0]] = i[3]
                    elif i[1] == "select":
                        optimizer_params[i[0]] = rng.choice(i[3])
                    else:
                        rge = None
                        if i[2] == 'bool':
                            rge = {'low':2}
                        elif i[1] == 'range':
                            rge = {'low':i[3][0], 'high':i[3][1]}
                        elif i[1] == 'threshold':
                            rge = {'low':i[3][1]}

                        if i[2] in ('int', 'bool'):
                            optimizer_params[i[0]] = rng.integers(**rge)
                            if i[2] == 'bool':
                                optimizer_params[i[0]] = True if params[i[0]] == 0 else False
                        elif i[2] == 'float':
                            optimizer_params[i[0]] = rng.uniform(**rge)

                        if i[3][0] == '>':
                            optimizer_params[i[0]] += i[3][1]
                params["optimizer_params"] = optimizer_params

        return [model.new_model(params), model, params, acc_passed] # model, template, parameter settings, performance


    def _mutate(self, model):
        rng = np.random.default_rng()
        params = model[2].copy()
        # mutate by random mize some param
        for i in model[1].params_desc:
            if i[1] != 'const' and rng.random() < self.p_mutate:
                if i[1] == "select":
                    params[i[0]] = rng.choice(i[3])
                else:
                    rge = None
                    if i[2] == 'bool':
                        rge = {'low':2}
                    elif i[1] == 'range':
                        rge = {'low':i[3][0], 'high':i[3][1]}
                    elif i[1] == 'threshold':
                        rge = {'low':i[3][1]}

                    if i[2] in ('int', 'bool'):
                        params[i[0]] = rng.integers(**rge)
                        if i[2] == 'bool':
                            params[i[0]] = True if params[i[0]] == 0 else False
                    elif i[2] == 'float':
                        params[i[0]] = rng.uniform(**rge)

                    if i[3][0] == '>':
                        params[i[0]] += i[3][1]

        if "neural network" in model[1].name:
            front = True
            for i in range(params["layers"]):
                layer_params = params["layer_{}".format(i)] if "layer_{}".format(i) in params else dict()
                if not "type" in layer_params or rng.random() < self.p_mutate:
                    choose_layer = True
                    while choose_layer:
                        layer_params["type"] = rng.choice(model[1].layer_type[3])
                        choose_layer = "conv" in layer_params["type"] and front

                    if layer_params["type"] != "dropout":
                        for j in model[1].layer_params["general"]:
                            if j[1] == 'const':
                                layer_params[j[0]] = j[3]
                            elif j[1] == "select":
                                layer_params[j[0]] = rng.choice(j[3])
                            else:
                                rge = None
                                if j[2] == 'bool':
                                    rge = {'low':2}
                                elif j[1] == 'range':
                                    rge = {'low':j[3][0], 'high':j[3][1]}
                                elif j[1] == 'threshold':
                                    rge = {'low':j[3][1]}

                                if j[2] in ('int', 'bool'):
                                    layer_params[j[0]] = rng.integers(**rge)
                                    if j[2] == 'bool':
                                        layer_params[j[0]] = True if layer_params[j[0]] == 0 else False
                                elif j[2] == 'float':
                                    layer_params[j[0]] = rng.uniform(**rge)

                                if j[3][0] == '>':
                                    layer_params[j[0]] += j[3][1]

                        if "conv" in layer_params["type"]:
                            for j in model[1].layer_params["conv"]:
                                if j[1] == 'const':
                                    layer_params[j[0]] = j[3]
                                elif j[1] == "select":
                                    layer_params[j[0]] = rng.choice(j[3])
                                else:
                                    rge = None
                                    if j[2] == 'bool':
                                        rge = {'low':2}
                                    elif j[1] == 'range':
                                        rge = {'low':j[3][0], 'high':j[3][1]}
                                    elif j[1] == 'threshold':
                                        rge = {'low':j[3][1]}

                                    if j[2] in ('int', 'bool'):
                                        layer_params[j[0]] = rng.integers(**rge)
                                        if j[2] == 'bool':
                                            layer_params[j[0]] = True if layer_params[j[0]] == 0 else False
                                    elif j[2] == 'float':
                                        layer_params[j[0]] = rng.uniform(**rge)

                                    if j[3][0] == '>':
                                        layer_params[j[0]] += j[3][1]

                        elif layer_params["type"] in ("lstm", "gru"):
                            for j in model[1].layer_params["rnn"]:
                                if j[1] == 'const':
                                    layer_params[j[0]] = j[3]
                                elif j[1] == "select":
                                    layer_params[j[0]] = rng.choice(j[3])
                                else:
                                    rge = None
                                    if j[2] == 'bool':
                                        rge = {'low':2}
                                    elif j[1] == 'range':
                                        rge = {'low':j[3][0], 'high':j[3][1]}
                                    elif j[1] == 'threshold':
                                        rge = {'low':j[3][1]}

                                    if j[2] in ('int', 'bool'):
                                        layer_params[j[0]] = rng.integers(**rge)
                                        if j[2] == 'bool':
                                            layer_params[j[0]] = True if layer_params[j[0]] == 0 else False
                                    elif j[2] == 'float':
                                        layer_params[j[0]] = rng.uniform(**rge)

                                    if j[3][0] == '>':
                                        layer_params[j[0]] += j[3][1]

                    else:
                        for j in model[1].layer_params["dropout"]:
                            if j[1] == 'const':
                                layer_params[j[0]] = j[3]
                            elif j[1] == "select":
                                layer_params[j[0]] = rng.choice(j[3])
                            else:
                                rge = None
                                if j[2] == 'bool':
                                    rge = {'low':2}
                                elif j[1] == 'range':
                                    rge = {'low':j[3][0], 'high':j[3][1]}
                                elif j[1] == 'threshold':
                                    rge = {'low':j[3][1]}

                                if j[2] in ('int', 'bool'):
                                    layer_params[j[0]] = rng.integers(**rge)
                                    if j[2] == 'bool':
                                        layer_params[j[0]] = True if layer_params[j[0]] == 0 else False
                                elif j[2] == 'float':
                                    layer_params[j[0]] = rng.uniform(**rge)

                                if j[3][0] == '>':
                                    layer_params[j[0]] += j[3][1]

                front = "conv" in layer_params["type"] and front

                params["layer_{}".format(i)] = layer_params

            optimizer_params = dict()
            for j in model[1].optimizer_param_list[params["optimizer"]]:
                i = model[1].optimizer_params[j]
                if i[1] == 'const':
                    optimizer_params[i[0]] = i[3]
                elif i[1] == "select":
                    optimizer_params[i[0]] = rng.choice(i[3])
                else:
                    rge = None
                    if i[2] == 'bool':
                        rge = {'low':2}
                    elif i[1] == 'range':
                        rge = {'low':i[3][0], 'high':i[3][1]}
                    elif i[1] == 'threshold':
                        rge = {'low':i[3][1]}

                    if i[2] in ('int', 'bool'):
                        optimizer_params[i[0]] = rng.integers(**rge)
                        if i[2] == 'bool':
                            optimizer_params[i[0]] = True if params[i[0]] == 0 else False
                    elif i[2] == 'float':
                        optimizer_params[i[0]] = rng.uniform(**rge)

                    if i[3][0] == '>':
                        optimizer_params[i[0]] += i[3][1]
            params["optimizer_params"] = optimizer_params

        return self._new_model(model[1], params=params, acc_passed=model[3])


    def _crossover(self, model1, model2, crossing_params):
        #print(crossing_params)
        rng = np.random.default_rng()
        params = (model1[2].copy(), model2[2].copy())
        layers_crossed = False
        optimizer_crossed = False
        # randomly cross over params
        for i in crossing_params:
            if not i in ['optimizer_params'] and rng.random() < self.p_cross:
                if i == 'layers':
                    layers_crossed = (params[0][i], params[1][i])
                if i == 'optimizer':
                    optimizer_crossed = True
                tmp = params[0][i]
                params[0][i] = params[1][i]
                params[1][i] = tmp

        if layers_crossed != False:
            obja = None
            objb = None
            if params[0]['layers'] > params[1]['layers']:
                obja = params[0]
                objb = params[1]
            else:
                obja = params[1]
                objb = params[0]

            for i in range(min(layers_crossed), max(layers_crossed)):
                obja["layer_{}".format(i)] = objb.pop("layer_{}".format(i))

        if optimizer_crossed == True:
            tmp = params[0]['optimizer_params']
            params[0]['optimizer_params'] = params[1]['optimizer_params']
            params[1]['optimizer_params'] = tmp

        avg_acc = (float(model1[3]) + float(model2[3])) / 2
        return (self._new_model(model1[1], params=params[0], acc_passed=avg_acc), self._new_model(model2[1], params=params[1], acc_passed=avg_acc))


    def _select(self):
        self._pool = self._pool + self.picked
        self._pool.sort(key=lambda x: x[3].score(), reverse=True)
        #print(self._pool)
        self.best_acc = self._pool[0][3].test.acc
        sum = 0
        self.picked = []
        picked_models = [ 0 for i in self._models ]
        for i in self._pool:
            if sum <= self._topick * self._acc:
                self.picked.append(i)
                sum += float(i[3])

                picked_models[self._models.index(i[1])] += 1
            elif self._no_extinct != False and picked_models[self._models.index(i[1])] < int(self._no_extinct):
                # do not let algo completely extinct
                self.picked.append(i)
                picked_models[self._models.index(i[1])] += 1
        self._pool = []
        gc.collect()


    def _populate(self):
        crossable = list()
        crossed_models = set()
        rng = np.random.default_rng()

        for i in range(len(self.picked)):
            for j in range(i+1, len(self.picked)):
                parami = set(self.picked[i][2].keys())
                paramj = set(self.picked[j][2].keys())
                its = parami.intersection(paramj)
                if len(its) > 0:
                    crossable.append((self.picked[i], self.picked[j], its))

        for i in crossable:
            pcross = 0
            if i[0][1] == i[1][1]:
                pcross = self.p_r * float(i[0][3]) * float(i[1][3])
            else:
                pcross = self.p_r * self.hybrid_p * float(i[0][3]) * float(i[1][3])

            n = 0
            while n < self.max_children * self.p_cross:
                n += 1
                if rng.random() < pcross:
                    model, model2 = self._crossover(i[0], i[1], i[2])

                    # mutation after crossover
                    if self._mutate_after_crossover:
                        model = self._mutate(model)
                        model2 = self._mutate(model)

                    self._pool.append(model)
                    self._pool.append(model2)

        for i in self.picked:
            # add rule that if model has children
            for j in range(self.max_mutate):
                self._pool.append(self._mutate(i))


    def _train(self, Xdata, ydata):
        X = Xdata
        y = ydata
        #print(X.shape)
        #print(y.shape)
        X_train = X
        X_test = X
        y_train = y
        y_test = y
        if self._split_data != None:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self._split_data, shuffle=True)

        for i in self._pool:
            perf = i[0].train(X_train, y_train, X_test, y_test)
            #print(perf)
            i[3] = ModelPerformance(self._score_weightings, perf)
        #print(self._pool)


    def _spawn(self):
        for i in self._models:
            self._pool.append(self._new_model(i))
        if len(self._models) < self._ps:
            rng = np.random.default_rng()
            model_chosen = rng.integers(0, len(self._models), (self._ps - len(self._pool)))
            for i in model_chosen:
                self._pool.append(self._new_model(self._models[i]))
        else:
            for i in self._models:
                for _ in range(self._iv - 1):
                    self._pool.append(self._new_model(i))


    def _cleanup(self):
        self._select()
        self.best = self.picked[0]
        self.best_acc = self.picked[0][3].test.acc


    def run(self, X_data, y_data):
        self._spawn()
        count = 0
        while count < self._gen and self.best_acc < self._acc:
            self._train(X_data, y_data)
            self._select()
            print(count)
            count += 1
            if (count < self._gen and self.best_acc < self._acc):
                self._populate()
        self._cleanup()


class GeneralPerformance:
    def __init__(self, weighting={'auc':1}):
        self.acc = 0
        self.recall = 0
        self.precision = 0
        self.f1 = 0
        self.log = 0
        self.auc = 0

        self._scr_wght = weighting


    def set_data(self, perf):
        for i in perf:
            if i == 'acc':
                self.acc = perf[i]
            elif i == 'recall':
                self.recall = perf[i]
            elif i == 'precision':
                self.precision = perf[i]
            elif i == 'f1':
                self.f1 = perf[i]
            elif i == 'log':
                self.log = perf[i]
            elif i == 'auc':
                self.auc = perf[i]


    def score(self):
        scr = 0
        if self._scr_wght is None:
            return 0
        for i in self._scr_wght:
            if i == 'acc':
                scr += self.acc * self._scr_wght[i]
            elif i == 'recall':
                scr += self.recall * self._scr_wght[i]
            elif i == 'precision':
                scr += self.precision * self._scr_wght[i]
            elif i == 'f1':
                scr += self.f1 * self._scr_wght[i]
            elif i == 'log':
                scr += self.log * self._scr_wght[i]
            elif i == 'auc':
                scr += self.auc * self._scr_wght[i]
        return scr


    def to_dict(self):
        return {"acc": self.acc, "recall": self.recall, "precision": self.precision, "f1": self.f1, "log": self.log, "auc": self.auc}


    def __float__(self):
        return self.score()


    def __str__(self):
        objstr = "accuracy: {}\n".format(self.acc)
        objstr += "recall: {}\n".format(self.recall)
        objstr += "precision: {}\n".format(self.precision)
        objstr += "F1 score: {}\n".format(self.f1)
        objstr += "log loss: {}\n".format(self.log)
        objstr += "AUC: {}\n".format(self.auc)
        return objstr


class ModelPerformance:
    def __init__(self, weightings={'train':{}, 'test':{'auc':1}}, perf=None):
        self.test = GeneralPerformance(weightings['test'])
        self.train = GeneralPerformance(weightings['train'])

        self._scr_wght = weightings

        if perf != None:
            self.train.set_data(perf['train'])
            self.test.set_data(perf['test'])


    def score(self):
        train_sum = np.sum(list(self._scr_wght['train'].values()))
        test_sum = np.sum(list(self._scr_wght['test'].values()))
        sums = train_sum + test_sum
        return self.train.score() * (train_sum / sums) + self.test.score() * (test_sum / sums)


    def to_dict(self):
        return {"train": self.train.to_dict(), "test": self.test.to_dict()}


    def __float__(self):
        return self.score()


    def __eq__(self, other):
        return float(self) == float(other)


    def __lt__(self, other):
        return float(self) < float(other)


    def __gt__(self, other):
        return float(self) > float(other)


    def __ne__(self, other):
        return not (self == other)


    def __le__(self, other):
        return self < other or self == other


    def __ge__(self, other):
        return self > other or self == other


    def __str__(self):
        objstr = "Training result:\n"
        objstr += str(self.train) + "\n"
        objstr += "Test results:\n"
        objstr += str(self.test) + "\n"
        return objstr
