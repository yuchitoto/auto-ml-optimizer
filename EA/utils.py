from .Models import *

"""
model_list: dict of model lists, key as models, value of dict of model parameters in_dim array if is children, else dict
in_dim: input dimension
out_dim: output dimension
"""
def load_models_from_dict(model_list, in_dim, out_dim):
    models = []

    for i in model_list:
        prm = model_list[i]
        md = None

        if i == "Random Forest":
            md = RandomForestClassifierTemplate(input_dim=in_dim, output_dim=out_dim)
        elif i == "BernoulliNB":
            md = BernoulliNBTemplate(input_dim=in_dim, output_dim=out_dim)
        elif i == "ComplementNB":
            md = ComplementNBTemplate(input_dim=in_dim, output_dim=out_dim)
        elif i == "CategoricalNB":
            md = CategoricalNBTemplate(input_dim=in_dim, output_dim=out_dim)
        elif i == "GaussianNB":
            md = GaussianNBTemplate(input_dim=in_dim, output_dim=out_dim)
        elif i == "MultinomialNB":
            md = MultinomialNBTemplate(input_dim=in_dim, output_dim=out_dim)
        elif i == "KNeighborsClassifier":
            md = KNeighborsClassifierTemplate(input_dim=in_dim, output_dim=out_dim)
        elif i == "RadiusNeighborsClassifier":
            md = RadiusNeighborsClassifierTemplate(input_dim=in_dim, output_dim=out_dim)
        elif i == "DecisionTreeClassifier":
            md = DecisionTreeClassifierTemplate(input_dim=in_dim, output_dim=out_dim)
        elif i == "ExtraTreeClassifier":
            md = ExtraTreeClassifierTemplate(input_dim=in_dim, output_dim=out_dim)
        elif i == "neural network":
            md = EnhancedNNClassifierTemplate(input_dim=in_dim, output_dim=out_dim)
        elif i == "deep neural network":
            md = DNNClassifierTemplate(input_dim=in_dim, output_dim=out_dim)
        elif i == "wbd neural network":
            md = WeightedBinaryDNNClassifierTemplate(input_dim=in_dim, output_dim=out_dim)
        elif i == "convolutional neural network":
            md = CNNClassifierTemplate(input_dim=in_dim, output_dim=out_dim)
        elif i == "recurrent neural network":
            md = RNNClassifierTemplate(input_dim=in_dim, output_dim=out_dim)

        for j in prm:
            updt = False
            for k in range(len(md.params_desc)):
                if j == md.params_desc[k][0]:
                    md.params_desc[k][1] = prm[j][0]
                    md.params_desc[k][3] = prm[j][1]
                    updt = True
                    break
            if updt == False and "neural network" in md.name:
                if j == "type":
                    md.layer_type[3] = prm[j][1]
                elif j in md.layer_params:
                    for a in prm[j]:
                        for b in range(len(md.layer_params[j])):
                            if a == md.layer_params[j][b][0]:
                                md.layer_params[j][b][1] = prm[j][a][0]
                                md.layer_params[j][b][3] = prm[j][a][1]
                                break

                elif j in md.optimizer_params:
                    md.optimizer_params[j][1] = prm[j][0]
                    md.optimizer_params[j][3] = prm[j][1]

        models.append(md)

    return models
