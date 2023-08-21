import itertools
import os
import random
import sys
import webbrowser
from distutils.util import strtobool
from itertools import combinations
from typing import List

import numpy as np
import pandas
import plotly.graph_objects as go
import plotly.io as io
import seaborn
from plotly import express as px
from plotly import figure_factory as ff
from pycorrcat.pycorrcat import corr_matrix, plot_corr
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

# a.html is the final html file with all the required tables!!!

path = os.path.dirname(os.path.realpath(__file__))

TITANIC_PREDICTORS = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "embarked",
    "parch",
    "fare",
    "who",
    "adult_male",
    "deck",
    "embark_town",
    "alone",
    "class",
]


def get_test_data_set(data_set_name: str = None) -> (pandas.DataFrame, List[str], str):
    """Function to load a few test data sets

    :param:
    data_set_name : string, optional
        Data set to load

    :return:
    data_set : :class:`pandas.DataFrame`
        Tabular data, possibly with some preprocessing applied.
    predictors :list[str]
        List of predictor variables
    response: str
        Response variable
    """
    seaborn_data_sets = ["mpg", "tips", "titanic", "titanic_2"]
    sklearn_data_sets = ["boston", "diabetes", "breast_cancer"]
    all_data_sets = seaborn_data_sets + sklearn_data_sets

    if data_set_name is None:
        data_set_name = random.choice(all_data_sets)
    else:
        if data_set_name not in all_data_sets:
            raise Exception(f"Data set choice not valid: {data_set_name}")

    if data_set_name in seaborn_data_sets:
        if data_set_name == "mpg":
            data_set = seaborn.load_dataset(name="mpg").dropna().reset_index()
            data_set.drop(columns="name", inplace=True)
            predictors = [
                "cylinders",
                "displacement",
                "horsepower",
                "weight",
                "acceleration",
                "origin",
            ]
            response = "mpg"
        elif data_set_name == "tips":
            data_set = seaborn.load_dataset(name="tips").dropna().reset_index()
            predictors = [
                "total_bill",
                "sex",
                "smoker",
                "day",
                "time",
                "size",
            ]
            response = "tip"
        elif data_set_name == "titanic":
            data_set = seaborn.load_dataset(name="titanic").dropna()
            predictors = TITANIC_PREDICTORS
            response = "survived"
        elif data_set_name == "titanic_2":
            data_set = seaborn.load_dataset(name="titanic").dropna()
            predictors = TITANIC_PREDICTORS
            response = "alive"
    elif data_set_name in sklearn_data_sets:
        if data_set_name == "boston":
            data = datasets.load_boston()
            data_set = pandas.DataFrame(data.data, columns=data.feature_names)
            data_set["CHAS"] = data_set["CHAS"].astype(str)
        elif data_set_name == "diabetes":
            data = datasets.load_diabetes()
            data_set = pandas.DataFrame(data.data, columns=data.feature_names)
            data_set["gender"] = ["1" if i > 0 else "0" for i in data_set["sex"]]
        elif data_set_name == "breast_cancer":
            data = datasets.load_breast_cancer()
            data_set = pandas.DataFrame(data.data, columns=data.feature_names)

        data_set["target"] = data.target
        predictors = data.feature_names
        response = "target"

    print(f"Data set selected: {data_set_name}")
    return data_set, predictors, response


def fill_na(data):
    if isinstance(data, pandas.Series):
        return data.fillna(0)
    else:
        return np.array([value if value is not None else 0 for value in data])


def cat_cont_correlation_ratio(categories, values):
    """
    Correlation Ratio: https://en.wikipedia.org/wiki/Correlation_ratio
    SOURCE:
    1.) https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
    :param categories: Numpy array of categories
    :param values: Numpy array of values
    :return: correlation
    """
    f_cat, _ = pandas.factorize(categories)
    cat_num = np.max(f_cat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = values[np.argwhere(f_cat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    numerator = np.sum(
        np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2))
    )
    denominator = np.sum(np.power(np.subtract(values, y_total_avg), 2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator / denominator)
    return eta


def main():
    # print(list(df.columns))
    df.drop(["index"], axis=1, inplace=True, errors="ignore")
    X = df.loc[:, df.columns != response]
    y = df[response]
    con_list = []
    cat_list = []

    for att in X.columns:
        if X[att].dtype.kind in "bO":
            cat_list.append(att)
        if X[att].dtype.kind in "ifc":
            con_list.append(att)

    if len(y.unique()) == 2:
        y = np.array(y)
        if 1 not in y:
            for i in range(len(y)):
                y[i] = strtobool(y[i])
        print("response is cat")
        theresponse = "cat"
    else:
        print("response is con")
        theresponse = "con"

    con_X = X[con_list]

    print(len(con_list))
    print(con_list)
    print(len(cat_list))
    print(cat_list)

    if len(con_list) != 0:
        # con/con corr
        corr_p = con_X.corr("pearson")
        dff1 = corr_p.stack().reset_index(name="value")
        # print(dff1)
        # print(corr_p)

        fig = go.Figure()
        fig.add_trace(go.Heatmap(x=corr_p.columns, y=corr_p.index, z=np.array(corr_p)))
        # fig.show()
        con_con_cor_plot = io.to_html(fig, include_plotlyjs="cdn")
        dff1 = dff1.sort_values(by="value", ascending=False)

    cat_X = X[cat_list]

    if len(cat_list) != 0 and len(con_list) != 0:
        corr2list = []
        # cat/cont
        for i1, i2 in itertools.product(cat_X, con_X):
            rest = cat_cont_correlation_ratio(np.array(cat_X[i1]), np.array(con_X[i2]))
            corr2list.append([i1, i2, rest])
        corr2 = pandas.DataFrame(
            corr2list, columns=["predictor1", "predictor2", "value"]
        )
        corr3 = corr2.pivot(index="predictor1", columns="predictor2", values="value")
        # print(dff)
        fig = go.Figure()
        fig.add_trace(go.Heatmap(x=corr3.columns, y=corr3.index, z=np.array(corr3)))
        # fig.show()
        cat_cont_cor_plot = io.to_html(fig, include_plotlyjs="cdn")
        corr2 = corr2.sort_values(by="value", ascending=False)

    # cat/cat
    if len(cat_list) != 0:
        correlation_matrix = corr_matrix(cat_X, cat_list)
        catcor = correlation_matrix.stack().reset_index(name="value")
        plot_corr(cat_X, cat_list)
        # print("cat/cat metrix")
        # print(correlation_matrix)
        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                z=np.array(correlation_matrix),
            )
        )
        # fig.show()
        cat_cat_cor_plot = io.to_html(fig, include_plotlyjs="cdn")
        catcor = catcor.sort_values(by="value", ascending=False)

    # brute force cat/cat

    if len(cat_list) != 0:
        final1 = []
        for i1, i2 in combinations(cat_X.columns, 2):
            # print([cat_X[i1],cat_X[i2],y])
            ldf = pandas.DataFrame()
            ldf["X1"] = cat_X[i1]
            ldf["X2"] = cat_X[i2]
            ldf["response"] = y
            thislist1 = []

            # print((i1, i2))
            for key, group in ldf.groupby(["X1", "X2"]):
                calcc = np.array(group["response"])
                difference_array = calcc - np.mean(y)
                squared_array = np.square(difference_array)
                mse = squared_array.mean()
                count1 = len(group) / len(ldf)
                wmse = mse * count1
                thislist1.append([key[0], key[1], mse, wmse])
                thatlist1 = pandas.DataFrame(
                    thislist1, columns=["X1", "X2", "mse", "wmse"]
                )
                thoselist1 = thatlist1.pivot(index="X1", columns="X2", values="mse")

            fig = go.Figure()
            fig.add_trace(
                go.Heatmap(
                    x=thoselist1.columns, y=thoselist1.index, z=np.array(thoselist1)
                )
            )
            # fig.show()
            fig.write_html(f"bcc{i1}{i2}.html")
            # print([i1,i2])
            a1 = np.array(thatlist1["mse"])
            mean_m1 = np.mean(a1)
            # print(mean_m1)
            b1 = np.array(thatlist1["wmse"])
            mean_wm1 = np.mean(b1)
            # print(mean_wm1)
            final1.append(
                [i1, i2, f"<a href='//{path}/bcc{i1}{i2}.html'>{mean_m1}</a>", mean_wm1]
            )
        finaldf1 = pandas.DataFrame(
            final1, columns=["Predictor1", "Predictor2", "mse", "wmse"]
        )
        finaldf1 = finaldf1.sort_values(by="wmse", ascending=False)

    # brute force con/con
    if len(con_list) != 0:
        final2 = []

        for i1, i2 in combinations(con_X.columns, 2):
            # ddf = pandas.cut(con_X.i1, bins=2)
            # print([i1, i2])
            ddf1 = pandas.cut(con_X[i1], bins=10)
            ddf2 = pandas.cut(con_X[i2], bins=10)
            ldf2 = pandas.DataFrame()
            ldf2["X1"] = ddf1
            ldf2["X2"] = ddf2
            ldf2["response"] = y
            # print(ldf2)
            thislist2 = []
            for key, group in ldf2.groupby(["X1", "X2"]):
                # print(i1, i2)
                # print(key)
                # print(group)
                calcc = np.array(group["response"])
                difference_array = calcc - np.mean(y)
                squared_array = np.square(difference_array)
                mse = squared_array.mean()
                count2 = len(group) / len(ldf2)
                wmse = mse * count2
                thislist2.append([key[0], key[1], mse, wmse])
                thatlist2 = pandas.DataFrame(
                    thislist2, columns=["X1", "X2", "mse", "wmse"]
                )
                thoselist2 = thatlist2.pivot(index="X1", columns="X2", values="mse")
            # print(thoselist)
            fig = go.Figure()
            fig.add_trace(
                go.Heatmap(
                    x=thoselist2.columns.astype("str"),
                    y=thoselist2.index.astype("str"),
                    z=np.array(thoselist2),
                )
            )
            # fig.show()
            fig.write_html(f"boo{i1}{i2}.html")

            a2 = np.array(thatlist2["mse"])
            mean_m2 = np.mean(a2)
            b2 = np.array(thatlist2["wmse"])
            mean_wm2 = np.mean(b2)
            final2.append(
                [i1, i2, f"<a href='//{path}/boo{i1}{i2}.html'>{mean_m2}</a>", mean_wm2]
            )
        finaldf2 = pandas.DataFrame(
            final2, columns=["Predictor1", "Predictor2", "mse", "wmse"]
        )
        finaldf2 = finaldf2.sort_values(by="wmse", ascending=False)

    # brute force con/cat
    if len(con_list) != 0 and len(cat_list) != 0:
        final3 = []
        for i1, i2 in itertools.product(cat_X, con_X):
            # ddf = pandas.cut(con_X.i1, bins=2)
            # print([i1, i2])
            ddf11 = pandas.cut(con_X[i2], bins=10)
            # print(ddf11)

            ldf3 = pandas.DataFrame()
            ldf3["X1"] = cat_X[i1]
            ldf3["X2"] = ddf11
            ldf3["response"] = y
            thislist3 = []
            for key, group in ldf3.groupby(["X1", "X2"]):
                # print(i1, i2)
                # print(key)
                # print(group)
                calcc = np.array(group["response"])
                difference_array = calcc - np.mean(y)
                squared_array = np.square(difference_array)
                mse = squared_array.mean()
                count3 = len(group) / len(ldf3)
                wmse = mse * count3
                thislist3.append([key[0], key[1], mse, wmse])
                thatlist3 = pandas.DataFrame(
                    thislist3, columns=["X1", "X2", "mse", "wmse"]
                )
                thoselist3 = thatlist3.pivot(index="X1", columns="X2", values="mse")

            fig = go.Figure()
            fig.add_trace(
                go.Heatmap(
                    x=thoselist3.columns.astype("str"),
                    y=thoselist3.index,
                    z=np.array(thoselist3),
                )
            )
            # fig.show()
            fig.write_html(f"bco{i1}{i2}.html")
            a3 = np.array(thatlist3["mse"])
            mean_m3 = np.mean(a3)
            b3 = np.array(thatlist3["wmse"])
            mean_wm3 = np.mean(b3)
            final3.append(
                [i1, i2, f"<a href='//{path}/bco{i1}{i2}.html'>{mean_m3}</a>", mean_wm3]
            )
        finaldf3 = pandas.DataFrame(
            final3, columns=["Predictor1", "Predictor2", "mse", "wmse"]
        )
        finaldf3 = finaldf3.sort_values(by="wmse", ascending=False)

    plottable = pandas.DataFrame()
    # plot cat/cat
    plotslist = []

    if len(cat_list) != 0 and theresponse == "cat":
        stry = []
        tfy = (y > 0).tolist()
        for i in tfy:
            if i is True:
                stry.append("True")
            else:
                stry.append("False")

        for att in cat_X.columns:
            XXX = np.array(cat_X[att].astype(str))
            conf_matrix = confusion_matrix(XXX, np.array(stry))
            fig_2 = go.Figure(
                data=go.Heatmap(z=conf_matrix, zmin=0, zmax=conf_matrix.max())
            )
            fig_2.update_layout(
                title=att,
                xaxis_title="Response",
                yaxis_title="Predictor",
            )
            # fig_no_relationship.show()
            fig_2.write_html(f"hw4cc{att}.html")
            plotslist.append(f"<a href='//{path}/hw4cc{att}.html'>{att}</a>")

    # plot con/cat
    if len(con_list) != 0 and theresponse == "cat":
        plot_df = con_X
        plot_df["target"] = y

        plot_dfoc = plot_df.loc[:, ~plot_df.T.duplicated(keep="last")]

        for i in plot_dfoc.iloc[:, :-1]:
            hist_data = []
            classes = plot_dfoc.groupby("target")[i].apply(list)
            for a in classes:
                hist_data.append(a)

            group_labels = list(plot_dfoc.groupby("target").groups.keys())

            # Create distribution plot with custom bin_size
            fig_1 = ff.create_distplot(hist_data, group_labels, bin_size=0.2)
            fig_1.update_layout(
                title="Continuous Predictor by Categorical Response",
                xaxis_title="Predictor",
                yaxis_title="Distribution",
            )
            # fig_1.show()
            fig_1.write_html(f"hw4oc{i}.html")
            plotslist.append(f"<a href='//{path}/hw4oc{i}.html'>{i}</a>")

    # plot con/con
    if len(con_list) != 0 and theresponse == "con":
        for i in range(np.shape(con_X)[1]):
            fig = px.scatter(x=con_X.iloc[:, i], y=y, trendline="ols")
            fig.update_layout(
                title="Continuous Response by Continuous Predictor",
                xaxis_title="Predictor",
                yaxis_title="Response",
            )
            # fig.show()
            name = con_X.iloc[:, i].name
            fig.write_html(f"hw4oo{name}.html")
            plotslist.append(f"<a href='//{path}/hw4oo{name}.html'>{name}</a>")

    # plot cat/con
    if len(cat_list) != 0 and theresponse == "con":
        plot_df2 = pandas.DataFrame(cat_X)
        plot_df2["target"] = y
        for i in plot_df2.iloc[:, :-1]:
            label = LabelEncoder()
            plot_df2[i] = label.fit_transform(plot_df2[i])
            hist_data = []
            classes = plot_df2.groupby(i)["target"].apply(list)
            for a in classes:
                hist_data.append(a)

            group_labels = list(plot_df2.groupby(i).groups.keys())

            fig_1 = ff.create_distplot(hist_data, group_labels, bin_size=0.2)
            fig_1.update_layout(
                title="Continuous Predictor by Categorical Response",
                xaxis_title="Predictor",
                yaxis_title="Distribution",
            )
            # fig_1.show()
            fig_1.write_html(f"hw4co{i}.html")
            plotslist.append(f"<a href='//{path}/hw4co{i}.html'>{i}</a>")

    plottable["predictors"] = plotslist

    with open("a.html", "w") as _file:
        _file.write("<h1>Mid-term Brian</h1>")
        if len(con_list) != 0:
            _file.write("<h1>con/con correlation</h1>" + dff1.to_html())
            _file.write("<h1>con/con correlation plot</h1>" + con_con_cor_plot)
        if len(con_list) != 0 and len(cat_list) != 0:
            _file.write("<h1>cat/con correlation</h1>" + corr2.to_html())
            _file.write("<h1>cat/con correlation polt</h1>" + cat_cont_cor_plot)
        if len(cat_list) != 0:
            _file.write("<h1>cat/cat correlation" + catcor.to_html())
            _file.write("<h1>cat/cat correlation polt</h1>" + cat_cat_cor_plot)

        if len(cat_list) != 0:
            _file.write(
                "<h1>brute force cat/cat</h1>"
                + finaldf1.to_html(render_links=True, escape=False)
            )

        if len(con_list) != 0:
            _file.write(
                "<h1>brute force con/con</h1>"
                + finaldf2.to_html(render_links=True, escape=False)
            )

        if len(con_list) != 0 and len(cat_list) != 0:
            _file.write(
                "<h1>brute force cat/con</h1>"
                + finaldf3.to_html(render_links=True, escape=False)
            )

        _file.write(
            "<h1>HW4 plot table (all predictors)</h1>"
            + plottable.to_html(render_links=True, escape=False)
        )

    url = f"file:///{path}/a.html"
    webbrowser.open(url, new=2)


if __name__ == "__main__":
    df, predictors, response = get_test_data_set()
    sys.exit(main())
