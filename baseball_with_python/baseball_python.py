import os
import sys
import webbrowser
from itertools import combinations

import numpy as np
import pandas
import plotly.graph_objects as go
import plotly.io as io
import statsmodels.api
from plotly import express as px
from plotly import figure_factory as ff
from pyspark import StorageLevel
from pyspark.sql import SparkSession
from sklearn import linear_model, metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split


def main():
    path = os.path.dirname(os.path.realpath(__file__))

    # jar file location in config
    spark = (
        SparkSession.builder.appName("myApp")
        .master("local[*]")
        .config("spark.jars", f"{path}/mariadb-java-client.jar")
        .getOrCreate()
    )

    user = "root"
    password = "*"

    jdbc_url = "jdbc:mysql://localhost:3306/baseball?permitMysqlScheme"
    jdbc_driver = "org.mariadb.jdbc.Driver"

    df1 = (
        spark.read.format("jdbc")
        .option("url", jdbc_url)
        .option("dbtable", "baseball.result")
        .option("user", user)
        .option("password", password)
        .option("driver", jdbc_driver)
        .load()
    )

    df1.createOrReplaceTempView("train_df")
    df1.persist(StorageLevel.DISK_ONLY)
    df = df1.toPandas()
    # df['plate_appear'] = df['plate_appear'].astype(float)
    response = "hometeam_win"
    df.drop("gameid", inplace=True, axis=1)
    df.drop("date", inplace=True, axis=1)

    # print(list(df.columns))
    X = df.loc[:, df.columns != response]
    y = df[response]

    finaltable = pandas.DataFrame(
        index=list(X.columns),
        columns=["feature_names", "mse", "wmse", "t_value", "p_value", "random_forest"],
    )
    finaltable["feature_names"] = list(X.columns)

    # con/con corr
    corr_p = X.corr("pearson")
    dff1 = corr_p.stack().reset_index(name="value")
    # print(dff1)
    # print(corr_p)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=corr_p.columns, y=corr_p.index, z=np.array(corr_p)))
    # fig.show()
    con_con_cor_plot = io.to_html(fig, include_plotlyjs="cdn")
    dff1 = dff1.sort_values(by="value", ascending=False)

    # brute force con/con

    final2 = []

    for i1, i2 in combinations(X.columns, 2):
        # ddf = pandas.cut(con_X.i1, bins=2)
        # print([i1, i2])
        ddf1 = pandas.cut(X[i1], bins=10)
        ddf2 = pandas.cut(X[i2], bins=10)
        ldf2 = pandas.DataFrame()
        ldf2["X1"] = ddf1
        ldf2["X2"] = ddf2
        ldf2["response"] = y
        # print(ldf2)
        thislist2 = []
        count_l2 = []
        for key, group in ldf2.groupby(["X1", "X2"]):
            # print(i1, i2)
            # print(key)
            # print(group)
            calcc = np.array(group["response"])
            mse = calcc.mean()
            count_l2.append(len(group))
            thislist2.append([key[0], key[1], mse])
            thatlist2 = pandas.DataFrame(thislist2, columns=["X1", "X2", "mse"])
            thoselist2 = thatlist2.pivot(index="X1", columns="X2", values="mse")
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
        mean_m2 = a2 - np.mean(y)
        mean_mm2 = np.square(mean_m2)
        mean_m2 = sum(mean_mm2)
        mean_m2 = mean_m2 / len(a2)

        mean_wm2 = np.multiply(mean_mm2, np.array(count_l2) / sum(count_l2))
        mean_wm2 = sum(mean_wm2)

        final2.append(
            [i1, i2, f"<a href='//{path}/boo{i1}{i2}.html'>{mean_m2}</a>", mean_wm2]
        )
    finaldf2 = pandas.DataFrame(
        final2, columns=["Predictor1", "Predictor2", "mse", "wmse"]
    )
    finaldf2 = finaldf2.sort_values(by="wmse", ascending=False)

    plottable = pandas.DataFrame()
    plotslist = []

    # plot con/cat
    plot_df = X
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

    plottable["predictors"] = plotslist

    X.drop("target", inplace=True, axis=1)

    # mean of response mse

    for att in X.columns:
        pindf = pandas.DataFrame()
        # print(df[att])
        pins = pandas.cut(df[att], bins=10)
        pindf["X1"] = pins
        pindf["response"] = y
        alisted = []
        # print(pins)
        for key, group in pindf.groupby(["X1"]):
            # print(group)
            caly = np.array(group["response"])
            mee = np.mean(caly)
            alisted.append([key, mee, len(group)])
            blisted = pandas.DataFrame(alisted, columns=["key", "diff mean", "pop"])
            blisted["bin_center"] = blisted["key"].apply(lambda x: x.mid)
            blisted["popmean"] = np.mean(y)

        fig_2 = go.Figure(
            layout=go.Layout(
                title="Binned difference with mean of response vs mean",
                yaxis2=dict(overlaying="y"),
            )
        )
        fig_2.add_trace(go.Bar(x=blisted["bin_center"], y=blisted["pop"], yaxis="y1"))
        fig_2.add_trace(
            go.Scatter(
                x=blisted["bin_center"],
                y=blisted["diff mean"],
                yaxis="y2",
                mode="lines",
                line=go.scatter.Line(color="red"),
            )
        )
        fig_2.add_trace(
            go.Scatter(
                x=blisted["bin_center"],
                y=blisted["popmean"],
                yaxis="y2",
                mode="lines",
                line=go.scatter.Line(color="green"),
                showlegend=False,
            )
        )
        fig_2.write_html(f"mor{att}.html")

        ac = np.array(blisted["diff mean"])
        calcu = ac - np.mean(y)
        calcuu = np.square(calcu)
        calcu = np.nansum(calcuu)
        calcu = calcu / len(ac)

        w_calcu = np.multiply(
            calcuu, (np.array(blisted["pop"]) / np.array(blisted["pop"]).sum())
        )
        w_calcu = np.nansum(w_calcu)

        finaltable.loc[att, "mse"] = f"<a href='//{path}/mor{att}.html'>{calcu}</a>"
        finaltable.loc[att, "wmse"] = w_calcu

    stat_X = np.array(X)
    for idx, column in enumerate(stat_X.T):
        feature_name = list(X.columns)[idx]
        predictor = statsmodels.api.add_constant(column)
        linear_regression_model = statsmodels.api.Logit(y, predictor, missing="drop")
        linear_regression_model_fitted = linear_regression_model.fit()
        # print(f"Variable: {feature_name}")
        # print(linear_regression_model_fitted.summary())

        # Get the stats
        t_value = round(linear_regression_model_fitted.tvalues[1], 6)
        p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])
        # print(t_value)
        # print(p_value)
        finaltable.loc[
            feature_name, "t_value"
        ] = f"<a href='//{path}/stat{att}.html'>{t_value}</a>"
        finaltable.loc[
            feature_name, "p_value"
        ] = f"<a href='//{path}/stat{att}.html'>{p_value}</a>"

        # Plot the figure
        fig_6 = px.scatter(x=column, y=y, trendline="lowess")
        fig_6.update_layout(
            title=f"Variable: {feature_name}: (t-value={t_value}) (p-value={p_value})",
            xaxis_title=f"Variable: {feature_name}",
            yaxis_title="y",
        )
        fig_6.write_html(f"stat{att}.html")

    rf = RandomForestRegressor(n_estimators=50)
    rf.fit(X, y)

    for i in range(len(list(X.columns))):
        finaltable.loc[list(X.columns)[i], "random_forest"] = rf.feature_importances_[i]

    fig_7 = px.bar(x=list(X.columns), y=rf.feature_importances_)
    fig_7.update_layout(
        title="forest ranking",
        xaxis_title="predictor",
        yaxis_title="y",
    )
    rf_plot = io.to_html(fig_7, include_plotlyjs="cdn")

    X.drop("a_ip", inplace=True, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, shuffle=False
    )

    rf = RandomForestClassifier(n_estimators=50)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))

    logr = linear_model.LogisticRegression()
    logr.fit(X_train, y_train)
    predicted = logr.predict(X_test)
    print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, predicted))

    with open("a.html", "w") as _file:
        _file.write("<h1>Homework 5 Brian</h1>")

        _file.write("<h1>con/con correlation</h1>" + dff1.to_html())
        _file.write("<h1>con/con correlation plot</h1>" + con_con_cor_plot)

        _file.write(
            "<h1>brute force con/con</h1>"
            + finaldf2.to_html(index=False, render_links=True, escape=False)
        )

        _file.write(
            "<h1>HW4 plot table (all predictors)</h1>"
            + plottable.to_html(index=False, render_links=True, escape=False)
        )

        _file.write(
            "<h1>homework 4 table</h1>"
            + finaltable.to_html(index=False, render_links=True, escape=False)
        )

        _file.write("<h1>random forest plot</h1>" + rf_plot)

        _file.write(
            f"<h2>random forest model: accuracy{metrics.accuracy_score(y_test, y_pred)}</h2>"
        )
        _file.write(
            f"<h2>logistic regression model: accuracy{metrics.accuracy_score(y_test, predicted)}</h2>"
        )
        _file.write(
            "<h4>logistic regression is better, the accuracy is a little higher and more stable</h4>"
        )

    url = f"file:///{path}/a.html"
    webbrowser.open(url, new=2)


if __name__ == "__main__":
    sys.exit(main())
