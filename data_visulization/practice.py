import sys

import numpy as np
import pandas
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


def main():
    # load Fisher's Iris dataset into a DataFrame
    iris_df = pandas.read_csv("iris.data", header=None)
    iris_df.columns = [
        "sepal length",
        "sepal width",
        "petal length",
        "petal width",
        "class",
    ]

    # get simple statistic summary using numpy
    iris_np = iris_df.drop("class", axis=1)
    iris_np1 = np.array(iris_np)
    iris_mean = np.mean(iris_np1, axis=0)
    iris_max = np.max(iris_np1, axis=0)
    iris_min = np.min(iris_np1, axis=0)
    iris_quantile = np.quantile(iris_np1, 0.5, axis=0)
    print(f"data mean:{iris_mean}")
    print(f"data max:{iris_max}")
    print(f"data min:{iris_min}")
    print(f"data quantile:{iris_quantile}")

    # trying five different plots

    fig1 = px.scatter_matrix(
        iris_df,
        dimensions=["sepal width", "sepal length", "petal width", "petal length"],
        color="class",
    )
    fig1.show()

    label = LabelEncoder()
    iris_df["class"] = label.fit_transform(iris_df["class"])
    fig2 = px.parallel_coordinates(
        iris_df,
        color="class",
        labels={
            "species_id": "class",
            "sepal_width": "spal width",
            "sepal_length": "sepal length",
            "petal_width": "petal width",
            "petal_length": "petal length",
        },
        color_continuous_scale=px.colors.diverging.Tealrose,
        color_continuous_midpoint=2,
    )
    fig2.show()
    iris_df["class"] = label.inverse_transform(iris_df["class"])

    fig3 = px.scatter_3d(
        iris_df,
        x="sepal length",
        y="sepal width",
        z="petal length",
        color="class",
        hover_data=iris_df.columns,
    )
    fig3.update_layout(scene_zaxis_type="log")
    fig3.show()

    fig4_1 = px.violin(
        iris_df,
        y="petal length",
        x="petal width",
        color="class",
        box=True,
        points="all",
        hover_data=iris_df.columns,
    )
    fig4_2 = px.violin(
        iris_df,
        y="sepal length",
        x="sepal width",
        color="class",
        box=True,
        points="all",
        hover_data=iris_df.columns,
    )

    fig4 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)

    for i in fig4_1.data:
        fig4.add_trace(i, row=1, col=1)

    for i in fig4_2.data:
        fig4.add_trace(i, row=2, col=1)

    fig4.show()

    fig5_1 = px.histogram(iris_df, x="sepal width", color="class")
    fig5_2 = px.histogram(iris_df, x="sepal length", color="class")
    fig5_3 = px.histogram(iris_df, x="petal width", color="class")
    fig5_4 = px.histogram(iris_df, x="petal length", color="class")

    fig5 = make_subplots(rows=2, cols=2, shared_xaxes=True, vertical_spacing=0.02)

    for i in fig5_1.data:
        fig5.add_trace(i, row=1, col=1)

    for i in fig5_2.data:
        fig5.add_trace(i, row=2, col=1)

    for i in fig5_3.data:
        fig5.add_trace(i, row=1, col=2)

    for i in fig5_4.data:
        fig5.add_trace(i, row=2, col=2)
    fig5.update_layout(
        title_text="Histogram-Types of comparisons based on different attributes"
    )
    fig5.show()

    # scikit-learn(random forest)
    x = iris_np
    y = iris_df["class"]
    scaler = StandardScaler()
    x_tr = scaler.fit_transform(x)

    random_tree = RandomForestClassifier(n_estimators=10)
    random_tree.fit(x_tr, y)

    predict = random_tree.predict(x)
    score = random_tree.score(x, y)
    print(f"random tree prediction{predict}")
    print(f"random tree score{score}")

    # scikit-learn(k-means)
    knn = KNeighborsClassifier(n_neighbors=7, p=2, metric="minkowski")
    knn.fit(x_tr, y)
    print(f"KNN Training data accuracy: {(knn.score(x_tr, y) * 100)}")
    print(f"KNN Testing data accuracy: {(knn.score(x_tr, y) * 100)}")
    print(f"K-mean prediction:{knn.predict(x_tr)}")

    # pipline
    print("------------Model(Random Forest) via Pipeline Predictions------------")

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("randomtree", RandomForestClassifier(n_estimators=10)),
        ]
    )
    pipeline.fit(x, y)

    prediction = pipeline.predict(x)
    score = pipeline.score(x, y)
    print(f"Predictions: {prediction}")
    print(f"score: {score}")


if __name__ == "__main__":
    sys.exit(main())
