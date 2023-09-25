import pandas as pd
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import numpy as np
import umap
import seaborn
import time
from sklearn import preprocessing
from sklearn.manifold import TSNE

DATASET_FILE = "./geometrics.mtb-news.de.csv"
MNIST_DATASET_FILE = "./fashion-mnist_test.csv"


def read_data(file: str, sep: str = ';') -> pd.DataFrame:
    return pd.read_csv(file, sep=sep)


def print_info(data: pd.DataFrame):
    print(data.info())
    print(data.head(10))


def pre_process_data(data: pd.DataFrame):
    data[data.columns[10:]] = data[data.columns[10:]].fillna(0)
    del data["Frame Config"]
    data.dropna(inplace=True)


def _set_base_graph_layout(figure: go.Figure):
    figure.update_layout(
        title={
            "x": 0.5,
            "y": 0.99,
            "font_size": 20,
            "xanchor": "center",
            "yanchor": "top",
        },
        xaxis={
            "title_font_size": 16,
            "tickfont_size": 14,
            "tickangle": 315,
            "gridcolor": "ivory",
            "gridwidth": 2,
        },
        yaxis={
            "title_font_size": 16,
            "tickfont_size": 14,
            "gridcolor": "ivory",
            "gridwidth": 2,
        },
        width=None,
        height=700,
        margin=dict(l=0, r=0, t=40, b=0),
    )


def build_bar_graph(data: pd.DataFrame):
    data = data[["Category", "Reach"]].groupby("Category").mean().reset_index()

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=data["Category"],
            y=data["Reach"],
            marker=dict(
                color=data["Reach"].values,
                coloraxis="coloraxis",
                line=dict(
                    color="black",
                    width=2,
                ),
            ),
        )
    )

    _set_base_graph_layout(figure)
    figure.update_layout(
        title_text="Avg Reach value",
        xaxis_title="Category",
        yaxis_title="Reach",
    )

    figure.show()


def build_pie_graph(data: pd.DataFrame):
    data = data[data["Category"] == "Mountain"]["Wheel Size"].value_counts()

    figure = go.Figure()
    figure.add_trace(
        go.Pie(
            labels=data.index,
            values=data.values,
            marker=dict(
                line=dict(
                    color="black",
                    width=2,
                ),
            ),
        )
    )

    _set_base_graph_layout(figure)
    figure.update_layout(
        title_text="Wheel sizes on Mountain bikes",
    )

    figure.show()


def build_linear_graph(data: pd.DataFrame):
    data = data[
        (data["Head Tube Angle"] > 0)
        & (data["Year"] > 2010)
        & (data["Category"] == "Mountain")
        & (data["Suspension Travel (front)"] > 0)
    ]

    data1 = data[["Year", "Head Tube Angle"]].groupby("Year").mean().reset_index()
    data2 = data[["Year", "Seat Tube Angle Effective"]].groupby("Year").mean().reset_index()
    data3 = data[["Suspension Travel (front)", "Head Tube Angle"]].groupby(["Suspension Travel (front)"]).mean().reset_index()

    plt.rcParams.update({
        "axes.grid": True,
        "grid.color": "mistyrose",
        "grid.linewidth": 2,
        "lines.markersize": 2,
        "lines.marker": ".",
        "lines.markeredgecolor": "black",
        "lines.markerfacecolor": "white",
    })
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Mountain bikes analysis")

    ax1.plot(data1["Year"].values, data1["Head Tube Angle"].values, color="crimson")
    ax1.set_xlabel("year")
    ax1.set_ylabel("head tube angle")

    ax2.plot(data2["Year"].values, data2["Seat Tube Angle Effective"].values, color="crimson")
    ax2.set_xlabel("year")
    ax2.set_ylabel("seat tube angle")

    ax3.plot(data3["Suspension Travel (front)"].values, data3["Head Tube Angle"].values, color="crimson")
    ax3.set_xlabel("front suspension travel")
    ax3.set_ylabel("head tube angle")

    plt.savefig("./linear.png")


def build_tsne(data: pd.DataFrame):
    d = data.drop("label", axis=1)

    scaler = preprocessing.MinMaxScaler()
    d = pd.DataFrame(scaler.fit_transform(d), columns=d.columns)

    tsne_features = {}
    perplexities = (5, 25, 50)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for perplexity in perplexities:
        start_clock = time.monotonic_ns()

        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=123)
        tsne_features[perplexity] = tsne.fit_transform(d)

        execution_time = time.monotonic_ns() - start_clock
        print("Time of execution t-SNE =", execution_time * 10**(-6), "ms")

    for i, perplexity in enumerate(perplexities):
        tsne_unit = tsne_features[perplexity]

        d_copy = d.copy()
        d_copy["x"] = tsne_unit[:, 0]
        d_copy["y"] = tsne_unit[:, 1]

        plot = seaborn.scatterplot(
            x="x",
            y="y",
            hue=data["label"],
            data=d_copy,
            palette="bright",
            ax=axes[i],
        )
        plot.set(title=f"perplexity={perplexity}")

        if i != 0:
            plot.get_legend().remove()

    plt.savefig(f"./tsne.png")


def build_umap(data: pd.DataFrame):
    d = data.drop("label", axis=1)

    scaler = preprocessing.MinMaxScaler()
    d = pd.DataFrame(scaler.fit_transform(d), columns=d.columns)

    n_neighbors = (5, 25, 50)
    min_dist = (0.1, 0.6)

    um = {}

    for i in range(len(n_neighbors)):
        for j in range(len(min_dist)):
            start_clock = time.monotonic_ns()

            um[(n_neighbors[i], min_dist[j])] = (
                umap.UMAP(
                    n_neighbors=n_neighbors[i],
                    min_dist=min_dist[j],
                    random_state=123,
                ).fit_transform(d)
            )

            execution_time = time.monotonic_ns() - start_clock
            print("Time of execution UMAP =", execution_time * 10**(-6), "ms")

    fig, axes = plt.subplots(len(n_neighbors), len(min_dist), figsize=(15, 15))

    for i in range(len(n_neighbors)):
        for j in range(len(min_dist)):
            umap_features = um[(n_neighbors[i], min_dist[j])]

            d_copy = d.copy()
            d_copy["x"] = umap_features[:, 0]
            d_copy["y"] = umap_features[:, 1]

            plot = seaborn.scatterplot(
                x="x",
                y="y",
                hue=data["label"],
                data=d_copy,
                palette="bright",
                ax=axes[i, j],
            )
            plot.set(title=f"n_neighbors={n_neighbors[i]}, min_dist={min_dist[j]}")

            if not (i == 0 and j == 0):
                axes[i, j].legend().set_visible(False)

    plt.savefig(f"./umap.png")



def main():
    data = read_data(file=DATASET_FILE)
    # pre_process_data(data)
    # print_info(data)
    # build_bar_graph(data)
    # build_pie_graph(data)
    build_linear_graph(data)

    # mnist_data = read_data(file=MNIST_DATASET_FILE, sep=',')
    # print_info(mnist_data)
    # build_tsne(mnist_data)
    # build_umap(mnist_data)


if __name__ == "__main__":
    main()
