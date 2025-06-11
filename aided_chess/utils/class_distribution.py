import json

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_class_distribution():
    f = open("../../data/train-keypoints.json")

    d = json.load(f)

    count = {}

    data = []

    image_count_with_classes = {}

    for a in d.get("annotations"):
        i = a.get("id")
        image = a.get("image_id")
        category = a.get("category_id")
        data.append({"id": i, "image": image, "category": category})
        c = count.get(category, 0)
        c = c + 1
        count.update({category: c})

    m = list(
        map(
            lambda i: {
                "id": i.get("id"),
                "name": i.get("name"),
            },
            d.get("categories"),
        )
    )

    df = pd.DataFrame(data)

    print(df.head())

    grouped = df.groupby("category")

    unique_images = grouped.image.unique()

    categories = {
        1: "board",
        6: "black-pawn",
        11: "white-pawn",
        16: "black-rook",
        21: "white-rook",
        26: "black-bishop",
        31: "white-bishop",
        36: "black-knight",
        41: "white-knight",
        46: "black-queen",
        51: "white-queen",
        56: "black-king",
        61: "white-king",
    }

    for k, u in unique_images.items():
        print(categories.get(k), len(u))

    for i in m:
        id = i.get("id")
        i.update({"count": count.get(id)})

    df = pd.DataFrame(m)
    for idx, (name, count) in df.loc[:, ["name", "count"]].iterrows():
        print(name, count)
    # df2 = pd.DataFrame(data)
    # df2.set_index("id", inplace=True)

    # g = df2.groupby(["image", "category"])

    # print(g.sum())

    # # ruff: noqa: F841
    # figure = plt.figure(figsize=(8, 8))

    # ax = sns.barplot(df, x="name", y="count", palette="tab10")

    # ax.set_ylabel("Count", labelpad=12)
    # ax.set_xlabel("Class", labelpad=12)
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=50, ha="right")
    # ax.set_title("Class distribution")

    # plt.savefig("classes-distribution.png")


get_class_distribution()
