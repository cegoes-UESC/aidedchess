import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

f = open("train-keypoints.json")

d = json.load(f)

count = {}

for a in d.get("annotations"):
    category = a.get("category_id")
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

for i in m:
    i.update({"count": count.get(i.get("id"))})

df = pd.DataFrame(m)

figure = plt.figure(figsize=(8, 8))

ax = sns.barplot(df, x="name", y="count")

ax.set_ylabel("Count", labelpad=12)
ax.set_xlabel("Class", labelpad=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=50, ha="right")
ax.set_title("Class distribution")

plt.savefig("classes-distribution.png")
