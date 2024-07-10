from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import scipy.stats

def read_df(dir, filename, stage, name):
    df = pd.read_csv(dir / filename)
    df.rename(columns={col: col.replace(f"{stage}.", "") for col in df.columns},
                             inplace=True)
    #df.rename(columns={f"{name}.global_step": "global_step"}, inplace=True)
    df["method"] = name
    return df

def read_dfs(root_dir, name, versions, filename, stage):
    return pd.concat([
        read_df(root_dir / f"{name}_{i}", filename, stage, name)
        for i in range(versions)
    ])

root_dir = Path("/home/dustin/Documents/Uni/GU/Thesis/MIMo/models")
baseline_dir = root_dir / "selfbody"
pain_dir = root_dir / "pain"

baseline_train_df = read_dfs(root_dir, "baseline", 3, "train_info.csv", "train")

pain_v1_train_dfs = read_dfs(root_dir, "pain_v1", 3, "train_info.csv", "train")
#pain_v2_train_dfs = read_dfs(root_dir, "pain_v2", 3, "train_info.csv", "train")
#pain_v3_train_dfs = read_dfs(root_dir, "pain_v3", 3, "train_info.csv", "train")

train_df = pd.concat([baseline_train_df, pain_v1_train_dfs])# , pain_v2_train_dfs, pain_v3_train_dfs
train_df.set_index("global_step")
train_df["global_step_1k"] = train_df["global_step"] // 1000
train_df["global_step_10k"] = train_df["global_step"] // 10000

"""sns.lineplot(train_df, x="global_step", y="touch.max", hue="method")
plt.xlabel("Steps (1k)")
plt.ylabel("Avg. Pain Level")
plt.title("Stage: Training")
plt.legend()
plt.show()"""

print()
baseline_touch = train_df[train_df["method"] == "baseline"].groupby("global_step_1k")["touch.max"].apply(lambda x: x.values)
pain_v1_touch = train_df[train_df["method"] == "pain_v1"].groupby("global_step_1k")["touch.max"].apply(lambda x: x.values)

"""plt.plot(np.arange(1000), [baseline_touch[idx].std()**2 for idx in range(1000)], label="baseline")
plt.plot(np.arange(1000), [pain_v1_touch[idx].std()**2 for idx in range(1000)], label="pain_v1")
plt.xlabel("Steps (1k)")
plt.ylabel("Avg. Pain Level (Variance)")
plt.title("Stage: Training")
plt.legend()
plt.show()"""

"""bt = baseline_touch = train_df[train_df["method"] == "baseline"]["touch.max"]
pt = train_df[train_df["method"] == "pain_v1"]["touch.max"]

results = []
res = scipy.stats.ttest_ind(bt, pt, equal_var=False, alternative="greater")
results.append(res)"""

results = []
for idx in range(len(baseline_touch)):
    res = scipy.stats.ttest_ind(baseline_touch[idx], pain_v1_touch[idx], equal_var=False, alternative="greater")
    results.append(res)
dfs = [x.df for x in results]
pvals = [x.pvalue for x in results]
t0 = [x[0] for x in results]
steps = np.arange(len(dfs))
#plt.plot(steps, dfs, label="df")
plt.scatter(steps, pvals, label="pain_v1", marker=".")
plt.plot(steps, np.ones_like(steps) * 0.01, label="confidence threshold", c="red")
plt.title("Stage: Training; H0: similar pain, H1: baseline with greater pain levels")
plt.ylabel("pvalue")
plt.xlabel("Steps (1k)")
#plt.plot(steps, t0, label="t0")
plt.legend()
plt.show()