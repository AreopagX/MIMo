from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def read_df(dir, filename, stage, name):
    df = pd.read_csv(dir / filename)
    df.rename(columns={col: col.replace(f"{stage}.", "") for col in df.columns},
                             inplace=True)
    #df.rename(columns={f"{name}.global_step": "global_step"}, inplace=True)
    df["method"] = name
    return df

root_dir = Path("/home/dustin/Documents/Uni/GU/Thesis/MIMo/models")

dfs = []
for method in ["pain_v1", "pain_v3", "baseline"]:
    for version in range(3):
        for ckpt in range(1, 11):
            df = pd.read_csv(root_dir / f"{method}_{version}_test_{ckpt}.csv")
            df["method"] = method
            df["version"] = version
            df["ckpt"] = ckpt
            dfs.append(df)
df = pd.concat(dfs)
df.loc[df["method"] == "baseline", "test.reward.without_pain"] = df.loc[df["method"] == "baseline", "test.reward"]
# columns: [
# 'test.target_name', 'test.target_index', 'test.touch.avg', 'test.touch.min', 'test.touch.max',
# 'test.reward', 'test.reward.without_pain', 'test.reward.pain_penalty', 'test.global_step', 'method', 'version', 'ckpt']
"""tmp_df = df.groupby(["method", "ckpt"])["test.touch.max"].mean().reset_index(name="pain")
sns.lineplot(tmp_df, x="ckpt", y="pain", hue="method")
plt.title("Stage: Test")
plt.ylabel("Pain level")
plt.xlabel("Checkpoint from step (100k steps)")
plt.show()
tmp_df = df.groupby(["method", "ckpt"])["test.qvel.abs_max"].mean().reset_index(name="vel")
sns.lineplot(tmp_df, x="ckpt", y="vel", hue="method")
plt.title("Stage: Test")
plt.ylabel("Average maximum velocity")
plt.xlabel("Checkpoint from step (100k steps)")
plt.show()"""
ax1 = plt.gca()
ax2 = plt.twinx(ax1)
tmp_df = df.groupby(["test.target_name"]).count()
sns.barplot(tmp_df, x="test.target_name", y="test.target_index", ax=ax2, label="Count", alpha=0.3)
tmp_df = df.groupby(["test.target_name", "method"])["test.reward.without_pain"].mean().reset_index(name="reward")
sns.lineplot(tmp_df, x="test.target_name", y="reward", ax=ax1, hue="method")
plt.title("Stage: Test")
ax1.set_ylabel("Average reward")
ax2.set_ylabel("Steps with target")
plt.xlabel("Target body")
plt.show()
print()