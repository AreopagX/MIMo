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
pain_v2_train_dfs = read_dfs(root_dir, "pain_v2", 3, "train_info.csv", "train")
pain_v3_train_dfs = read_dfs(root_dir, "pain_v3", 3, "train_info.csv", "train")

"""
baseline_test_dfs = [
    read_df(root_dir / suffix, "test_info.csv", "test", "baseline") for suffix in
    ["selfbody_0", "selfbody_1", "selfbody_2"]
]
baseline_test_df = pd.concat(baseline_test_dfs)
pain_test_dfs = [
read_df(root_dir / suffix, "test_info.csv", "test", "pain") for suffix in
    ["pain_0", "pain_1", "pain_2"]
]
pain_test_df = pd.concat(pain_test_dfs)"""

#train_df = pd.merge(baseline_train_df, pain_train_df, on="global_step")
train_df = pd.concat([baseline_train_df, pain_v1_train_dfs, pain_v2_train_dfs, pain_v3_train_dfs])
train_df.set_index("global_step")
train_df["global_step_1k"] = train_df["global_step"] // 1000
train_df["global_step_10k"] = train_df["global_step"] // 10000

"""test_df = pd.concat([baseline_test_df, pain_test_df])
test_df.set_index("global_step")
test_df["global_step_1k"] = test_df["global_step"] // 1000"""

# Wie entwickelt sich das Schmerzniveau über das Training?
"""sns.lineplot(train_df, x="global_step_10k", y="touch.max", hue="method")
plt.xlabel("Time (10k steps)")
plt.ylabel("Avg. Pain Levels")
plt.title("Stage: Training")
plt.show()"""

# Wie entwickelt sich die Erfolgsrate über das Training?
"""train_df["success"] = train_df["reward"].ge(100)
tmp_df = train_df.groupby(["method", "global_step_10k"])["success"].sum().reset_index(name="c")
#sns.scatterplot(tmp_df, x="global_step_10k", y="c", hue="method")
sns.regplot(tmp_df[tmp_df["method"] == "pain"], x="global_step_10k", y="c", label="pain")
sns.regplot(tmp_df[tmp_df["method"] == "baseline"], x="global_step_10k", y="c", label="baseline")
plt.xlabel("Time (10k steps)")
plt.ylabel("Num. of successful episodes per time interval")
plt.title("Stage: Training")
plt.legend()
plt.show()"""
#df["test.successfull_attempts"] = data["test.reward"].ge(300).groupby(data.index // N).sum()

#? Trägt das Schmerzempfinden zu MIMos Erfolg (Reward) bei?
"""tmp_df = train_df[train_df["method"] != "baseline"].groupby(["method", "global_step_10k"])["reward.without_pain"].mean().reset_index(name="avg_reward")
sns.regplot(tmp_df[tmp_df["method"] == "pain_v1"], x="global_step_10k", y="avg_reward", label="pain_v1")
sns.regplot(tmp_df[tmp_df["method"] == "pain_v2"], x="global_step_10k", y="avg_reward", label="pain_v2")
sns.regplot(tmp_df[tmp_df["method"] == "pain_v3"], x="global_step_10k", y="avg_reward", label="pain_v3")
tmp_df = train_df[train_df["method"] == "baseline"].groupby(["global_step_10k"])["reward"].mean().reset_index(name="avg_reward")
sns.regplot(tmp_df, x="global_step_10k", y="avg_reward", label="baseline")
plt.xlabel("Time (10k steps)")
plt.ylabel("Avg. Reward per Episode")
plt.title("Stage: Training")
plt.legend()
plt.show()"""

"""tmp_df = test_df.groupby(["method", "global_step_1k"])["reward"].mean().reset_index(name="avg_reward")
sns.lineplot(tmp_df, x="global_step_1k", y="avg_reward", hue="method")
plt.xlabel("Time (1k steps)")
plt.ylabel("Avg. Reward per Episode")
plt.title("Stage: Test")
plt.legend()
plt.show()"""

"""sns.lineplot(train_df[train_df["method"] == "pain"], x="global_step_10k", y="reward.pain_penalty", hue="method")
plt.xlabel("Time (10k steps)")
plt.ylabel("Pain Penalty")
plt.title("Stage: Training")
plt.legend()
plt.show()"""


"""pain_df = train_df[train_df["method"] == "pain"]
sns.histplot(pain_df[pain_df["reward.pain_penalty"] > 1.0], x="reward.pain_penalty")
plt.show()"""

# Handelt MIMo mit Schmerzempfinden vorsichtiger?

# Reduziert MIMo die Länge der Kontakte?
print()
df = train_df[train_df["method"] == "baseline"]
arr = df["touch.avg"].to_numpy()
touches = np.where(arr)[0]
arr[touches[:-1]] = np.diff(touches)
np.histogram(np.diff(touches))
plt.hist(np.diff(touches))
plt.yscale("log")
plt.show()
print()

