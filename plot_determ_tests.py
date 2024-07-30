from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def read_df(dir, filename, stage, name):
    df.rename(columns={col: col.replace(f"{stage}.", "") for col in df.columns},
              inplace=True)
    #df.rename(columns={f"{name}.global_step": "global_step"}, inplace=True)
    df["method"] = name
    return df


def read_dfs(root_dir, name, c_max=11, c_step=1):
    dfs = []
    for v in range(3):
        for c in range(1, c_max, c_step):
            df = pd.read_csv(root_dir / f"{name}_trial_{v}_ckpt_{c}.csv")
            df["method"] = name
            df["version"] = v
            df["ckpt"] = c
            dfs.append(df)
    return pd.concat(dfs)


#root_dir = Path("/home/dustin/Documents/Uni/GU/Thesis/MIMo/models/cw28/eval_v4_random_qpos_0_1")
root_dir = Path("/home/dustin/Documents/Uni/GU/Thesis/MIMo/models/retrained_new_ckpts/eval")#, c_max=21, c_step=2)
baseline_df = read_dfs(root_dir, "baseline")
pain_v1_df = read_dfs(root_dir, "pain")

df = pd.concat([baseline_df, pain_v1_df])
df.set_index("global_step")
df["global_step_1k"] = df["global_step"] // 1000
df["global_step_100"] = df["global_step"] // 100
df["global_step_10"] = df["global_step"] // 10

# Wie schwierig sind die einzelnen Tasks?
"""tmp_df = df.groupby(["method", "ckpt", "target_name", "version", "seed"])["global_step"].mean().reset_index(name="duration")
sns.scatterplot(tmp_df, x="target_name", y="duration", hue="method")
plt.xlabel("Target Body")
plt.xticks(rotation=45)
plt.ylabel("Episode Duration")
plt.title("Stage: Testing")
plt.legend()
plt.savefig(root_dir / "_eval_duration_per_target.pdf", bbox_inches="tight")
plt.close()"""

tmp_df = df.groupby(["method", "ckpt", "target_index", "version", "seed"])["global_step"].max().reset_index(name="duration")
sns.displot(tmp_df, x="duration", hue="method", kind="kde", clip=(0, 500))
plt.yscale("log")
plt.xlabel("Episode Duration")
plt.ylabel("Density")
plt.title("Stage: Testing")
plt.savefig(root_dir / "_eval_duration_distplot.pdf", bbox_inches="tight")
plt.close()

tmp_df = df.groupby(["method", "ckpt", "target_index", "version", "seed"])["global_step"].max().reset_index(
    name="duration")
sns.lineplot(tmp_df, x="ckpt", y="duration", hue="method", err_style="bars", errorbar=("se", 1))
plt.xlabel("Checkpoint (@100k Steps)")
plt.ylabel("Density")
plt.title("Stage: Testing")
plt.legend()
plt.savefig(root_dir / "_eval_duration_lineplot.pdf", bbox_inches="tight")
plt.close()

# --> Pain V1 braucht länger

# Handelt MIMo mit Schmerzempfinden vorsichtiger?
sns.lineplot(df, x="ckpt", y="qvel", hue="method", err_style="bars", errorbar=("se", 1))
plt.xlabel("Checkpoint (@100k steps)")
plt.ylabel("Avg. Velocity")
plt.title("Stage: Testing")
plt.legend()
plt.savefig(root_dir / "_eval_avg_vel_checkpoint.pdf", bbox_inches="tight")
plt.close()

sns.lineplot(df, x="global_step", y="qvel", hue="method", err_style="bars", errorbar=("se", 1))
plt.xlabel("Step")
plt.ylabel("Avg. Velocity")
plt.title("Stage: Testing")
plt.legend()
plt.savefig(root_dir / "_eval_avg_vel_step.pdf", bbox_inches="tight")
plt.close()

right_arm_velocities = [
    "qvel_right_shoulder_horizontal", "qvel_right_shoulder_ad_ab",
    "qvel_right_shoulder_rotation", "qvel_right_elbow",
    "qvel_right_hand1", "qvel_right_hand2", "qvel_right_hand3",
    "qvel_right_fingers"
]

for vel in right_arm_velocities:
    sns.lineplot(df, x="global_step", y=vel, hue="method")
    plt.xlabel("Step")
    # plt.ylabel("Joint Velocity")
    plt.legend()
    plt.savefig(root_dir / f"_eval_{vel}.pdf", bbox_inches="tight")
    plt.close()

# --> Im Schnitt langsamere Bewegungen mit Pain V1


# Wie entwickelt sich das Schmerzniveau über das Training?
sns.lineplot(df, x="ckpt", y="touch.max", hue="method", err_style="bars", errorbar=("se", 1))
plt.xlabel("Checkpoint (@100k steps)")
plt.ylabel("Avg. Pain Levels")
plt.title("Stage: Testing")
plt.savefig(root_dir / "_eval_avg_pain.pdf", bbox_inches="tight")
plt.close()

# --> Niedrigeres Schmerzniveau

# Wie entwickelt sich die Erfolgsrate über das Training?

tmp_df = df.groupby(["method", "ckpt", "target_index", "version", "seed"])[["global_step", "success"]].max().reset_index()
sns.lineplot(tmp_df, x="ckpt", y="success", hue="method", err_style="bars", errorbar=("se", 1))
plt.xlabel("Checkpoint (@100k steps)")
plt.ylabel("Success Rate")
plt.title("Stage: Testing")
plt.legend()
plt.savefig(root_dir / "_eval_success_rate.pdf", bbox_inches="tight")
plt.close()

tmp_df = df.groupby(["method", "ckpt", "target_name", "version", "seed"])[["global_step", "success"]].max().reset_index()
sns.lineplot(tmp_df, x="target_name", y="global_step", hue="method", err_style="bars", errorbar=("se", 1))
plt.xlabel("Target Body")
plt.xticks(rotation=45)
plt.ylabel("Success Rate")
plt.title("Stage: Testing")
plt.legend()
plt.savefig(root_dir / "_eval_success_rate_per_target.pdf", bbox_inches="tight")
plt.close()

#? Trägt das Schmerzempfinden zu MIMos Erfolg (Reward) bei?
sns.lineplot(df, x="ckpt", y="reward.without_pain", hue="method")
plt.xlabel("Checkpoint (@100k steps)")
plt.ylabel("Avg. Reward per Episode")
plt.title("Stage: Testing")
plt.legend()
plt.savefig(root_dir / "_eval_avg_reward.pdf", bbox_inches="tight")
plt.close()

# Reduziert MIMo die Länge der Kontakte?
"""print()
df = df[df["method"] == "baseline"]
arr = df["touch.avg"].to_numpy()
touches = np.where(arr)[0]
arr[touches[:-1]] = np.diff(touches)
np.histogram(np.diff(touches))
plt.hist(np.diff(touches))
plt.yscale("log")
plt.show()
print()"""
