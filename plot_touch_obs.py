import numpy as np
import pandas as pd

from mimoEnv.envs.fall import TOUCH_PARAMS
from mimoEnv.envs.selfbodypain import PAIN_PARAMS
import matplotlib.pyplot as plt
import seaborn.objects as so
import seaborn as sns


body_to_plot_idx = {
    "head": 0,
    "left_eye": 0,
    "right_eye": 0,
    "hip": 1,
    "lower_body": 1,
    "upper_body": 1,
    "left_fingers": 2,
    "left_hand": 2,
    "left_lower_arm": 2,
    "left_upper_arm": 2,
    "right_fingers": 3,
    "right_hand": 3,
    "right_lower_arm": 3,
    "right_upper_arm": 3,
    "left_foot": 4,
    "left_toes": 4,
    "left_lower_leg": 4,
    "left_upper_leg": 4,
    "right_foot": 5,
    "right_toes": 5,
    "right_lower_leg": 5,
    "right_upper_leg": 5,
}

body_names = sorted(PAIN_PARAMS["scales"].keys())
touch_obs = np.load("models/pain/touch_observations_0.npz")
#pain_obs = np.load("models/pain/pain_observations_0.npz")

print()

T = touch_obs.shape[1]

fig, axes = plt.subplots(2, 3)
axes = axes.flatten()

colors = sns.color_palette("hls", len(body_names))
for i in range(6):
    p = so.Plot(x=np.arange(T + 1)).label(x="Timestep", y="Average Force / Pain on Body Part", color="").limit(y=(0, 1000))
    for body_name in body_to_plot_idx.keys():
        if body_to_plot_idx[body_name] != i:
            continue
        if body_name not in body_names:
            continue
        idx = body_names.index(body_name)
        touch_ob = touch_obs[idx]
        p = p.add(so.Line(color=colors[idx]), y=touch_ob[:, 1], label=f"{body_name} force")

        #pain_ob = pain_obs[idx]
        #p = p.add(so.Line(color=colors[idx], alpha=0.5, linestyle="--"), y=pain_ob[:, 1], label=f"{body_name} pain")
        #p = p.add(so.Band(), ymin=touch_ob[:, 0], ymax=touch_ob[:, 2])
    p.on(axes[i]).plot()

    # hacky solution to move legend
    if len(fig.legends) > 0:
        l1 = fig.legends.pop(0)
        axes[i].legend(l1.legend_handles, [t.get_text() for t in l1.texts])

fig.show()

plt.show()
