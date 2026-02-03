"""
Simplified plotting: only show linear force (top) and torque (bottom).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

from column_def_csv import ColumnDefinitionCSVFile as c_def
from column_def_csv import Robot_Attribute as rob_att


try:
    filename = sys.argv[1]
    if not filename.endswith(".csv"):
        filename += ".csv"
except Exception:
    raise Exception("filename must be given, e.g. python plot_only_force.py test.csv")

# default sample time
deltatime = 0.01

# Load CSV (skip commented metadata lines)
df_all = pd.read_csv(filename, comment='#')
# downsample as original script did
df = df_all.iloc[::10].reset_index(drop=True)

# optional filtering modes preserved
if len(sys.argv) > 2:
    if sys.argv[2] == "fila":
        df = df[(df[c_def.numInsertion[0]] == 2)]
    elif sys.argv[2] == "fila-diff":
        first_iter = df[c_def.numInsertion[0]].min()
        last_iter = df[c_def.numInsertion[0]].max()
        first_rows = df[df[c_def.numInsertion[0]] == first_iter].reset_index(drop=True)
        last_rows = df[df[c_def.numInsertion[0]] == last_iter].reset_index(drop=True)
        cols_to_diff = [col for col in df.columns if col != c_def.numInsertion[0]]
        df = last_rows[cols_to_diff] - first_rows[cols_to_diff]

# time axis
time_array = np.arange(0, len(df) * deltatime, deltatime)
if len(time_array) > len(df):
    time_array = time_array[:len(df)]
elif len(time_array) < len(df):
    time_array = np.linspace(0, (len(df)-1)*deltatime, len(df))

# get tcp force columns and labels
tcp_force = c_def.mapping_new[rob_att.force.name][0]
names = c_def.mapping_new[rob_att.force.name][1]

# split into linear force and torque (assume first 3 = force, next 3 = torque)
if len(tcp_force) >= 6:
    linear_cols = tcp_force[:3]
    linear_names = names[:3]
    torque_cols = tcp_force[3:6]
    torque_names = names[3:6]
else:
    # fallback: split in half
    half = len(tcp_force) // 2
    linear_cols = tcp_force[:half]
    linear_names = names[:half]
    torque_cols = tcp_force[half:]
    torque_names = names[half:]

# plotting
fig, (ax_f, ax_t) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# plot linear force
plotted = False
for col, lab in zip(linear_cols, linear_names):
    if col not in df.columns:
        print(f"Warning: missing column '{col}' for linear force — skipping")
        continue
    ax_f.plot(time_array, df[col], label=lab)
    plotted = True
if not plotted:
    ax_f.text(0.5, 0.5, 'No linear force columns found', ha='center', va='center')
ax_f.set_title('Linear Force')
ax_f.set_ylabel(c_def.mapping_new[rob_att.force.name][2] if len(c_def.mapping_new[rob_att.force.name])>2 else 'Force')
ax_f.legend(fontsize=8)

# plot torque
plotted_t = False
for col, lab in zip(torque_cols, torque_names):
    if col not in df.columns:
        print(f"Warning: missing column '{col}' for torque — skipping")
        continue
    ax_t.plot(time_array, df[col], label=lab)
    plotted_t = True
if not plotted_t:
    ax_t.text(0.5, 0.5, 'No torque columns found', ha='center', va='center')
ax_t.set_title('Torque')
ax_t.set_ylabel('Torque')
ax_t.set_xlabel('Time (s)')
ax_t.legend(fontsize=8)

plt.tight_layout()
plt.show()
