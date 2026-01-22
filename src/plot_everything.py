# console argument for only plotting first and last insertion: fila
# console argument for plotting the difference from first and last insertion: fila-diff
# 7 graphs in total: 6 for each dof and one for boxplot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
import ast
import sys
import json

from column_def_csv import ColumnDefinitionCSVFile as c_def
from column_def_csv import Robot_Attribute as rob_att

try:
    filename = sys.argv[1]
    if not filename.endswith(".csv"):
        filename += ".csv"
except:
    raise Exception("filename must be given")

# Load metadata from corresponding .json file
# json_filename = filename.replace(".csv", ".json")
# try:
#     with open(json_filename, "r") as jf:
#         metadata = json.load(jf)
# except FileNotFoundError:
#     raise Exception(f"Metadata file not found: {json_filename}")
# except json.JSONDecodeError as e:
#     raise Exception(f"Error decoding JSON file: {e}")

# # Extract and convert required metadata values
# try:
#     deltatime = float(metadata.get("deltatime", 0.01))
# except ValueError:
deltatime = 0.01


# ------------------------
# Load CSV Data (skip metadata lines)
# ------------------------
df_all = pd.read_csv(filename, comment='#')

df = df_all.iloc[::10]

# Optionally, reset the index
df.reset_index(drop=True, inplace=True)


if len(sys.argv) > 2:
    if sys.argv[2] == "fila":
        df = df[(df[c_def.numInsertion[0]] == 2)]
    elif sys.argv[2] == "fila-diff":
        first_iter = df[c_def.numInsertion[0]].min()
        last_iter = df[c_def.numInsertion[0]].max()
        first_rows = df[df[c_def.numInsertion[0]] == first_iter]
        last_rows = df[df[c_def.numInsertion[0]] == last_iter]
        first_rows = first_rows.reset_index(drop=True)
        last_rows = last_rows.reset_index(drop=True)
        cols_to_diff = [col for col in df.columns if col != c_def.numInsertion[0]]
        df = last_rows[cols_to_diff] - first_rows[cols_to_diff]

# Compute time axis based on deltatime
time_array = np.arange(0, len(df) * deltatime, deltatime)
if len(time_array) > len(df):
    time_array = time_array[:len(df)]
elif len(time_array) < len(df):
    time_array = np.linspace(0, (len(df)-1)*deltatime, len(df))

# ------------------------
# Define Column Groups (as saved previously)
# ------------------------

tcp_force = c_def.mapping_new[rob_att.force.name][0]
tcp_pose_position = c_def.mapping_new[rob_att.tcp_pose_position.name][0]
tcp_pose_orientation = c_def.mapping_new[rob_att.tcp_pose_orientation.name][0]

tcp_velocity_position = c_def.mapping_new[rob_att.tcp_velocity_position.name][0]
tcp_velocity_orientation = c_def.mapping_new[rob_att.tcp_velocity_orientation.name][0]
# Tool Accelerometer remains unchanged (now placed at (0,2))
tool_accelerometer = c_def.mapping_new[rob_att.tool_accelerometer.name][0]
# Joint Angles
joint_angles = c_def.mapping_new[rob_att.joint_angles.name][0]
# Joint Velocity
joint_velocity = c_def.mapping_new[rob_att.joint_velocity.name][0]
# Current and Voltage Consumption:
current_whole = c_def.mapping_new[rob_att.current_whole.name][0]
voltage_whole = c_def.mapping_new[rob_att.voltage_whole.name][0]
current_joints = c_def.mapping_new[rob_att.current_joints.name][0]
# Joint Temperature
temp_joints = c_def.mapping_new[rob_att.temp_joints.name][0]

# ------------------------
# Create Plot using GridSpec for Nested Subplots
# ------------------------
fig = plt.figure(figsize=(22, 12))
main_gs = gridspec.GridSpec(2, 4, figure=fig, wspace=0.4, hspace=0.5)

# Overall title including deviation info
fig.suptitle(
    f"Insertion Data Visualization of {filename}",
    fontsize=16
)

# (0,0) TCP Force
ax00 = fig.add_subplot(main_gs[0, 0])
names = c_def.mapping_new[rob_att.force.name][1]
print(names)
for col in range(len(tcp_force)):
    ax00.plot(time_array, df[tcp_force[col]], label=names[col])
ax00.set_title(c_def.mapping_new[rob_att.force.name][3])
ax00.set_xlabel("Time (s)")
ax00.set_ylabel(c_def.mapping_new[rob_att.force.name][2])
ax00.legend(fontsize=8)

# (0,1) TCP Pose -> Split into two vertical subplots
gs_pose = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=main_gs[0, 1], hspace=0.3)
ax_pose_top = fig.add_subplot(gs_pose[0, 0])
names = c_def.mapping_new[rob_att.tcp_pose_position.name][1]
for col in range(len(tcp_pose_position)):
    ax_pose_top.plot(time_array, df[tcp_pose_position[col]], label=names[col])
ax_pose_top.set_title(c_def.mapping_new[rob_att.tcp_pose_position.name][3])
ax_pose_top.set_ylabel(c_def.mapping_new[rob_att.tcp_pose_position.name][2])  # adjust label as needed
ax_pose_top.legend(fontsize=8)

ax_pose_bot = fig.add_subplot(gs_pose[1, 0])
names = c_def.mapping_new[rob_att.tcp_pose_orientation.name][1]
for col in range(len(tcp_pose_orientation)):
    ax_pose_bot.plot(time_array, df[tcp_pose_orientation[col]], label=names[col])
ax_pose_bot.set_title(c_def.mapping_new[rob_att.tcp_pose_orientation.name][3])
ax_pose_bot.set_xlabel("Time (s)")
ax_pose_bot.set_ylabel(c_def.mapping_new[rob_att.tcp_pose_orientation.name][2])  # adjust label as needed
ax_pose_bot.legend(fontsize=8)

# (0,3) TCP Velocity -> Split into two vertical subplots
gs_velocity = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=main_gs[0, 2], hspace=0.3)
ax_vel_top = fig.add_subplot(gs_velocity[0, 0])
names = c_def.mapping_new[rob_att.tcp_velocity_position.name][1]
for col in range(len(tcp_velocity_position)):
    ax_vel_top.plot(time_array, df[tcp_velocity_position[col]], label=names[col])
ax_vel_top.set_title(c_def.mapping_new[rob_att.tcp_velocity_position.name][3])
ax_vel_top.set_ylabel(c_def.mapping_new[rob_att.tcp_velocity_position.name][2])
ax_vel_top.legend(fontsize=8)

ax_vel_bot = fig.add_subplot(gs_velocity[1, 0])
names = c_def.mapping_new[rob_att.tcp_velocity_orientation.name][1]
for col in tcp_velocity_orientation:
    ax_vel_bot.plot(time_array, df[col], label=col)
ax_vel_bot.set_title(c_def.mapping_new[rob_att.tcp_velocity_orientation.name][3])
ax_vel_bot.set_xlabel("Time (s)")
ax_vel_bot.set_ylabel(c_def.mapping_new[rob_att.tcp_velocity_orientation.name][2])
ax_vel_bot.legend(fontsize=8)

# (0,2) Tool Accelerometer (unchanged)
ax02 = fig.add_subplot(main_gs[0, 3])
names = c_def.mapping_new[rob_att.tool_accelerometer.name][1]
for col in range(len(tool_accelerometer)):
    ax02.plot(time_array, df[tool_accelerometer[col]], label=names[col])
ax02.set_title(c_def.mapping_new[rob_att.tool_accelerometer.name][3])
ax02.set_xlabel("Time (s)")
ax02.set_ylabel(c_def.mapping_new[rob_att.tool_accelerometer.name][2])
ax02.legend(fontsize=8)

# (1,0) Joint Angles
ax10 = fig.add_subplot(main_gs[1, 0])
names = c_def.mapping_new[rob_att.joint_angles.name][1]
for col in range(len(joint_angles)):
    ax10.plot(time_array, df[joint_angles[col]], label=names[col])
ax10.set_title(c_def.mapping_new[rob_att.joint_angles.name][3])
ax10.set_xlabel("Time (s)")
ax10.set_ylabel(c_def.mapping_new[rob_att.joint_angles.name][2])
ax10.legend(fontsize=8)

# (1,1) Joint Velocity
ax11 = fig.add_subplot(main_gs[1, 1])
names = c_def.mapping_new[rob_att.joint_velocity.name][1]
for col in range(len(joint_velocity)):
    ax11.plot(time_array, df[joint_velocity[col]], label=names[col])
ax11.set_title(c_def.mapping_new[rob_att.joint_velocity.name][3])
ax11.set_xlabel("Time (s)")
ax11.set_ylabel(c_def.mapping_new[rob_att.joint_velocity.name][2])
ax11.legend(fontsize=8)

# (1,2) Current & Voltage Consumption -> Split vertically:
gs_cons = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=main_gs[1, 2], hspace=0.3)
# Top subplot: Voltage consumption only
ax_cons_top = fig.add_subplot(gs_cons[0, 0])
ax_cons_top.plot(time_array, df[voltage_whole], linewidth=2)
ax_cons_top.set_title(c_def.mapping_new[rob_att.voltage_whole.name][3])
ax_cons_top.set_ylabel(c_def.mapping_new[rob_att.joint_velocity.name][2])
ax_cons_top.legend(fontsize=8)

# Bottom subplot: Current consumption (whole robot and each joint) with continuous lines
ax_cons_bot = fig.add_subplot(gs_cons[1, 0])
ax_cons_bot.plot(time_array, df[current_whole], label=c_def.mapping_new[rob_att.current_whole.name][1], linewidth=2)
names = c_def.mapping_new[rob_att.current_joints.name][1]
for col in range(len(current_joints)):
    ax_cons_bot.plot(time_array, df[current_joints[col]], label=names[col])  # continuous lines (default style)
ax_cons_bot.set_title(c_def.mapping_new[rob_att.current_whole.name][3])
ax_cons_bot.set_xlabel("Time (s)")
ax_cons_bot.set_ylabel(c_def.mapping_new[rob_att.current_joints.name][2])
ax_cons_bot.legend(fontsize=8)

# (1,3) Joint Temperature
ax13 = fig.add_subplot(main_gs[1, 3])
names = c_def.mapping_new[rob_att.temp_joints.name][1]
for col in range(len(temp_joints)):
    ax13.plot(time_array, df[temp_joints[col]], label=names[col])
ax13.set_title(c_def.mapping_new[rob_att.temp_joints.name][3])
ax13.set_xlabel("Time (s)")
ax13.set_ylabel(c_def.mapping_new[rob_att.temp_joints.name][2])
ax13.legend(fontsize=8)

plt.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.05, wspace=0.3, hspace=0.2)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
