#source ./.venv/bin/activate

#[0.12950492847737058, -0.43499271842346005, 0.3925542462754197, 1.273850784616154, -2.7994716520205136, 0.032368344188906606]
# joint angles, joint velocities, joint torques
# power/current consumption of each joint and total robot
# TCP position, orientation, velocity, angular velocity
# Acceleration data for the endeffector
import rtde_control
import rtde_receive
import spatialmath as sm
from scipy.spatial.transform import Rotation as R
import threading
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from datetime import datetime
import tqdm
import json
import sys
rtde_c = rtde_control.RTDEControlInterface("192.168.1.11")
rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.11")

from column_def_csv import ColumnDefinitionCSVFile as c_def
from column_def_csv import Robot_Attribute as rob_att

from center_tool import center_tool

NUM_TOTAL_INSERTIONS = 6000

x_deviation = 0.0
y_deviation = 0.0
# 1.0 1.9 2.8 3.7
movement_done = False
insertion = 1
data = []
deltatime = 0.001 #in seconds

# deviation_position = [grid_points[idx][0], grid_points[idx][1], 0.0]  # x, y, z deviation in meters
deviation_position = [0.0, 2.2, 0.0] # in mm
deviation_orientation = [0.0, 0.0, 0.0]     # rx, ry, rz deviation in degrees
deviation_position = [x / 1000.0 for x in deviation_position]
deviation_orientation = [x / 1000.0 for x in deviation_orientation]

# -2, -1, 0, 1, 2
# -3.5, -1.75, 0, 1.75, 3.5
exceeded_force_flag = False
calibrated_zero_position = np.zeros(6)


def listen_robot_data():
    while not movement_done:
        temp_data = np.array([insertion]) # number of current insertion [0]
        force = np.array(rtde_r.getActualTCPForce())
        actual_pose = np.array(rtde_r.getActualTCPPose())
        R_6x6 = np.zeros((6,6))
        R_6x6[:3,:3] = np.array(sm.SE3.Ry(180,'deg') * sm.SE3.EulerVec(actual_pose[3:]))[:3,:3]
        R_6x6[3:,3:] = np.array(sm.SE3.Ry(180,'deg') * sm.SE3.EulerVec(actual_pose[3:]))[:3,:3]
        force = (R_6x6 @ force.T).T
        temp_data = np.concatenate((temp_data, force)) # TCP Force [1:7]
        temp_data = np.concatenate((temp_data, actual_pose)) # TCP Pose [7:13]
        temp_data = np.concatenate((temp_data, rtde_r.getActualTCPSpeed())) # TCP Velocity [13:19]
        temp_data = np.concatenate((temp_data, rtde_r.getActualQ())) # Joint Angles [19:25]
        temp_data = np.concatenate((temp_data, rtde_r.getActualQd())) # Joint Velocity [25:31]
        temp_data = np.concatenate((temp_data, rtde_r.getActualToolAccelerometer())) # Tool Accelerometer [31:34]
        temp_data = np.concatenate((temp_data, [rtde_r.getActualRobotCurrent()])) # Current consumption of whole robot [34]
        temp_data = np.concatenate((temp_data, [rtde_r.getActualRobotVoltage()])) # Voltage consumption of whole robot [35]
        temp_data = np.concatenate((temp_data, rtde_r.getActualCurrent())) # Current consumption of each joint [36:42]
        temp_data = np.concatenate((temp_data, rtde_r.getJointTemperatures())) # Temperature of each joint [42:48]

        data.append(temp_data)
        time.sleep(deltatime)
        # print(actual_force)

def move_world_frame(desired_pose, vel = 0.2, acc = 0.3, asy = False):
    world_origin_tf = sm.SE3(np.append(calibrated_zero_position[:2], 0.05)) * sm.SE3.EulerVec(calibrated_zero_position[3:]) * sm.SE3.Ry(180,'deg')
    # world_origin_tf = sm.SE3(0.31675,-0.32044,0.05) * sm.SE3.EulerVec([1.20284,-2.90212,0.0]) * sm.SE3.Ry(180,'deg')
    togoto = world_origin_tf * desired_pose
    togoto = togoto * sm.SE3.Ry(-180,'deg') 
    target = np.concatenate((togoto.t,togoto.eulervec()))
    return rtde_c.moveL(target, vel, acc, asynchronous = asy) # TODO: do asynch true to be able to stop it

def save_data_to_csv(f_data, idx):
    column_names = (
        c_def.numInsertion +
        c_def.mapping_new[rob_att.force.name][0] +
        c_def.mapping_new[rob_att.tcp_pose_position.name][0]  + c_def.mapping_new[rob_att.tcp_pose_orientation.name][0]  + 
        c_def.mapping_new[rob_att.tcp_velocity_position.name][0]  + c_def.mapping_new[rob_att.tcp_velocity_orientation.name][0]  +
        c_def.mapping_new[rob_att.joint_angles.name][0]  + c_def.mapping_new[rob_att.joint_velocity.name][0]  +
        c_def.mapping_new[rob_att.tool_accelerometer.name][0]  +
        c_def.mapping_new[rob_att.current_whole.name][0]  +
        c_def.mapping_new[rob_att.voltage_whole.name][0]  +
        c_def.mapping_new[rob_att.current_joints.name][0]  +
        c_def.mapping_new[rob_att.temp_joints.name][0] 
    )

    f_data_numpy = np.array(f_data)
    print(f_data_numpy.shape)
    df = pd.DataFrame(f_data_numpy, columns=column_names)

    timestamp = datetime.now()
    timestamp_str = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    # Prepare filenames
    basename = f"wear_down_{idx}_{timestamp_str}"
    csv_filename = basename + ".csv"
    json_filename = basename + ".json"

    # 1) Save CSV (data only)
    df.to_csv(csv_filename, index=False)
    print("saved file:", csv_filename)

    # 2) Collect metadata
    metadata = {
        "saved_at": timestamp.isoformat(),
        "deltatime": deltatime,
        "deviation_position": deviation_position,
        "deviation_orientation": deviation_orientation,
        "num_total_insertions": NUM_TOTAL_INSERTIONS
    }

    # 3) Dump metadata to JSON
    with open(json_filename, "w") as jf:
        json.dump(metadata, jf, indent=2)
    print("saved metadata:", json_filename)



desired_pose = sm.SE3(deviation_position[0], deviation_position[1],-0.06 + deviation_position[2]) * sm.SE3.Rx(deviation_orientation[0],'deg') * sm.SE3.Ry(deviation_orientation[1],'deg') * sm.SE3.Rz(deviation_orientation[2],'deg') 
zero_pose = sm.SE3(deviation_position[0], deviation_position[1], deviation_position[2]) * sm.SE3.Rx(deviation_orientation[0],'deg') * sm.SE3.Ry(deviation_orientation[1],'deg') * sm.SE3.Rz(deviation_orientation[2],'deg') 

# move_world_frame(zero_pose)
print("devi translation", deviation_position)
print("devi orientation", deviation_orientation)
print("calibrating...")
center_tool(rtde_r, rtde_c)
calibrated_zero_position = center_tool(rtde_r, rtde_c, velocity=0.007)
print("moving to zero pose")
move_world_frame(zero_pose)
print("starting recording")

# exit()

thread1 = threading.Thread(target=listen_robot_data)
thread1.start()
# rtde_c.zeroFtSensor()
filenum = 1
for i in tqdm.tqdm(range(NUM_TOTAL_INSERTIONS), desc="Insertion"):
    rtde_c.zeroFtSensor()
    move_world_frame(desired_pose)
    # while True:
    #     if not rtde_c.getAsyncOperationProgressEx().isAsyncOperationRunning():
    #         break
    #     if exceeded_force_flag:
    #         rtde_c.stopL()
    #         print("force was exceeded")
    #         break
    move_world_frame(zero_pose)
    insertion += 1

    if (i+1) % 200 == 0:
        save_data_to_csv(data, filenum)
        data = []
        filenum += 1

# time.sleep(10)

movement_done = True

thread1.join()

