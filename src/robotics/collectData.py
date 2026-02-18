#source ./.venv/bin/activate

#[0.12950492847737058, -0.43499271842346005, 0.3925542462754197, 1.273850784616154, -2.7994716520205136, 0.032368344188906606]
# joint angles, joint velocities, joint torques
# power/current consumption of each joint and total robot
# TCP position, orientation, velocity, angular velocity
# Acceleration data for the endeffector
import random
import rtde_control
import rtde_receive
import numpy as np
import time
import spatialmath as sm
import pandas as pd
import json
from datetime import datetime
import threading
from column_def_csv import ColumnDefinitionCSVFile as c_def
from column_def_csv import Robot_Attribute as rob_att
from RobotiqHandE import RobotiqGripper
import socket

ARDUINO_IP = "192.168.4.1"
PORT = 5000

# --------------  CONFIGURATION  ----------------

folder_name = "devi01_test/"
recording = False
program_running = True
insertion = 1
num_insertions = 1
data = []
deltatime = 0.002 #in seconds
min_max_deviation = 0.0 #in meters, less devi: 0.0001, much devi: 0.001
angular_error = np.deg2rad(0.0) #in degrees
bigRodAbovePose = np.array([-0.3724, -0.3274, 0.06, 3.079, -0.625, 0.0])
bigRodGraspPose = np.array([-0.3724, -0.3274, 0.0, 3.079, -0.625, 0.0])
bigHole03AbovePose = np.array([-0.52932, -0.30131, 0.06, 3.079, -0.625, 0.0])
bigHole02AbovePose = np.array([-0.53889, -0.32441, 0.06, 3.079, -0.625, 0.0])
bigHole01AbovePose = np.array([-0.5484, -0.34798, 0.06, 3.079, -0.625, 0.0])

# -----------------------------------------------

# a failed

ur_ip = "192.168.1.11"
rtde_c = rtde_control.RTDEControlInterface(ur_ip)
rtde_r = rtde_receive.RTDEReceiveInterface(ur_ip)
gripper = RobotiqGripper()


# -2, -1, 0, 1, 2
# -3.5, -1.75, 0, 1.75, 3.5
exceeded_force_flag = False
calibrated_zero_position = np.zeros(6)

def approx_force_to_for_value(force_n: float,
                              force_min_n: float = 20.0,
                              force_max_n: float = 130.0) -> int:
    """
    Rough mapping only. Real grip force depends on speed, mechanics, object compliance, etc.
    Use experiments to calibrate if you need repeatability.
    Hand-E spec indicates ~20–130 N range. :contentReference[oaicite:3]{index=3}
    """
    force_n = max(force_min_n, min(force_n, force_max_n))
    scaled = (force_n - force_min_n) / (force_max_n - force_min_n)  # 0..1
    return int(round(scaled * 255))


def init_gripper(gripper: RobotiqGripper, ur_ip: str):
    # Typical URCap socket endpoint for Robotiq is port 63352. :contentReference[oaicite:4]{index=4}
    gripper.connect(hostname=ur_ip, port=63352, socket_timeout=2.0)

    # Activate + (optional) auto-calibrate travel range
    gripper.activate(auto_calibrate=True)

    # Open once to a known state
    gripper.move_and_wait_for_pos(gripper.get_open_position(), speed=128, force=64)


def close_until_force_limit(gripper: RobotiqGripper,
                            target_force_n: float,
                            speed: int = 128):
    """
    Closes toward fully closed. The gripper will stop when it hits an object and applies
    force up to the commanded limit (FOR). We treat this commanded limit as the "threshold".
    """
    for_value = approx_force_to_for_value(target_force_n)
    closed_pos = gripper.get_closed_position()

    final_pos, obj_status = gripper.move_and_wait_for_pos(
        position=closed_pos,
        speed=speed,
        force=for_value,
    )

    if obj_status in (
        RobotiqGripper.ObjectStatus.STOPPED_OUTER_OBJECT,
        RobotiqGripper.ObjectStatus.STOPPED_INNER_OBJECT,
    ):
        print(f"Object detected. Stopped at pos={final_pos}, force_setting(FOR)={for_value}")
        return True

    if obj_status == RobotiqGripper.ObjectStatus.AT_DEST:
        print(f"Reached target position (likely fully closed). pos={final_pos}")
        return False

    # Shouldn't usually happen if enums match firmware, but keep for safety:
    print(f"Ended with status={obj_status}, pos={final_pos}")
    return False


def listen_robot_data():
    while program_running:
        while recording:
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

def receive_loadcell_arduino():
    while True:
        try:
            print(f"Connecting to {ARDUINO_IP}:{PORT} ...")
            s = socket.create_connection((ARDUINO_IP, PORT), timeout=10.0)  # connect timeout only
            s.settimeout(None)  # blocking reads (no read timeout)
            s.sendall(b"START\n")

            print("Connected. Receiving values...")

            buf = b""
            while True:
                chunk = s.recv(4096)
                if not chunk:
                    raise ConnectionError("socket closed")
                buf += chunk

                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = int(line.decode("ascii", errors="ignore"))
                        print(raw)
                    except ValueError:
                        # e.g. HELLO
                        pass

        except (OSError, ConnectionError) as e:
            print(f"Disconnected ({e}). Reconnecting in 1s...")
            time.sleep(1)

def move_world_frame(desired_pose, vel = 0.2, acc = 0.3, asy = False):
    world_origin_tf = sm.SE3(np.append(calibrated_zero_position[:2], 0.05)) * sm.SE3.EulerVec(calibrated_zero_position[3:]) * sm.SE3.Ry(180,'deg')
    # world_origin_tf = sm.SE3(0.31675,-0.32044,0.05) * sm.SE3.EulerVec([1.20284,-2.90212,0.0]) * sm.SE3.Ry(180,'deg')
    togoto = world_origin_tf * desired_pose
    togoto = togoto * sm.SE3.Ry(-180,'deg') 
    target = np.concatenate((togoto.t,togoto.eulervec()))
    return rtde_c.moveL(target, vel, acc, asynchronous = asy) # TODO: do asynch true to be able to stop it

def save_data_to_csv(f_data, idx, deviation, angle_radius, successful_insertion, obj, holeSize):
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
    # basename = f"wear_down_{idx}_{timestamp_str}"
    basename = f"insertion_{idx}_{timestamp_str}"
    csv_filename = folder_name + basename + ".csv"
    json_filename = folder_name + basename + ".json"

    # 1) Save CSV (data only)
    df.to_csv(csv_filename, index=False)
    print("saved file:", csv_filename)

    # 2) Collect metadata
    metadata = {
        "saved_at": timestamp.isoformat(),
        "deltatime": deltatime,
        "deviation": deviation,
        "angle_radius": angle_radius,
        "successful_insertion": successful_insertion,
        "insertion_object": obj,
        "hole_size": holeSize
    }
    with open(json_filename, "w") as jf:
        json.dump(metadata, jf, indent=2)
    print("saved metadata:", json_filename)



# desired_pose = sm.SE3(deviation_position[0], deviation_position[1],-0.06 + deviation_position[2]) * sm.SE3.Rx(deviation_orientation[0],'deg') * sm.SE3.Ry(deviation_orientation[1],'deg') * sm.SE3.Rz(deviation_orientation[2],'deg') 
# zero_pose = sm.SE3(deviation_position[0], deviation_position[1], deviation_position[2]) * sm.SE3.Rx(deviation_orientation[0],'deg') * sm.SE3.Ry(deviation_orientation[1],'deg') * sm.SE3.Rz(deviation_orientation[2],'deg') 

# thread1 = threading.Thread(target=listen_robot_data)
# thread1.start()
# rtde_c.zeroFtSensor()
thread1 = threading.Thread(target=listen_robot_data)
thread2 = threading.Thread(target=receive_loadcell_arduino)

rtde_c.moveL(bigRodAbovePose)
init_gripper(gripper, ur_ip)
thread1.start()
thread2.start()

selection_vector = [1, 1, 1, 1, 1, 1]
wrench_down = [0, 0, -10, 0, 0, 0]
force_type = 2
limits = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3]

for i in range(num_insertions):
    rtde_c.zeroFtSensor()
    rtde_c.moveL(bigRodGraspPose, 0.1, 0.5)
    close_until_force_limit(gripper, target_force_n=50.0, speed=128)
    rtde_c.moveL(bigRodAbovePose, 0.1, 0.5)
    # close_until_force_limit(gripper, target_force_n=20.0, speed=128)

    # while gripper.get_current_position() > 200:
    #     print("Failed to grasp the rod properly, trying again")
    #     gripper.move_and_wait_for_pos(gripper.get_open_position(), speed=128, force=64)
    #     rtde_c.moveL(bigRodAbovePose + np.array([0, 0, -0.06, 0, 0, 0]), 0.1, 0.5)
    #     close_until_force_limit(gripper, target_force_n=20.0, speed=128)
    #     rtde_c.moveL(bigRodAbovePose, 0.1, 0.5)
    #     close_until_force_limit(gripper, target_force_n=20.0, speed=128)
        
    
    insertion = 1
    # angle = np.random.uniform(0, 2 * np.pi)  # Random angle in radians
    # radius = np.random.uniform(0, min_max_deviation)  # Random radius
    angle = 0.0
    radius = min_max_deviation
    
    # Convert polar to Cartesian coordinates
    deviation_x = radius * np.cos(angle)
    deviation_y = radius * np.sin(angle)
    deviation_position = [deviation_x, deviation_y, 0.0]
    deviation_orientation = [np.random.uniform(-angular_error, angular_error), np.random.uniform(-angular_error, angular_error), 0.0]
    devi = np.concatenate((deviation_position, deviation_orientation))

    # break
    # insertion
    #concat this shit
    insertionPoseUP = bigHole01AbovePose + devi
    insertionPoseDown = insertionPoseUP + np.array([0, 0, -0.05, 0, 0, 0])

    rtde_c.moveL(insertionPoseUP, 0.4, 0.5)
    recording = True

    rtde_c.forceMode([0,0,0,0,0,0], selection_vector, wrench_down, force_type, limits)
    rtde_c.moveL(insertionPoseDown, 0.1, 0.2, asynchronous=True)
    ts = time.time()

    while True:
        if time.time() - ts > 2.0:
            rtde_c.stopL(5.0)
            print("2 second time out")
            print("Fail, the distance is:", np.linalg.norm(current_pose - insertionPoseDown))
            timeout = True
            break
        current_pose = rtde_r.getActualTCPPose()
        if np.linalg.norm(current_pose - insertionPoseDown) <= 0.001:
            rtde_c.stopL(5.0)
            print("Success, the distance is:", np.linalg.norm(current_pose - insertionPoseDown))
            break
    time.sleep(0.002)
    recording = False
    rtde_c.forceModeStop()

    successful_insertion = bool(int(input("Was the insertion successful? (1/0): ")))
    # save_data_to_csv(data, i+1, devi.tolist(), [angle, radius], successful_insertion, "big_rod", "big_hole_tight")
    data = []
    rtde_c.moveL(insertionPoseUP, 0.05, 0.2)

    rtde_c.moveL(bigRodAbovePose, 0.4, 0.5)
    if not successful_insertion:
        gripper.move_and_wait_for_pos(gripper.get_open_position(), speed=128, force=64)
        input("Press Enter to continue...")
        continue
    
    rtde_c.moveL(bigRodAbovePose + np.array([0, 0, -0.06, 0, 0, 0]), 0.2, 0.5)
    gripper.move_and_wait_for_pos(gripper.get_open_position(), speed=128, force=64)
    rtde_c.moveL(bigRodAbovePose, 0.2, 0.5)


# time.sleep(10)

# movement_done = True
program_running = False

thread1.join()

