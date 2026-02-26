#source ./.venv/bin/activate

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

from record_config import RecordConfig, InsertionTask, ToleranceLevel



# ##########################
# Gloal variables
# ##########################
ARDUINO_IP = "192.168.4.1"
PORT = 5000

data = []
recording = False
program_running = True
load_cell_value: int = 0
insertion = 1

ur_ip = "192.168.1.11"
rtde_c = rtde_control.RTDEControlInterface(ur_ip)
rtde_r = rtde_receive.RTDEReceiveInterface(ur_ip)
gripper = RobotiqGripper()



# ##########################
# Configuration
# ##########################
use_load_cell_feedback = True

default_orientation = np.array([3.079, -0.625, 0.0]) # Euler XYZ in radians, same for all poses in this example
default_height = 0.285 - rtde_c.getTCPOffset()[2] # in meters, same for all poses in this example

bigRodHolderPose_Up = np.concatenate([np.array([-0.37373, -0.32883, default_height]), default_orientation])
bigRodHolderPose_Down = np.concatenate([np.array([-0.37373, -0.32883, default_height - 0.085]), default_orientation])

bigRodInsertionPose_01 = np.concatenate([np.array([-0.52932, -0.30131, default_height]), default_orientation]) # tight
bigRodInsertionPose_02 = np.concatenate([np.array([-0.53889, -0.32441, default_height]), default_orientation]) # medium
bigRodInsertionPose_03 = np.concatenate([np.array([-0.5492, -0.349, default_height]), default_orientation]) # loose

smallRodHolderPose_Up = np.concatenate([np.array([-0.38156, -0.34956, default_height]), default_orientation])
smallRodHolderPose_Down = np.concatenate([np.array([-0.38156, -0.34956, default_height - 0.085]), default_orientation])

smallRodInsertionPose_01 = np.concatenate([np.array([-0.50254, -0.36721, default_height]), default_orientation]) # tight
smallRodInsertionPose_02 = np.concatenate([np.array([-0.49305, -0.34405, default_height]), default_orientation]) # medium
smallRodInsertionPose_03 = np.concatenate([np.array([-0.48365, -0.32101, default_height]), default_orientation]) # loose

config = RecordConfig(
    sequence_length=4.0, #seconds
    num_insertions=5,
    insertionTask=InsertionTask.small_rod,
    tolerance=ToleranceLevel.tight,
    holderPose_Up=smallRodHolderPose_Up,
    holderPose_Down=smallRodHolderPose_Down,
    insertionPose=smallRodInsertionPose_01,
)
config.setSampleRateHz(400.0)
config.setSavePath("test_recorded_data/real_test10/")
config.setDeviationMM(0.2)
config.setAngularErrorDeg(3.0)

config.print_config()
sample_start_idx, session_start_idx = config.getSampleSessionStart()
# print(config.getSFCount())  # just a quick check to see how many samples/sessions already exist in the folder, for naming the next files to save without overwriting
# exit()
# -----------------------------------------------




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
    gripper.connect(hostname=ur_ip, port=63352, socket_timeout=2.0)
    gripper.activate(auto_calibrate=True)
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
        # print(f"Object detected. Stopped at pos={final_pos}, force_setting(FOR)={for_value}")
        return True

    if obj_status == RobotiqGripper.ObjectStatus.AT_DEST:
        print(f"Reached target position (likely fully closed). pos={final_pos}")
        return False

    # Shouldn't usually happen if enums match firmware, but keep for safety:
    print(f"Ended with status={obj_status}, pos={final_pos}")
    return False


def listen_robot_data():
    while program_running:
        if not recording:
            time.sleep(0.001)   # yield CPU, reduce jitter
            continue
        next_t = start = time.perf_counter()
        while recording:
            now = time.perf_counter()
            if now < next_t:
                time.sleep(next_t - now)
            

            t_sample = time.perf_counter()
            force = rtde_r.getActualTCPForce()            # 6
            pose = rtde_r.getActualTCPPose()              # 6
            speed = rtde_r.getActualTCPSpeed()            # 6
            q = rtde_r.getActualQ()                       # 6
            qd = rtde_r.getActualQd()                     # 6
            acc = rtde_r.getActualToolAccelerometer()     # 3
            Irobot = rtde_r.getActualRobotCurrent()       # 1
            Vrobot = rtde_r.getActualRobotVoltage()       # 1
            Ij = rtde_r.getActualCurrent()                # 6
            Tj = rtde_r.getJointTemperatures()            # 6

            row = np.empty((48,), dtype=np.float32)
            row[0] = t_sample - start
            row[1:7] = force
            row[7:13] = pose
            row[13:19] = speed
            row[19:25] = q
            row[25:31] = qd
            row[31:34] = acc
            row[34] = Irobot
            row[35] = Vrobot
            row[36:42] = Ij
            row[42:48] = Tj

            data.append(row)
            next_t += config.deltatime
            # print(actual_force)

def receive_loadcell_arduino():
    global load_cell_value
    while program_running:
        try:
            print(f"Connecting to {ARDUINO_IP}:{PORT} ...")
            s = socket.create_connection((ARDUINO_IP, PORT), timeout=10.0)  # connect timeout only
            s.settimeout(None)  # blocking reads (no read timeout)
            s.sendall(b"START\n")

            print("Connected. Receiving values...")

            buf = b""
            while program_running:
                chunk = s.recv(4096)
                if not chunk:
                    raise ConnectionError("socket closed")
                buf += chunk

                while b"\n" in buf and program_running:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        load_cell_value = int(line.decode("ascii", errors="ignore"))
                        # print(load_cell_value)
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

import numpy as np
import spatialmath as sm  # or whatever import name you use (you had `sm` already)

def transform_forces_after_recording(data: np.ndarray,
                                     force_cols: slice = slice(1, 7),
                                     pose_cols: slice = slice(7, 13)) -> np.ndarray:
    """
    Transform TCP wrench (force+torque) for each row AFTER recording.

    Modifies `data` in-place and returns it.

    Parameters
    ----------
    data : np.ndarray
        Shape [N, D]. Must contain force in force_cols (6 values)
        and pose in pose_cols (6 values), where pose[3:] is rotation vector.
    force_cols : slice
        Columns containing [Fx, Fy, Fz, Tx, Ty, Tz] (length 6).
    pose_cols : slice
        Columns containing [x, y, z, rx, ry, rz] (length 6).

    Returns
    -------
    np.ndarray
        The same array object (modified in-place).
    """
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)

    if data.ndim != 2:
        raise ValueError(f"data must be 2D [N,D], got shape {data.shape}")

    # Validate slices length
    if (force_cols.stop - force_cols.start) != 6:
        raise ValueError("force_cols must span exactly 6 columns")
    if (pose_cols.stop - pose_cols.start) != 6:
        raise ValueError("pose_cols must span exactly 6 columns")

    # Constant part
    Ry180 = sm.SE3.Ry(180, 'deg')

    for i in range(data.shape[0]):
        actual_pose = data[i, pose_cols]         # [x,y,z, rx,ry,rz]
        force = data[i, force_cols]              # [Fx,Fy,Fz, Tx,Ty,Tz]

        # Your exact conversion method
        R_6x6 = np.zeros((6, 6), dtype=np.float64)
        R3 = np.array(Ry180 * sm.SE3.EulerVec(actual_pose[3:]), dtype=np.float64)[:3, :3]
        R_6x6[:3, :3] = R3
        R_6x6[3:, 3:] = R3
        force_tf = (R_6x6 @ force.reshape(6, 1)).reshape(6,)

        # Write back in-place
        data[i, force_cols] = force_tf.astype(data.dtype, copy=False)

    return data

def save_data_to_csv(f_data, idx: int, session: int, deviation, angle_radius, successful_insertion, failure_reason: str):
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

    basename = f"sample_{idx}_session_{session}"
    csv_filename = config.folder_name + "/" + basename + ".csv"
    json_filename = config.folder_name + "/" + basename + ".json"

    df.to_csv(csv_filename, index=False)
    
    metadata = {
        "saved_at": timestamp.isoformat(),
        "saved_in": config.folder_name,
        "corresponding_csv": csv_filename,
        "sample_rate_Hz": config.sample_rate,
        "sample_deltatime_s": config.deltatime,
        "num_samples": len(f_data),
        "num_insertions": config.num_insertions,
        "sequence_length_s": config.sequence_length,
        "deviation": deviation,
        "angle_radius": angle_radius,
        "min_max_deviation_mm": config.min_max_deviation_mm,
        "min_max_deviation_m": config.min_max_deviation,
        "angular_error_deg": config.angular_error_deg,
        "angular_error_rad": config.angular_error,
        "successful_insertion": successful_insertion,
        "failure_reason": failure_reason,
        "insertion_task": config.insertionTask.value,
        "tolerance": config.tolerance.value,
        "holderPose_Up": config.holderPose_Up.tolist(),
        "holderPose_Down": config.holderPose_Down.tolist(),
        "insertionPose": config.insertionPose.tolist(),
    }
    with open(json_filename, "w") as jf:
        json.dump(metadata, jf, indent=2)
    print("saved csv and metadata json")



thread1 = threading.Thread(target=listen_robot_data)
if use_load_cell_feedback:
    thread2 = threading.Thread(target=receive_loadcell_arduino)

rtde_c.moveL(config.holderPose_Up)
init_gripper(gripper, ur_ip)
thread1.start()
if use_load_cell_feedback:
    thread2.start()

num_successful = 0
num_failed = 0

for i in range(config.num_insertions):
    print(f"\n=== Starting insertion {i+1}/{config.num_insertions} ===")
    rtde_c.zeroFtSensor()
    rtde_c.moveL(config.holderPose_Down, 0.1, 0.5)
    close_until_force_limit(gripper, target_force_n=50.0, speed=128)
    rtde_c.moveL(config.holderPose_Up, 0.1, 0.5)

    # exit()
    
    angle = np.random.uniform(0, 2 * np.pi)  # Random angle in radians
    radius = np.random.uniform(0, config.min_max_deviation)  # Random radius
    # angle = 0.0
    # radius = min_max_deviation
    
    # Convert polar to Cartesian coordinates
    deviation_x = radius * np.cos(angle)
    deviation_y = radius * np.sin(angle)
    deviation_position = [deviation_x, deviation_y, 0.0]
    deviation_orientation = [np.random.uniform(-config.angular_error, config.angular_error), np.random.uniform(-config.angular_error, config.angular_error), 0.0]
    devi = np.concatenate((deviation_position, deviation_orientation))

    # break
    # insertion
    #concat this shit
    insertionPoseUP = config.insertionPose + devi
    insertionPoseDown = insertionPoseUP + np.array([0, 0, -0.049, 0, 0, 0])

    sequence_start_time = time.perf_counter()

    recording = True
    rtde_c.moveL(insertionPoseUP, 0.4, 0.5)

    # Good practice: avoid entering force mode right after motion without a tiny pause
    time.sleep(0.02)  # UR recommends >=0.02s before force mode in many cases :contentReference[oaicite:6]{index=6}

    rtde_c.zeroFtSensor()  # reduce bias (especially important with gripper/payload) :contentReference[oaicite:7]{index=7}
    rtde_c.forceModeSetDamping(0.1)
    rtde_c.forceModeSetGainScaling(1.2)

    task_frame = insertionPoseUP
    selection_vector = [1, 1, 0, 1, 1, 0]
    wrench = [0, 0, 0, 0, 0, 0]
    limits = [0.05, 0.05, 0.02, 0.5, 0.5, 0.2]

    rtde_c.forceMode(task_frame, selection_vector, wrench, 2, limits)

    rtde_c.moveL(insertionPoseDown, 0.03, 0.3, asynchronous=True)

    t0 = time.time()
    stall_t0 = None
    TIMEOUT = 2.3
    successful_insertion = False
    failure_reason = None

    while True:
        pose  = rtde_r.getActualTCPPose()
        speed = rtde_r.getActualTCPSpeed()
        force = rtde_r.getActualTCPForce()

        if time.time() - t0 > TIMEOUT:
            print("Timeout -> Fail")
            failure_reason = "Timeout"
            successful_insertion = False
            num_failed += 1
            rtde_c.stopL(1.0)
            break

        # jam heuristics: not moving in z + pushing hard
        vz = abs(speed[2])
        fz = abs(force[2])

        if vz < 0.002 and fz > 15:         # tune thresholds to your setup
            if stall_t0 is None:
                stall_t0 = time.time()
            elif time.time() - stall_t0 > 0.25:
                print("Stall/jam -> Fail")
                failure_reason = "Stall/jam"
                successful_insertion = False
                num_failed += 1
                rtde_c.stopL(1.0)
                break
        else:
            stall_t0 = None

        # hard force safety threshold (optional)
        if fz > 40:
            print("Force limit -> Fail")
            failure_reason = "Force limit"
            successful_insertion = False
            num_failed += 1
            rtde_c.stopL(1.0)
            break

        if use_load_cell_feedback:
            if load_cell_value > 4000:  # tune threshold to your setup
                print(f"Successful insertion")
                successful_insertion = True
                num_successful += 1
                rtde_c.stopL(1.0)
                break

        time.sleep(0.01)

    rtde_c.forceModeStop()

    remaining = config.sequence_length - (time.perf_counter() - sequence_start_time)
    print(f"gotta wait {remaining:.2f} s")
    if remaining > 0:
        time.sleep(remaining)

    recording = False

    if not use_load_cell_feedback:
        successful_insertion = False
    
    # input("Continue?")
    data_np = np.asarray(data)  # if you currently have a list of rows
    data_np = transform_forces_after_recording(data_np)
    save_data_to_csv(data_np, i+1 + sample_start_idx, session_start_idx+1, devi.tolist(), [angle, radius], successful_insertion, failure_reason)
    data = []
    if not successful_insertion:
        gripper.move_and_wait_for_pos(gripper.get_open_position(), speed=128, force=64)

    rtde_c.moveL(insertionPoseUP, 0.05, 0.2)

    rtde_c.moveL(config.holderPose_Up, 0.4, 0.5)

    print(f"S: {num_successful}, F: {num_failed}")
    S_total, F_total = config.getSFCount()
    print(f"Total S: {S_total}, F: {F_total}")

    if not successful_insertion:
        input("place the rod and continue")
        continue
    
    rtde_c.moveL(config.holderPose_Down, 0.2, 0.5)
    gripper.move_and_wait_for_pos(gripper.get_open_position(), speed=128, force=64)
    rtde_c.moveL(config.holderPose_Up, 0.2, 0.5)



program_running = False # this stops the infinite loops in the threads

thread1.join()
if use_load_cell_feedback:
    thread2.join()

