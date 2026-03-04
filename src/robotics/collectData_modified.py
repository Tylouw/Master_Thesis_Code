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
import numpy as np
import spatialmath as sm  # or whatever import name you use (you had `sm` already)

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

default_orientation = np.array([3.079, -0.625, 0.0]) # Rotation vector in radians, same for all poses in this example
default_height = 0.285 - rtde_c.getTCPOffset()[2] # in meters, same for all poses in this example

# --- Load cell / depth configuration -----------------------------------------
# We define the load-cell surface height in BASE coordinates (z in meters).
# The script will adjust the commanded insertion depth so that the TCP target
# reaches (load_cell_surface_z + tcp_to_rod_tip_z), instead of using a fixed
# delta-z.
#
# Defaults are chosen to preserve your previous behavior:
#   insertionPoseDown = insertionPoseUP + [0,0,-0.049,...]
# If you physically adjust the load cell height, update LOAD_CELL_SURFACE_Z_BASE_M.
TCP_TO_ROD_TIP_Z_M = 0.0       # [m] set if you want "rod tip hits load cell" instead of TCP
CONTACT_OVERTRAVEL_M = 0.0     # [m] additional travel beyond contact target (keep 0.0 for gentle hits)
LOAD_CELL_SURFACE_Z_BASE_M = None  # [m] if None -> derived from default_height and old delta-z

# Load cell trigger threshold (Arduino units). Tune to your setup.
LOAD_CELL_TRIGGER_THRESHOLD = 4000
# -----------------------------------------------------------------------------



bigRodHolderPose_Up = np.concatenate([np.array([-0.37296, -0.32824, default_height]), default_orientation])
bigRodHolderPose_Down = np.concatenate([np.array([-0.37296, -0.32824, default_height - 0.085]), default_orientation])

bigRodInsertionPose_01 = np.concatenate([np.array([-0.5492, -0.349, default_height]), default_orientation]) # tight
bigRodInsertionPose_02 = np.concatenate([np.array([-0.53949, -0.32519, default_height]), default_orientation]) # medium
bigRodInsertionPose_03 = np.concatenate([np.array([-0.52982, -0.30205, default_height]), default_orientation]) # loose

smallRodHolderPose_Up = np.concatenate([np.array([-0.38113, -0.34856, default_height]), default_orientation])
smallRodHolderPose_Down = np.concatenate([np.array([-0.38113, -0.34856, default_height - 0.085]), default_orientation])

smallRodInsertionPose_01 = np.concatenate([np.array([-0.50254, -0.36721, default_height]), default_orientation]) # tight
smallRodInsertionPose_02 = np.concatenate([np.array([-0.49305, -0.34405, default_height]), default_orientation]) # medium
smallRodInsertionPose_03 = np.concatenate([np.array([-0.48365, -0.32101, default_height]), default_orientation]) # loose

rectRodHolderPose_Up = np.concatenate([np.array([-0.38931, -0.3687, default_height]), default_orientation])
rectRodHolderPose_Down = np.concatenate([np.array([-0.38931, -0.3687, default_height - 0.085]), default_orientation])

rectRodInsertionPose_01 = np.concatenate([np.array([-0.59493, -0.32949, default_height]), default_orientation]) # tight
rectRodInsertionPose_02 = np.concatenate([np.array([-0.58525, -0.30643, default_height]), default_orientation]) # medium
rectRodInsertionPose_03 = np.concatenate([np.array([-0.57579, -0.28315, default_height]), default_orientation]) # loose

config = RecordConfig(
    sequence_length=4.0, #seconds
    num_insertions=35,
    insertionTask=InsertionTask.rect_rod,
    tolerance=ToleranceLevel.medium,
    holderPose_Up=rectRodHolderPose_Up,
    holderPose_Down=rectRodHolderPose_Down,
    insertionPose=rectRodInsertionPose_02,
)
config.setSampleRateHz(500.0)
# config.setSavePath("test_recorded_data/real_test11/")
config.setSavePath("training_data/batch_1/rect_rod_t0.2/")
config.setDeviationMM(0.15)
config.setAngularErrorDeg(5.0)

config.print_config()

# Derive default load-cell surface z from your previous hard-coded delta-z (0.049 m)
# if not explicitly specified.
if LOAD_CELL_SURFACE_Z_BASE_M is None:
    LOAD_CELL_SURFACE_Z_BASE_M = float(default_height - 0.049)

print(f"LOAD_CELL_SURFACE_Z_BASE_M = {LOAD_CELL_SURFACE_Z_BASE_M:.6f} m  (TCP_TO_ROD_TIP_Z_M={TCP_TO_ROD_TIP_Z_M:.4f}, CONTACT_OVERTRAVEL_M={CONTACT_OVERTRAVEL_M:.4f})")

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

    Returns a dict with the commanded force setting and the detected status.
    """
    for_value = approx_force_to_for_value(target_force_n)
    closed_pos = gripper.get_closed_position()

    final_pos, obj_status = gripper.move_and_wait_for_pos(
        position=closed_pos,
        speed=speed,
        force=for_value,
    )

    info = {
        "target_force_n": float(target_force_n),
        "for_value": int(for_value),
        "speed": int(speed),
        "final_pos": int(final_pos) if isinstance(final_pos, (int, np.integer)) else final_pos,
        "obj_status": str(obj_status),
        "object_detected": False,
    }

    if obj_status in (
        RobotiqGripper.ObjectStatus.STOPPED_OUTER_OBJECT,
        RobotiqGripper.ObjectStatus.STOPPED_INNER_OBJECT,
    ):
        info["object_detected"] = True
        return info

    if obj_status == RobotiqGripper.ObjectStatus.AT_DEST:
        print(f"Reached target position (likely fully closed). pos={final_pos}")
        return info

    # Shouldn't usually happen if enums match firmware, but keep for safety:
    print(f"Ended with status={obj_status}, pos={final_pos}")
    return info




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


# --- Systematic deviation plan ------------------------------------------------
def build_deviation_plan(
    n: int,
    max_pos_dev_m: float,
    max_ang_dev_rad: float,
    *,
    pos_radii_steps: int = 5,
    pos_angle_steps: int = 12,
    ori_steps: int = 3,
):
    """
    Build a deterministic (non-random) deviation plan that covers position + orientation space.

    Position (x,y): polar grid (r,theta) with r in [0..max] and theta evenly spaced.
    Orientation (rx,ry): small grid in [-max..+max]. rz stays 0.

    The plan is deterministic and repeats if n exceeds the grid size.
    """
    max_pos_dev_m = float(max(0.0, max_pos_dev_m))
    max_ang_dev_rad = float(max(0.0, max_ang_dev_rad))

    # Position grid
    radii = np.linspace(0.0, max_pos_dev_m, num=max(2, int(pos_radii_steps)))
    angles = np.linspace(0.0, 2.0 * np.pi, num=max(4, int(pos_angle_steps)), endpoint=False)

    pos_list = [(0.0, 0.0)]  # (theta, r) include center once
    for r in radii[1:]:
        for th in angles:
            pos_list.append((float(th), float(r)))

    # Orientation grid (rx, ry)
    ori_vals = np.linspace(-max_ang_dev_rad, max_ang_dev_rad, num=max(2, int(ori_steps)))
    ori_list = [(float(rx), float(ry)) for rx in ori_vals for ry in ori_vals]

    plan = []
    for i in range(int(n)):
        th, r = pos_list[i % len(pos_list)]
        rx, ry = ori_list[(i // len(pos_list)) % len(ori_list)]
        dx = r * np.cos(th)
        dy = r * np.sin(th)

        plan.append(
            {
                "deviation_index": int(i),
                "pos_theta_rad": float(th),
                "pos_radius_m": float(r),
                "dx_m": float(dx),
                "dy_m": float(dy),
                "rx_rad": float(rx),
                "ry_rad": float(ry),
            }
        )

    return {
        "version": "grid_v1",
        "pos_radii_steps": int(pos_radii_steps),
        "pos_angle_steps": int(pos_angle_steps),
        "ori_steps": int(ori_steps),
        "pos_grid_size": int(len(pos_list)),
        "ori_grid_size": int(len(ori_list)),
        "plan": plan,
    }
# -----------------------------------------------------------------------------

def save_data_to_csv(f_data, idx: int, session: int, successful_insertion: bool, failure_reason: str, extra_metadata: dict | None = None):
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

    if extra_metadata:
        # Shallow merge: keys in extra_metadata override defaults above.
        metadata.update(extra_metadata)

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


# Build a deterministic deviation plan (systematic coverage, no random sampling)
deviation_plan_info = build_deviation_plan(
    n=config.num_insertions,
    max_pos_dev_m=config.min_max_deviation,
    max_ang_dev_rad=config.angular_error,
    pos_radii_steps=5,
    pos_angle_steps=12,
    ori_steps=3,
)
deviation_plan = deviation_plan_info["plan"]
print(f"Deviation plan: version={deviation_plan_info['version']} pos_grid={deviation_plan_info['pos_grid_size']} ori_grid={deviation_plan_info['ori_grid_size']}")

for i in range(config.num_insertions):
    print(f"\n=== Starting insertion {i+1}/{config.num_insertions} ===")
    rtde_c.zeroFtSensor()
    rtde_c.moveL(config.holderPose_Down, 0.1, 0.5)
    grip_info = close_until_force_limit(gripper, target_force_n=50.0, speed=128)
    rtde_c.moveL(config.holderPose_Up, 0.1, 0.5)

    # exit()
    # Systematic (deterministic) deviation instead of random sampling
    dev = deviation_plan[i]
    angle = dev["pos_theta_rad"]
    radius = dev["pos_radius_m"]

    deviation_x = dev["dx_m"]
    deviation_y = dev["dy_m"]
    deviation_position = [deviation_x, deviation_y, 0.0]
    deviation_orientation = [dev["rx_rad"], dev["ry_rad"], 0.0]

    devi = np.concatenate((deviation_position, deviation_orientation))


    # break
    # insertion
    #concat this shit
    insertionPoseUP = config.insertionPose + devi
    insertionPoseDown = insertionPoseUP.copy()
    # Compute TCP z target so that (load_cell_surface_z + tcp_to_tip_z) is reached
    tcp_target_z_contact = float(LOAD_CELL_SURFACE_Z_BASE_M + TCP_TO_ROD_TIP_Z_M)
    insertionPoseDown[2] = tcp_target_z_contact - float(CONTACT_OVERTRAVEL_M)



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

    # task_frame = insertionPoseUP.copy()
    # task_frame[3:] = config.insertionPose[3:]  # remove random tilt from the force-frame

    # # Non-compliant: x, y, rz  -> position/orientation is respected
    # # Compliant:     z, rx, ry -> force push + angular compliance
    # selection_vector = [0, 0, 1, 0, 0, 0]

    # # Downward insertion force (tune to your setup)
    # wrench = [0, 0, 25.0, 0, 0, 0]

    # # Limits meaning depends on compliance:
    # #  - compliant axes (1): max TCP speed [m/s] or [rad/s]
    # #  - non-compliant axes (0): max deviation [m] or [rad]
    # limits = [0.001, 0.001, 0.08, 0.8, 0.8, 0.20]

    rtde_c.forceMode(task_frame, selection_vector, wrench, 2, limits)

    run_meta["force_mode"] = {
        "task_frame": task_frame.tolist() if hasattr(task_frame, "tolist") else list(task_frame),
        "selection_vector": list(selection_vector),
        "wrench": list(wrench),
        "type": 2,
        "limits": list(limits),
        "damping": 0.1,
        "gain_scaling": 1.2,
        "moveL_down_vel": 0.03,
        "moveL_down_acc": 0.3,
    }


    rtde_c.moveL(insertionPoseDown, 0.03, 0.3, asynchronous=True)

    t0 = time.time()
    stall_t0 = None
    TIMEOUT = 2.3
    successful_insertion = False
    failure_reason = None

# Per-run metadata (will be saved to JSON)
    run_meta = {
        "deviation_plan": {k: v for k, v in deviation_plan_info.items() if k != "plan"},
        "deviation": {
            "mode": deviation_plan_info["version"],
            "deviation_index": int(dev["deviation_index"]),
            "pos_theta_rad": float(angle),
            "pos_radius_m": float(radius),
            "pos_dx_m": float(deviation_x),
            "pos_dy_m": float(deviation_y),
            "ori_rx_rad": float(dev["rx_rad"]),
            "ori_ry_rad": float(dev["ry_rad"]),
            "ori_rz_rad": 0.0,
            "pos_dx_mm": float(deviation_x * 1000.0),
            "pos_dy_mm": float(deviation_y * 1000.0),
            "ori_rx_deg": float(np.degrees(dev["rx_rad"])),
            "ori_ry_deg": float(np.degrees(dev["ry_rad"])),
        },
        "load_cell": {
            "use_load_cell_feedback": bool(use_load_cell_feedback),
            "arduino_ip": ARDUINO_IP,
            "arduino_port": PORT,
            "trigger_threshold": int(LOAD_CELL_TRIGGER_THRESHOLD),
            "surface_z_base_m": float(LOAD_CELL_SURFACE_Z_BASE_M),
            "tcp_to_rod_tip_z_m": float(TCP_TO_ROD_TIP_Z_M),
            "contact_overtravel_m": float(CONTACT_OVERTRAVEL_M),
        },
        "targets": {
            "insertionPose_nominal": config.insertionPose.tolist(),
            "insertionPoseUP_actual": insertionPoseUP.tolist(),
            "insertionPoseDown_actual": insertionPoseDown.tolist(),
            "tcp_target_z_contact_m": float(tcp_target_z_contact),
        },
        "gripper": grip_info,
    }


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
            if load_cell_value > LOAD_CELL_TRIGGER_THRESHOLD:  # tune threshold to your setup
                print(f"Successful insertion")
                successful_insertion = True
                num_successful += 1
                rtde_c.stopL(1.0)
                break

        time.sleep(0.01)

    rtde_c.forceModeStop()

    # Store stop info (at break condition)
    run_meta["stop"] = {
        "successful_insertion": bool(successful_insertion),
        "failure_reason": failure_reason,
        "load_cell_value_at_stop": int(load_cell_value) if use_load_cell_feedback else None,
    }

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
    save_data_to_csv(data_np, i+1 + sample_start_idx, session_start_idx+1, successful_insertion, failure_reason, extra_metadata=run_meta)
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

