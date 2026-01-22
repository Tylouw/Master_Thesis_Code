from enum import Enum

class Robot_Attribute(Enum):
    force = "force"
    tcp_pose_position = "tcp_pose_position"
    tcp_pose_orientation = "tcp_pose_orientation"
    tcp_velocity_position = "tcp_velocity_position"
    tcp_velocity_orientation = "tcp_velocity_orientation"
    joint_angles = "joint_angles"
    joint_velocity = "joint_velocity"
    tool_accelerometer = "tool_accelerometer"
    current_whole = "current_whole"
    voltage_whole = "voltage_whole"
    current_joints = "current_joints"
    temp_joints = "temp_joints"

class ColumnDefinitionCSVFile:
    numInsertion: str = ["num_insertion"]
    numInsertionOld: str = ["number of current insertion [0]"]
    tcp_force: str = ["tcp_force_x[N]", "tcp_force_y[N]", "tcp_force_z[N]", "tcp_force_rx[Nm?]", "tcp_force_ry[Nm?]", "tcp_force_rz[Nm?]"]
    tcp_pose_position: str = ["tcp_pose_x[m]", "tcp_pose_y[m]", "tcp_pose_z[m]"]
    tcp_pose_orientation: str = ["tcp_pose_rx[rot_vec_x]", "tcp_pose_ry[rot_vec_y]", "tcp_pose_rz[rot_vec_z]"]
    tcp_velo_position: str = ["tcp_velo_x[m/s]", "tcp_velo_y[m/s]", "tcp_velo_z[m/s]"]
    tcp_velo_orientation: str = ["tcp_velo_rx[rot_vec_x/s]", "tcp_velo_ry[rot_vec_y/s]", "tcp_velo_rz[rot_vec_z/s]"]
    joint_angles: str = ["q1[rad]","q2[rad]","q3[rad]","q4[rad]","q5[rad]","q6[rad]"]
    joint_velo: str = ["qd1[rad/s]","qd2[rad/s]","qd3[rad/s]","qd4[rad/s]","qd5[rad/s]","qd6[rad/s]"]
    tool_accel: str = ["tool_acc_x", "tool_acc_y", "tool_acc_z"]
    current_whole_robot: str = ["current_of_whole_robot[A]"]
    voltage_whole_robot: str = ["voltage_of_whole_robot[V]"]
    current_each_joint: str = ["current_J1[A]", "current_J2[A]", "current_J3[A]", "current_J4[A]", "current_J5[A]", "current_J6[A]"]
    temp_each_joint: str = ["temp_J1[C]", "temp_J2[C]", "temp_J3[C]", "temp_J4[C]", "temp_J5[C]", "temp_J6[C]"]

    # attribute: (csv file column name, plot axis legend names, plot y axis name, plot name)
    mapping_new = {
        'force':                    (["tcp_force_x[N]", "tcp_force_y[N]", "tcp_force_z[N]", "tcp_force_rx[Nm?]", "tcp_force_ry[Nm?]", "tcp_force_rz[Nm?]"], ["x", "y", "z", "rx", "ry", "rz"],            "Force (N)",             "TCP Force"),
        'tcp_pose_position':        (["tcp_pose_x[m]", "tcp_pose_y[m]", "tcp_pose_z[m]"],                                                                   ["x", "y", "z"],                              "Position (m)",          "TCP Pose (Position)"),
        'tcp_pose_orientation':     (["tcp_pose_rx[rot_vec_x]", "tcp_pose_ry[rot_vec_y]", "tcp_pose_rz[rot_vec_z]"],                                        ["rx", "ry", "rz"],                           "Orientation (rad)",       "TCP Pose (Orientation)"),
        'tcp_velocity_position':    (["tcp_velo_x[m/s]", "tcp_velo_y[m/s]", "tcp_velo_z[m/s]"],                                                             ["xd", "yd", "zd"],                           "Velocity (m/s)",     "TCP Velocity (Position)"),
        'tcp_velocity_orientation': (["tcp_velo_rx[rot_vec_x/s]", "tcp_velo_ry[rot_vec_y/s]", "tcp_velo_rz[rot_vec_z/s]"],                                  ["rxd", "ryd","rzd"],                         "Velocity (rad/s)",    "TCP Velocity (Orientation)"),
        'joint_angles':             (["q1[rad]","q2[rad]","q3[rad]","q4[rad]","q5[rad]","q6[rad]"],                                                         ["q1", "q2", "q3", "q4", "q5", "q6"],         "Angle (rad)",             "Joint Angles"),
        'joint_velocity':           (["qd1[rad/s]","qd2[rad/s]","qd3[rad/s]","qd4[rad/s]","qd5[rad/s]","qd6[rad/s]"],                                       ["qd1", "qd2", "qd3", "qd4", "qd5", "qd6"],   "Velocity (rad/s)",          "Joint Velocity"),
        'tool_accelerometer':       (["tool_acc_x", "tool_acc_y", "tool_acc_z"],                                                                            ["x", "y", "z"],                              "Acceleration", "Tool Accelerometer"),
        'current_whole':            (["current_of_whole_robot[A]"],                                                                                         ["Amps"]  ,                                   "Current (A)",           "Current Consumption (Whole Robot)"),
        'voltage_whole':            (["voltage_of_whole_robot[V]"],                                                                                         ["Volts"],                                    "Voltage (V)",           "Voltage Consumption"),
        'current_joints':           (["current_J1[A]", "current_J2[A]", "current_J3[A]", "current_J4[A]", "current_J5[A]", "current_J6[A]"],                ["J1", "J2", "J3", "J4", "J5", "J6"],         "Current (A)","Current Consumption (Each Joint)"),
        'temp_joints':              (["temp_J1[C]", "temp_J2[C]", "temp_J3[C]", "temp_J4[C]", "temp_J5[C]", "temp_J6[C]"],                                  ["J1", "J2", "J3", "J4", "J5", "J6"],         "Temperature (C)",       "Joint Temperature"),
    }