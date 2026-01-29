import csv
import os
from robotiq_gripper import RobotiqGripper
# import asyncio
# from semantic.common.utils import redMsg, greenMsg, defineSchemaId
# from semantic.fera.fera_types.type_constructors import tool_center_point_state

# from force_based_learning.core import RobotRTDE, close_gripper, open_gripper

def moveL_and_record(rtde_c, rtde_r, final_pose, execution_time, dt, real_date):
# async def moveL_and_record(rtde_c, rtde_r, final_pose, execution_time, dt, parent):
    ##################################################################
    ### Assume that until now the object has been picked correctly ###
    ##################################################################

    time_counter = 0.0
    list_timestamp = []
    list_force_base_X = []
    list_force_base_Y = []
    list_force_base_Z = []
    list_torque_base_X = []
    list_torque_base_Y = []
    list_torque_base_Z = []
    list_TCP_pose_X = []
    list_TCP_pose_Y = []
    list_TCP_pose_Z = []
    list_TCP_pose_RX = []
    list_TCP_pose_RY = []
    list_TCP_pose_RZ = []
    
    print("Before sending the moveL async=True")
    rtde_c.moveL(final_pose, 0.25*0.3, 1.2*0.3, asynchronous=True)
    print("After sending the moveL async=True")

    # try:
    while (time_counter <= execution_time) :
        t_start = rtde_c.initPeriod()

        # Pose and ForceTorque
        robot_pose = rtde_r.getActualTCPPose()
        force_torque = rtde_r.getActualTCPForce()

        # Save
        list_timestamp.append(time_counter)
        list_force_base_X.append(force_torque[0])
        list_force_base_Y.append(force_torque[1])
        list_force_base_Z.append(force_torque[2])
        list_torque_base_X.append(force_torque[3])
        list_torque_base_Y.append(force_torque[4])
        list_torque_base_Z.append(force_torque[5])
        list_TCP_pose_X.append(robot_pose[0])
        list_TCP_pose_Y.append(robot_pose[1])
        list_TCP_pose_Z.append(robot_pose[2])
        list_TCP_pose_RX.append(robot_pose[3])
        list_TCP_pose_RY.append(robot_pose[4])
        list_TCP_pose_RZ.append(robot_pose[5])
        
        rtde_c.waitPeriod(t_start)
        time_counter += dt
    
    return [list_timestamp, list_force_base_X, list_force_base_Y, list_force_base_Z, list_torque_base_X, list_torque_base_Y, list_torque_base_Z, list_TCP_pose_X, list_TCP_pose_Y, list_TCP_pose_Z, list_TCP_pose_RX, list_TCP_pose_RY, list_TCP_pose_RZ]

def save_lists2csv(output_file,
                   list_timestamp, 
                   list_force_base_X, list_force_base_Y, list_force_base_Z, 
                   list_torque_base_X, list_torque_base_Y, list_torque_base_Z, 
                   list_TCP_pose_X, list_TCP_pose_Y, list_TCP_pose_Z, 
                   list_TCP_pose_RX, list_TCP_pose_RY, list_TCP_pose_RZ):
    
    # Writing to CSV file
    headers = ["Timestamp", "FX_Base", "FY_Base", "FZ_Base", "MuX_Base", "MuY_Base", "MuZ_Base", "TCP_X", "TCP_Y", "TCP_Z", "TCP_RX", "TCP_RY", "TCP_RZ"]
    data = zip(list_timestamp, 
            list_force_base_X, list_force_base_Y, list_force_base_Z, 
            list_torque_base_X, list_torque_base_Y, list_torque_base_Z, 
            list_TCP_pose_X, list_TCP_pose_Y, list_TCP_pose_Z,
            list_TCP_pose_RX, list_TCP_pose_RY, list_TCP_pose_RZ)
    
    # Check if the file exists
    if not os.path.exists(output_file):
        # If the file does not exist, create it and write to it
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(data)


def main():

    #################### Parameters
    robot_ip_address = "172.28.60.10"   
    number_iterations = 2
    offsetZ = 0.15
    isCorrectDirection = False
    output_folder = "force_based_learning/force_data/"
    output_file_name = "robot_data_"
    execution_time = 3.0
    dt = 1.0/500

    #################### Some Important Joint configurations
    DESIRED_HOME_Q = [-2.2379072348224085, -1.5813790760436, -1.7870478630065918, 0.21874873220410151, 1.0885045528411865, -3.119746510182516]
    DESIRED_PLACE_L = [-0.41299371953796277, -0.27833311373668, 0.3435522225375916, 1.3752103703764937, -0.9033642259087404, -0.9033700202428113]
    DESIRED_PLACE_L_INVERTED = [-0.4129963183458661, -0.2783355545238703, 0.35871146843172896, 1.37522475022197, -0.9033594853912637, -0.9033649117187459]

    # #################### Initialization
    # print("Initializing robot...")
    # try:
    #     ur_robot = RobotRTDE(robot_ip_address)
    # except Exception as e:
    #     print("Failed to connect with the robot...\n " + str(e))

    # ur_robot.setTcp([0, 0, 0, 0, 0, 0])
    # ur_robot.zeroFtSensor()

    # #################### Gripper Configurations
    # print("Creating gripper...")
    # gripper = RobotiqGripper()
    # print("Connecting to gripper...")
    # gripper.connect(robot_ip_address, 63352)

    # #################### Start of pipeline
    # ur_robot.moveJ(DESIRED_HOME_Q)
    # open_gripper(gripper)
    
    # for iteration in range(number_iterations):
    #     print("#######################################################################")
    #     print("Iteration number: " + str(iteration))

    #     output_file = output_folder + output_file_name + str(iteration) + ".csv"

    #     if(isCorrectDirection):
    #         ur_robot.moveL(DESIRED_PLACE_L)
    #     else:
    #         ur_robot.moveL(DESIRED_PLACE_L_INVERTED)
    #     close_gripper(gripper)

    #     current_tcp_pose = ur_robot.getActualTCPPose()
    #     preplace_pose = current_tcp_pose[:]

    #     # Move up the TCP along Z axis by offset_z
    #     preplace_pose[2] += offsetZ 
    #     ur_robot.moveL(preplace_pose)
    #     print("Robot will move to Pre Place position, which is: \n        " + str(preplace_pose))

    #     ##################################################################
    #     ### Assume that until now the object has been picked correctly ###
    #     ##################################################################

    #     res = moveL_and_record(ur_robot, DESIRED_PLACE_L, execution_time, dt)

    #     print("Opening the gripper")
    #     open_gripper(gripper)
    #     ur_robot.moveJ(DESIRED_HOME_Q)

    #     save_lists2csv(output_file, 
    #                    res[0],
    #                    res[1], res[2], res[3], res[4], res[5], res[6],
    #                    res[7], res[8], res[9], res[10], res[11], res[12])


# This checks if the script is being run directly, not being imported
if __name__ == "__main__":
    main()
