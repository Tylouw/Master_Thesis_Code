import rtde_receive
import rtde_control
import time
from pynput import keyboard
import threading
from RobotiqHandE import RobotiqGripper

from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
# rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.11")
# rtde_c = rtde_control.RTDEControlInterface("192.168.0.12")
# actual_q = rtde_r.getActualTCPPose()
# a_pressed = False
# print(actual_q)

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


def main():
    ur_ip = "192.168.1.11"  # <-- set your robot IP

    rtde_c = RTDEControl(ur_ip)
    rtde_r = RTDEReceive(ur_ip)

    gripper = RobotiqGripper()

    try:
        init_gripper(gripper, ur_ip)

        # Example UR motion
        # rtde_c.moveL([...], speed=..., acceleration=...)
        time.sleep(0.2)

        # Close and stop when the gripper hits an object, applying up to ~60 N worth of force setting.
        close_until_force_limit(gripper, target_force_n=60.0, speed=128)

        # Continue robot logic...
        time.sleep(3)

        # Optionally open again
        gripper.move_and_wait_for_pos(gripper.get_open_position(), speed=128, force=64)

    finally:
        # Always clean up
        try:
            gripper.disconnect()
        except Exception:
            pass
        rtde_c.disconnect()
        rtde_r.disconnect()


if __name__ == "__main__":
    main()

# def on_press(key, injected):
#     try:
#         a_pressed = key.char =='a'
#         print('alphanumeric key {} pressed; it was {}'.format(
#             key.char, 'faked' if injected else 'not faked'))
#     except AttributeError:
#         print('special key {} pressed'.format(
#             key))

# def on_release(key, injected):
#     print('{} released; it was {}'.format(
#         key, 'faked' if injected else 'not faked'))
#     if key == keyboard.Key.esc:
#         # Stop listener
#         return False

# # Collect events until released
# with keyboard.Listener(
#         on_press=on_press,
#         on_release=on_release) as listener:
#     listener.join()

# # ...or, in a non-blocking fashion:
# listener = keyboard.Listener(
#     on_press=on_press,
#     on_release=on_release)
# listener.start()

# def listen_robot_data():
#     while True:
#         actual_force = rtde_r.getActualTCPForce()
#         print(actual_force)

# rtde_c.moveUntilContact([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
# print("done")