import time
import airsim
import numpy as np
from navigation.logger import log_step


def run_normal_navigation(
    env,
    log_file,
    start_from_current_state=False
):
    client = env.drone
    start = time.time()
    success = True

    # ---------------- FULL NOMINAL PATH ----------------
    full_path = [
        airsim.Vector3r(502, 501.5, 0),
        airsim.Vector3r(504, 502.5, 0),
        airsim.Vector3r(505, 503.0, 0),

        airsim.Vector3r(507, 502.0, 0),
        airsim.Vector3r(509, 501.0, 0),
        airsim.Vector3r(511, 500.0, 0),

        airsim.Vector3r(513, 501.5, 0),
        airsim.Vector3r(515, 503.0, 0),
        airsim.Vector3r(517, 504.0, 0),

        airsim.Vector3r(519, 502.0, 0),
        airsim.Vector3r(521, 499.0, 0),
        airsim.Vector3r(524, 497.0, 0),

        airsim.Vector3r(525, 499.0, 0),
        airsim.Vector3r(526, 501.0, 0),
        airsim.Vector3r(527, 502.0, 0),

        airsim.Vector3r(529, 500.0, 0),
        airsim.Vector3r(530, 498.0, 0),
        airsim.Vector3r(531, 497.0, 0),

        airsim.Vector3r(534, 498.0, 0),
        airsim.Vector3r(538, 499.0, 0),
        airsim.Vector3r(542, 500.0, 0),
    ]

    # ---------------- RESUME FROM CURRENT STATE ----------------
    if start_from_current_state:
        state = client.getMultirotorState()
        curr_pos = state.kinematics_estimated.position

        curr_xy = np.array([curr_pos.x_val, curr_pos.y_val])

        # find nearest waypoint
        dists = []
        for wp in full_path:
            wp_xy = np.array([wp.x_val, wp.y_val])
            dists.append(np.linalg.norm(curr_xy - wp_xy))

        nearest_idx = int(np.argmin(dists))

        # resume path from nearest waypoint
        path = full_path[nearest_idx:]

        print(
            f"🛡️ Fallback navigation resumed from waypoint {nearest_idx} "
            f"({path[0].x_val:.1f}, {path[0].y_val:.1f})"
        )

    else:
        path = full_path

    # ---------------- START PATH FOLLOWING ----------------
    task = client.moveOnPathAsync(
        path,
        velocity=3.0,
        timeout_sec=60,
        drivetrain=airsim.DrivetrainType.ForwardOnly,
        yaw_mode=airsim.YawMode(is_rate=False),
        lookahead=5,
        adaptive_lookahead=1
    )

    # ---------------- MONITOR ----------------
    while time.time() - start < 60:
        state = client.getMultirotorState()
        pos = state.kinematics_estimated.position
        t = time.time() - start

        collision = client.simGetCollisionInfo()
        if collision.has_collided:
            success = False
            client.cancelLastTask()
            break

        log_step(log_file, [
            round(t, 2),
            pos.x_val, pos.y_val, pos.z_val,
            0.0,
            collision.has_collided,
            success,
            False
        ])

        time.sleep(0.1)

    return success
