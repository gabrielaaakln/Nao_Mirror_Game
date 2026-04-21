import numpy as np

# ── Head tracking / scanning state ───────────────────────────────────────────
FOV_H = np.radians(60.9)
FOV_V = np.radians(47.6)

YAW_MIN,   YAW_MAX   = -2.0857,  2.0857
PITCH_MIN, PITCH_MAX = -0.6720,  0.5149

_scan_yaw       = 0.0          # current scan position
_scan_direction = 1            # +1 = sweeping left, -1 = sweeping right
_scan_step      = 0.04         # radians per call (~2.3° per frame)
_scan_pitch     = 0.0          # keep head level while scanning

_track_yaw   = 0.0
_track_pitch = 0.0


R_HAND_DEFAULT_ANGLES = {
    "RShoulderPitch": 1.44,
    "RShoulderRoll": -0.22,
    "RElbowYaw": 1.19,
    "RElbowRoll": 0.40,
    "RWristYaw": 0.1,
    "RHand": 0.30
}

L_HAND_DEFAULT_ANGLES = {
    "LShoulderPitch": 1.47,
    "LShoulderRoll": 0.19,
    "LElbowYaw": -1.18,
    "LElbowRoll": -0.40,
    "LWristYaw": 0.09,
    "LHand": 0.30
}

def scan_head():
    """
    Sweeps HeadYaw left and right until a person is detected.
    Call this every frame when pose_landmarks is None/empty.
    Returns (head_yaw, head_pitch).
    """
    global _scan_yaw, _scan_direction

    _scan_yaw += _scan_step * _scan_direction

    if _scan_yaw >= YAW_MAX * 0.85:
        _scan_direction = -1
    elif _scan_yaw <= YAW_MIN * 0.85:
        _scan_direction = 1

    return float(_scan_yaw), float(_scan_pitch)


def calculate_head_tracking(pose_landmarks, kp_yaw=0.3, kp_pitch=0.3):
    """
    Centers the robot's head to frame the person from the waist up.
    Uses a visibility fallback to prevent snapping if the hips go out of frame.
    """
    global _track_yaw, _track_pitch, _scan_yaw, _scan_pitch

    NOSE  = 0
    L_SHOULDER = 11
    R_SHOULDER = 12
    L_HIP = 23
    R_HIP = 24

    # 1. Horizontal Tracking (X) - Always use shoulders for stability
    shoulder_cx = (pose_landmarks[L_SHOULDER].x + pose_landmarks[R_SHOULDER].x) / 2.0
    target_x = shoulder_cx

    # 2. Vertical Tracking (Y) - Frame from Hips to Head
    nose_y = pose_landmarks[NOSE].y
    
    # MediaPipe gives a visibility score (0.0 to 1.0) for every landmark.
    # We check if the hips are actually inside the camera view.
    hip_visibility = (pose_landmarks[L_HIP].visibility + pose_landmarks[R_HIP].visibility) / 2.0

    if hip_visibility > 0.5:
        # Hips are visible! Center the camera halfway between the nose and the hips.
        hip_cy = (pose_landmarks[L_HIP].y + pose_landmarks[R_HIP].y) / 2.0
        target_y = (nose_y + hip_cy) / 2.0
    else:
        # Hips are out of frame. Estimate where the waist is so the head doesn't snap up.
        # A human torso is roughly twice the distance from the nose to the shoulders.
        shoulder_cy = (pose_landmarks[L_SHOULDER].y + pose_landmarks[R_SHOULDER].y) / 2.0
        nose_to_shoulder_dist = shoulder_cy - nose_y
        
        estimated_hip_y = shoulder_cy + (nose_to_shoulder_dist * 2.0)
        target_y = (nose_y + estimated_hip_y) / 2.0

    # Calculate error (distance from the center of the screen, which is 0.5)
    err_x = target_x - 0.5   
    err_y = target_y - 0.5   

    # DEADZONE: If the person is already roughly centered, don't move the head.
    # This stops the motors from whining and jittering over 1% errors.
    DEADZONE = 0.05
    if abs(err_x) < DEADZONE: err_x = 0
    if abs(err_y) < DEADZONE: err_y = 0

    # Update the angles based on error
    _track_yaw   += -kp_yaw * err_x * FOV_H
    _track_pitch +=  kp_pitch * err_y * FOV_V

    # Clamp to NAO's physical joint limits
    _track_yaw   = float(np.clip(_track_yaw,   YAW_MIN,   YAW_MAX))
    _track_pitch = float(np.clip(_track_pitch, PITCH_MIN, PITCH_MAX))

    # Keep the scanner in sync so there's no jump if tracking is lost entirely
    _scan_yaw   = _track_yaw
    _scan_pitch = _track_pitch

    return _track_yaw, _track_pitch


def right_hand_joints(detection_result, right_hand_landmarks):
    NAO_LIMITS = {
        "LShoulderPitch": [-2.0857, 2.0857],
        "LShoulderRoll":  [-0.3142, 1.3265],
        "LElbowYaw":      [-2.0857, 2.0857],
        "LElbowRoll":     [-1.5446, -0.0349],
        "LWristYaw":      [-1.8238, 1.8238],
        "LHand":          [0.0, 1.0]
    }

    pose_landmarks_list = detection_result.pose_world_landmarks[0]

    # Extragem landmark-urile pentru brațul DREPT al omului
    Rshoulder_Point = np.array([pose_landmarks_list[12].x, pose_landmarks_list[12].y, pose_landmarks_list[12].z])
    Lshoulder_Point = np.array([pose_landmarks_list[11].x, pose_landmarks_list[11].y, pose_landmarks_list[11].z])
    Relbow_Point    = np.array([pose_landmarks_list[14].x, pose_landmarks_list[14].y, pose_landmarks_list[14].z])
    RWrist_Point    = np.array([pose_landmarks_list[16].x, pose_landmarks_list[16].y, pose_landmarks_list[16].z])
    Rhip_Point      = np.array([pose_landmarks_list[24].x, pose_landmarks_list[24].y, pose_landmarks_list[24].z])

    # Sistem de referință trunchi (Om)
    X = Lshoulder_Point - Rshoulder_Point # Spre stânga omului
    X /= np.linalg.norm(X)

    Y = Rhip_Point - Rshoulder_Point      # În jos
    Y /= np.linalg.norm(Y)

    Z = np.cross(Y, X)                    # Spre cameră / În față
    Z /= np.linalg.norm(Z)

    # Vectori braț și antebraț (Om)
    Varm = Relbow_Point - Rshoulder_Point
    Varm /= np.linalg.norm(Varm)

    Vforearm = RWrist_Point - Relbow_Point
    Vforearm /= np.linalg.norm(Vforearm)

    # 1. Calculăm Shoulder Pitch și Roll (Logica ta funcționează perfect pentru mirroring)
    L_Shoulder_Pitch = np.arctan2(np.dot(Varm, Y), np.dot(Varm, Z))
    L_Shoulder_Roll  = -np.arctan2(np.dot(Varm, X), np.sqrt(np.dot(Varm, Y)**2 + np.dot(Varm, Z)**2))

    # IMPORTANT: Facem clamp la umăr ÎNAINTE de a calcula cotul. 
    # Dacă NAO nu poate atinge unghiul tău de umăr, referința pentru cot trebuie calculată
    # de la poziția fizică în care se află brațul robotului, nu al tău.
    L_Shoulder_Pitch = np.clip(L_Shoulder_Pitch, *NAO_LIMITS["LShoulderPitch"])
    L_Shoulder_Roll  = np.clip(L_Shoulder_Roll,  *NAO_LIMITS["LShoulderRoll"])

    # 2. Elbow Roll
    L_Elbow_Roll = -np.arccos(np.clip(np.dot(Varm, Vforearm), -1.0, 1.0))

    # 3. Elbow Yaw - Mapare pe cinematica robotului NAO
    # Mapăm direcțiile brațului omului direct pe sistemul de coordonate stâng de la NAO:
    # X_nao (Față) = Z_om, Y_nao (Stânga) = -X_om, Z_nao (Sus) = -Y_om
    Varm_nao = np.array([np.dot(Varm, Z), -np.dot(Varm, X), -np.dot(Varm, Y)])
    Vforearm_nao = np.array([np.dot(Vforearm, Z), -np.dot(Vforearm, X), -np.dot(Vforearm, Y)])

    # Normala planului format de braț și antebraț
    n_arm = np.cross(Varm_nao, Vforearm_nao)
    norm_n = np.linalg.norm(n_arm)

    if norm_n > 1e-4: # Brațul nu este complet întins (gimbal lock natural)
        n_arm /= norm_n

        # Reconstruim orientarea umărului robotului folosind unghiurile calculate
        cp, sp = np.cos(L_Shoulder_Pitch), np.sin(L_Shoulder_Pitch)
        cr, sr = np.cos(L_Shoulder_Roll),  np.sin(L_Shoulder_Roll)

        # Deduse din înmulțirea matricilor RotY(Pitch) * RotZ(Roll)
        # Acestea sunt axele locale Y și Z ale brațului superior DUPĂ rotirea din umăr
        Y_local = np.array([-cp*sr, cr, sp*sr]) 
        Z_local = np.array([sp, 0, cp])         

        # Calculăm Yaw proiectând normala brațului pe axele locale
        L_Elbow_Yaw = np.arctan2(np.dot(n_arm, Y_local), -np.dot(n_arm, Z_local))
    else:
        # Dacă brațul e perfect întins, Yaw-ul este incalculabil geometric. Păstrăm 0.
        L_Elbow_Yaw = 0.0

    # 3.5. WRIST YAW - Referință stabilă bazată pe trunchi

    if right_hand_landmarks is None:
        L_Wrist_Yaw = L_HAND_DEFAULT_ANGLES["LWristYaw"]
        L_Hand = L_HAND_DEFAULT_ANGLES["LHand"]
    else:
        # landmark 5 = INDEX_MCP, landmark 17 = PINKY_MCP (baza degetelor, foarte stabile)
        index_mcp = np.array([right_hand_landmarks[5].x, right_hand_landmarks[5].y])
        pinky_mcp = np.array([right_hand_landmarks[17].x, right_hand_landmarks[17].y])

        dx = index_mcp[0] - pinky_mcp[0]
        dy = index_mcp[1] - pinky_mcp[1]

        raw_wrist_angle = np.arctan2(dy, dx)
        L_Wrist_Yaw = raw_wrist_angle + np.pi/2

        # HAND OPEN/CLOSE
        FINGERTIPS = [8, 12, 16, 20]
        FINGER_MCP  = [5,  9, 13, 17]
        wrist = np.array([right_hand_landmarks[0].x,
                        right_hand_landmarks[0].y,
                        right_hand_landmarks[0].z])
        total_score = 0.0
        for tip_idx, mcp_idx in zip(FINGERTIPS, FINGER_MCP):
            tip = np.array([right_hand_landmarks[tip_idx].x,
                            right_hand_landmarks[tip_idx].y,
                            right_hand_landmarks[tip_idx].z])
            mcp = np.array([right_hand_landmarks[mcp_idx].x,
                            right_hand_landmarks[mcp_idx].y,
                            right_hand_landmarks[mcp_idx].z])
            dist_tip = np.linalg.norm(tip - wrist)
            dist_mcp = np.linalg.norm(mcp - wrist)
            total_score += dist_tip / (dist_mcp + 1e-6)

        avg_score = total_score / len(FINGERTIPS)
        L_Hand = float(np.clip((avg_score - 1.0) / 1.5, 0.0, 1.0))

    # Limităm unghiurile
    L_Elbow_Roll     = np.clip(L_Elbow_Roll,     *NAO_LIMITS["LElbowRoll"])
    L_Elbow_Yaw      = np.clip(L_Elbow_Yaw,      *NAO_LIMITS["LElbowYaw"])
    L_Wrist_Yaw      = np.clip(L_Wrist_Yaw,      *NAO_LIMITS["LWristYaw"])

    return L_Shoulder_Pitch, L_Shoulder_Roll, L_Elbow_Roll, L_Elbow_Yaw, L_Wrist_Yaw, L_Hand



def left_hand_joints(detection_result, left_hand_landmarks):
    # Această funcție controlează BRAȚUL DREPT al lui NAO (pentru a oglindi mâna ta stângă)
    NAO_LIMITS = {
        "RShoulderPitch": [-2.0857, 2.0857],
        "RShoulderRoll":  [-1.3265, 0.3142],
        "RElbowYaw":      [-2.0857, 2.0857],
        "RElbowRoll":     [0.0349, 1.5446],
        "RWristYaw":      [-1.8238, 1.8238]
    }

    pose = detection_result.pose_world_landmarks[0]

    Lshoulder = np.array([pose[11].x, pose[11].y, pose[11].z])
    Rshoulder = np.array([pose[12].x, pose[12].y, pose[12].z])
    Lhip      = np.array([pose[23].x, pose[23].y, pose[23].z])
    Rhip      = np.array([pose[24].x, pose[24].y, pose[24].z])

    Z_torso = (Lshoulder + Rshoulder)/2 - (Lhip + Rhip)/2
    Z_torso /= np.linalg.norm(Z_torso)

    Y_torso = Lshoulder - Rshoulder
    Y_torso /= np.linalg.norm(Y_torso)

    X_torso = np.cross(Y_torso, Z_torso)
    X_torso /= np.linalg.norm(X_torso)

    Y_torso = np.cross(Z_torso, X_torso)

    Lelbow = np.array([pose[13].x, pose[13].y, pose[13].z])
    Lwrist = np.array([pose[15].x, pose[15].y, pose[15].z])

    Varm_world = Lelbow - Lshoulder
    Varm_world /= np.linalg.norm(Varm_world)
    # FIX OGLINDĂ: Am adăugat minus la Y_torso
    Varm = np.array([np.dot(Varm_world, X_torso), -np.dot(Varm_world, Y_torso), np.dot(Varm_world, Z_torso)])

    Vforearm_world = Lwrist - Lelbow
    Vforearm_world /= np.linalg.norm(Vforearm_world)
    # FIX OGLINDĂ: Am adăugat minus la Y_torso
    Vforearm = np.array([np.dot(Vforearm_world, X_torso), -np.dot(Vforearm_world, Y_torso), np.dot(Vforearm_world, Z_torso)])

    R_Shoulder_Pitch = np.arctan2(-Varm[2], Varm[0])
    R_Shoulder_Roll  = np.arcsin(np.clip(Varm[1], -1.0, 1.0))

    R_Shoulder_Pitch = np.clip(R_Shoulder_Pitch, *NAO_LIMITS["RShoulderPitch"])
    R_Shoulder_Roll  = np.clip(R_Shoulder_Roll,  *NAO_LIMITS["RShoulderRoll"])
    
    R_Elbow_Roll = np.arccos(np.clip(np.dot(Varm, Vforearm), -1.0, 1.0))

    cp, sp = np.cos(R_Shoulder_Pitch), np.sin(R_Shoulder_Pitch)
    cr, sr = np.cos(R_Shoulder_Roll),  np.sin(R_Shoulder_Roll)
    Ry_pitch = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz_roll  = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    R_shoulder = Ry_pitch @ Rz_roll
    
    n_arm_nao = np.cross(Varm, Vforearm)
    n_arm_2 = R_shoulder.T @ n_arm_nao
    
    if np.linalg.norm(n_arm_2) > 1e-4:
        R_Elbow_Yaw = np.arctan2(-n_arm_2[1], n_arm_2[2])
    else:
        R_Elbow_Yaw = 0.0



    if left_hand_landmarks is None:
        R_Wrist_Yaw = R_HAND_DEFAULT_ANGLES["RWristYaw"]
        R_Hand = R_HAND_DEFAULT_ANGLES["RHand"]
    else:
        wrist_lm = np.array([left_hand_landmarks[0].x, left_hand_landmarks[0].y, left_hand_landmarks[0].z])
        index_lm = np.array([left_hand_landmarks[5].x, left_hand_landmarks[5].y, left_hand_landmarks[5].z])
        pinky_lm = np.array([left_hand_landmarks[17].x, left_hand_landmarks[17].y, left_hand_landmarks[17].z])
    
        # FIX PALMĂ: Am inversat pinky_lm cu index_lm pentru a roti palma cu 180 de grade
        palm_normal_world = np.cross(pinky_lm - wrist_lm, index_lm - wrist_lm)
        palm_normal_world /= (np.linalg.norm(palm_normal_world) + 1e-6)
    
        # FIX OGLINDĂ ȘI LA PALMĂ: Axa Y este inversată
        palm_normal_nao = np.array([
            np.dot(palm_normal_world, X_torso), 
            -np.dot(palm_normal_world, Y_torso), 
            np.dot(palm_normal_world, Z_torso)
        ])
    
        cy, sy = np.cos(R_Elbow_Yaw), np.sin(R_Elbow_Yaw)
        Rx_yaw = np.array([[1, 0, 0], [0, cy, -sy], [0, sy, cy]])
        ce, se = np.cos(R_Elbow_Roll), np.sin(R_Elbow_Roll)
        Rz_eroll = np.array([[ce, -se, 0], [se, ce, 0], [0, 0, 1]])
    
        R_forearm = R_shoulder @ Rx_yaw @ Rz_eroll
        palm_normal_3 = R_forearm.T @ palm_normal_nao
        R_Wrist_Yaw = np.arctan2(palm_normal_3[2], palm_normal_3[1])

        FINGERTIPS, FINGER_MCP = [8, 12, 16, 20], [5, 9, 13, 17]
        total_score = sum(
            np.linalg.norm(np.array([left_hand_landmarks[t].x, left_hand_landmarks[t].y, left_hand_landmarks[t].z]) - wrist_lm) /
            (np.linalg.norm(np.array([left_hand_landmarks[m].x, left_hand_landmarks[m].y, left_hand_landmarks[m].z]) - wrist_lm) + 1e-6)
            for t, m in zip(FINGERTIPS, FINGER_MCP)
        )
        R_Hand = float(np.clip(((total_score / len(FINGERTIPS)) - 1.0) / 1.5, 0.0, 1.0))

    R_Elbow_Roll     = np.clip(R_Elbow_Roll,     *NAO_LIMITS["RElbowRoll"])
    R_Elbow_Yaw      = np.clip(R_Elbow_Yaw,      *NAO_LIMITS["RElbowYaw"])
    R_Wrist_Yaw      = np.clip(R_Wrist_Yaw,      *NAO_LIMITS["RWristYaw"])

    return R_Shoulder_Pitch, R_Shoulder_Roll, R_Elbow_Roll, R_Elbow_Yaw, R_Wrist_Yaw, R_Hand





#metrics
# At module level
_angle_history = {"L": [], "R": []}

def log_stability(side, joints):
    _angle_history[side].append(joints)
    if len(_angle_history[side]) > 100:
        _angle_history[side].pop(0)

def get_stability_report():
    report = {}
    for side, history in _angle_history.items():
        if len(history) > 10:
            arr = np.array(history)
            report[side] = {
                "std_per_joint": np.std(arr, axis=0).tolist(),
                "mean_per_joint": np.mean(arr, axis=0).tolist()
            }
    return report