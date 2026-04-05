import numpy as np

def calculate_head_tracking(pose_landmarks_list):
    """Calculates how much NAO needs to move its head to center the hips."""
    
    # 23 = Left Hip, 24 = Right Hip
    l_hip = pose_landmarks_list[23]
    r_hip = pose_landmarks_list[24]

    # Find the midpoint between the two hips
    hip_center_x = (l_hip.x + r_hip.x) / 2.0
    hip_center_y = (l_hip.y + r_hip.y) / 2.0

    # Calculate error from the exact center of the camera (0.5, 0.5)
    error_x = hip_center_x - 0.5
    error_y = hip_center_y - 0.5

    # --- THE DEADZONE ---
    # If the hips are close enough to the center, do nothing to prevent jitter!
    if abs(error_x) < 0.08: error_x = 0
    if abs(error_y) < 0.08: error_y = 0

    # --- THE PROPORTIONAL CONTROLLER (P-Gain) ---
    # Multiply the error by a factor to get an angle in radians. 
    # Smaller = smoother but slower. Larger = faster but might oscillate.
    p_gain_yaw = 0.15   
    p_gain_pitch = 0.15 

    # NAO HeadYaw: Positive is Left, Negative is Right
    # If error_x is positive (hips are on the right), we need a NEGATIVE yaw offset
    yaw_offset = -error_x * p_gain_yaw

    # NAO HeadPitch: Positive is Down, Negative is Up
    # If error_y is positive (hips are at the bottom), we need a POSITIVE pitch offset
    pitch_offset = error_y * p_gain_pitch

    return yaw_offset, pitch_offset

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
    NAO_LIMITS = {
        "RShoulderPitch": [-2.0857, 2.0857],
        "RShoulderRoll":  [-1.3265, 0.3142],
        "RElbowYaw":      [-2.0857, 2.0857],
        "RElbowRoll":     [0.0349, 1.5446],
        "RWristYaw":      [-1.8238, 1.8238]
    }

    pose_landmarks_list = detection_result.pose_world_landmarks[0]

    Lshoulder_Point = np.array([pose_landmarks_list[11].x, pose_landmarks_list[11].y, pose_landmarks_list[11].z])
    Rshoulder_Point = np.array([pose_landmarks_list[12].x, pose_landmarks_list[12].y, pose_landmarks_list[12].z])
    Lelbow_Point    = np.array([pose_landmarks_list[13].x, pose_landmarks_list[13].y, pose_landmarks_list[13].z])
    LWrist_Point    = np.array([pose_landmarks_list[15].x, pose_landmarks_list[15].y, pose_landmarks_list[15].z])
    Lhip_Point      = np.array([pose_landmarks_list[23].x, pose_landmarks_list[23].y, pose_landmarks_list[23].z])

    # Sistem de referință trunchi
    # X = lateral (dreapta omului), Y = jos, Z = spre cameră
    X = Rshoulder_Point - Lshoulder_Point
    X /= np.linalg.norm(X)

    Y = Lhip_Point - Lshoulder_Point   # în jos
    Y /= np.linalg.norm(Y)

    Z = np.cross(Y, X)                 # spre cameră
    Z /= np.linalg.norm(Z)

    Varm = Lelbow_Point - Lshoulder_Point
    Varm /= np.linalg.norm(Varm)

    Vforearm = LWrist_Point - Lelbow_Point
    Vforearm /= np.linalg.norm(Vforearm)

    # -------------------------------------------------------
    # SHOULDER PITCH
    # NAO Pitch = rotație în planul sagital (față/spate + sus/jos)
    # Când brațul e jos: pitch ~ +1.5, când e sus: pitch ~ -1.5
    # dot(Varm, Y) > 0 când brațul e în jos (Y e în jos)
    # dot(Varm, Z) = componenta față/spate
    R_Shoulder_Pitch = np.arctan2(np.dot(Varm, Y), -np.dot(Varm, Z))

    # SHOULDER ROLL
    # NAO Roll drept: negativ = braț depărtat lateral de corp
    # dot(Varm, X) > 0 când brațul merge spre dreapta omului
    # Oglindă: stânga om → dreapta NAO, deci Roll negativ când brațul e lateral stânga
    R_Shoulder_Roll = -np.arctan2(-np.dot(Varm, X),
                       np.sqrt(np.dot(Varm, Y)**2 + np.dot(Varm, Z)**2))

    R_Shoulder_Pitch = np.clip(R_Shoulder_Pitch, *NAO_LIMITS["RShoulderPitch"])
    R_Shoulder_Roll  = np.clip(R_Shoulder_Roll,  *NAO_LIMITS["RShoulderRoll"])

    # ELBOW ROLL - întotdeauna pozitiv pentru dreapta NAO
    R_Elbow_Roll = np.arccos(np.clip(np.dot(Varm, Vforearm), -1.0, 1.0))
    R_Elbow_Roll = np.clip(R_Elbow_Roll, *NAO_LIMITS["RElbowRoll"])

    # ELBOW YAW - mapare directă fără oglindire pe X
    # Pentru dreapta, Y_nao = -X_om (oglindă), Z_nao = -Y_om (sus)
    # Elbow Yaw - identic ca la dreapta, doar variabilele de unghi se schimbă
    Varm_nao     = np.array([np.dot(Varm, Z), -np.dot(Varm, X), -np.dot(Varm, Y)])
    Vforearm_nao = np.array([np.dot(Vforearm, Z), -np.dot(Vforearm, X), -np.dot(Vforearm, Y)])

    n_arm = np.cross(Varm_nao, Vforearm_nao)
    norm_n = np.linalg.norm(n_arm)

    if norm_n > 1e-4:
        n_arm /= norm_n

        cp, sp = np.cos(R_Shoulder_Pitch), np.sin(R_Shoulder_Pitch)
        cr, sr = np.cos(R_Shoulder_Roll),  np.sin(R_Shoulder_Roll)

        Y_local = np.array([-cp*sr, cr, sp*sr])
        Z_local = np.array([sp, 0, cp])

        R_Elbow_Yaw = np.arctan2(np.dot(n_arm, Y_local), -np.dot(n_arm, Z_local))
    else:
        R_Elbow_Yaw = 0.0

    R_Elbow_Roll     = np.clip(R_Elbow_Roll,     *NAO_LIMITS["RElbowRoll"])
    R_Elbow_Yaw      = np.clip(R_Elbow_Yaw,      *NAO_LIMITS["RElbowYaw"])


    # 3.5. WRIST YAW - Referință stabilă bazată pe trunchi
    # 3.5. WRIST YAW
    # wrist = np.array([left_hand_landmarks[0].x, left_hand_landmarks[0].z])
    # middle_mcp = np.array([left_hand_landmarks[9].x, left_hand_landmarks[9].z])

    index_mcp = np.array([left_hand_landmarks[5].x, left_hand_landmarks[5].z])
    pinky_mcp = np.array([left_hand_landmarks[17].x, left_hand_landmarks[17].z])

    dx = index_mcp[0] - pinky_mcp[0]
    dz = index_mcp[1] - pinky_mcp[1]

    raw_wrist_angle = np.arctan2(dz, dx)

# Detectează orientarea palmei folosind Z-ul landmark-urilor
# Dacă palma e spre cameră, index_mcp.z < pinky_mcp.z (în MediaPipe world coords)
    palm_facing_camera = left_hand_landmarks[5].z < left_hand_landmarks[17].z

# Aplică offset de π dacă palma e spre cameră (față de exterior)
    if palm_facing_camera:
        raw_wrist_angle += np.pi

# Inversăm pentru oglindă
    R_Wrist_Yaw = -(raw_wrist_angle - np.pi / 2)

    R_Wrist_Yaw = np.clip(R_Wrist_Yaw, *NAO_LIMITS["RWristYaw"])

    # HAND OPEN/CLOSE
    FINGERTIPS = [8, 12, 16, 20]
    FINGER_MCP  = [5,  9, 13, 17]
    wrist = np.array([left_hand_landmarks[0].x,
                      left_hand_landmarks[0].y,
                      left_hand_landmarks[0].z])
    total_score = 0.0
    for tip_idx, mcp_idx in zip(FINGERTIPS, FINGER_MCP):
        tip = np.array([left_hand_landmarks[tip_idx].x,
                        left_hand_landmarks[tip_idx].y,
                        left_hand_landmarks[tip_idx].z])
        mcp = np.array([left_hand_landmarks[mcp_idx].x,
                        left_hand_landmarks[mcp_idx].y,
                        left_hand_landmarks[mcp_idx].z])
        dist_tip = np.linalg.norm(tip - wrist)
        dist_mcp = np.linalg.norm(mcp - wrist)
        total_score += dist_tip / (dist_mcp + 1e-6)

    avg_score = total_score / len(FINGERTIPS)
    R_Hand = float(np.clip((avg_score - 1.0) / 1.5, 0.0, 1.0))

    return R_Shoulder_Pitch, R_Shoulder_Roll, R_Elbow_Roll, R_Elbow_Yaw, R_Wrist_Yaw, R_Hand