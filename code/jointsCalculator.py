import numpy as np

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