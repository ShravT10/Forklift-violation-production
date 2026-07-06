import cv2
import numpy as np

s_zone = [
    (145, 295), (191, 326), (242, 361), (311, 405), (385, 451),
    (478, 504), (578, 560), (693, 617), (807, 669), (924, 718),
    (296, 719), (210, 595), (148, 490), (110, 414), (87, 361),
    (72, 326),
]
e_zone = [
    (248, 250), (338, 281), (427, 305), (502, 319), (538, 334),
    (504, 488), (530, 508), (559, 513), (582, 488), (666, 375),
    (872, 451), (1033, 504), (1181, 546), (1279, 572), (1278, 304),
    (710, 219), (553, 178), (401, 164),
]

frame = cv2.imread("dataset-maker\dataset\session_20260702_134850\img_20260702_134850_570834.jpg")
pts = np.array(s_zone, dtype=np.int32)
pts2 = np.array(e_zone, dtype=np.int32)
cv2.polylines(frame, [pts,pts2], isClosed=True, color=(0, 255, 0), thickness=2)
cv2.imshow("Zone Check", frame)
cv2.waitKey(0)