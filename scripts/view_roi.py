import cv2
import numpy as np

zone = [(
      145,
      295
    ),
    (
      191,
      326
    ),
    (
      242,
      361
    ),
    (
      311,
      405
    ),
    (
      385,
      451
    ),
    (
      478,
      504
    ),
    (
      578,
      560
    ),
    (
      693,
      617
    ),
    (
      807,
      669
    ),
    (
      924,
      718
    ),
    (
      296,
      719
    ),
    (
      210,
      595
    ),
    (
      148,
      490
    ),
    (
      110,
      414
    ),
    (
      87,
      361
    ),
    (
      72,
      326
    )
]

frame = cv2.imread("snapshots\image2.jpg")
pts = np.array(zone, dtype=np.int32)
cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
cv2.imshow("Zone Check", frame)
cv2.waitKey(0)