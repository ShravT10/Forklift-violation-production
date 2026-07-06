import cv2, numpy as np

IMG = "snapshots\\i1.jpg"   # image/frame path
OUT = "Regions/exc_points.txt"         # output filename (note: not valid JSON)
CLOSE = True                    # False=polyline, True=polygon

pts = []
base = cv2.imread(IMG)
img = base.copy()

def redraw():
    global img
    img = base.copy()
    for i, p in enumerate(pts):
        cv2.circle(img, p, 4, (0, 255, 0), -1)
        cv2.putText(img, str(i), (p[0] + 5, p[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    if len(pts) > 1:
        cv2.polylines(img, [np.array(pts, np.int32)], CLOSE, (0, 0, 255), 2)

def mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append((x, y))
        redraw()

cv2.namedWindow("ROI Tool")
cv2.setMouseCallback("ROI Tool", mouse)
redraw()

while True:
    cv2.imshow("ROI Tool", img)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('u') and pts:
        pts.pop()
        redraw()
    elif key == ord('s'):
        # Write a Python-literal style file with tuple formatting (not valid JSON)
        with open(OUT, "w") as f:
            f.write('{\n')
            f.write('  "points": [\n')
            for i, p in enumerate(pts):
                f.write('    (\n')
                f.write(f'      {p[0]},\n')
                f.write(f'      {p[1]}\n')
                f.write('    )')
                if i != len(pts) - 1:
                    f.write(',\n')
                else:
                    f.write('\n')
            f.write('  ]\n')
            f.write('}\n')
        print("Saved (Python-literal format):", pts)
    elif key == ord('q'):
        break

cv2.destroyAllWindows()