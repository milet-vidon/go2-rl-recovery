"""Extract labeled, uncropped frames for visual validation of local rollouts."""
import argparse
from pathlib import Path
import cv2
import numpy as np


def build(video, output, seconds):
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f'Cannot decode {video}')
    tiles = []
    try:
        for second in seconds:
            capture.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f'Cannot decode {video} at {second}s')
            frame = cv2.resize(frame, (640, 360))
            tile = np.zeros((390, 640, 3), np.uint8)
            tile[:360] = frame
            cv2.putText(tile, f'{second:.2f} s', (10, 382), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
            tiles.append(tile)
    finally:
        capture.release()
    if len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i+2]) for i in range(0, len(tiles), 2)])
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet):
        raise RuntimeError(f'Cannot write {output}')
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--seconds', type=float, nargs='+', default=[0, .5, 1, 2, 5, 10])
    args = parser.parse_args()
    build(args.video, args.output, args.seconds)
