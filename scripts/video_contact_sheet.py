"""Extract labeled, uncropped frames for visual validation of local rollouts."""
import argparse
from pathlib import Path
import cv2
import numpy as np


def build(video, output, seconds, jpeg_quality=85):
    if not 1 <= jpeg_quality <= 100:
        raise ValueError('JPEG quality must be between 1 and 100')
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
            # Preserve the camera aspect ratio, including wide paired reviews.
            scale = min(640 / frame.shape[1], 360 / frame.shape[0])
            frame = cv2.resize(frame, (round(frame.shape[1] * scale), round(frame.shape[0] * scale)))
            tile = np.zeros((390, 640, 3), np.uint8)
            x, y = (640 - frame.shape[1]) // 2, (360 - frame.shape[0]) // 2
            tile[y:y + frame.shape[0], x:x + frame.shape[1]] = frame
            cv2.putText(tile, f'{second:.2f} s', (10, 382), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
            tiles.append(tile)
    finally:
        capture.release()
    if len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i+2]) for i in range(0, len(tiles), 2)])
    output.parent.mkdir(parents=True, exist_ok=True)
    options = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] if output.suffix.lower() in ('.jpg', '.jpeg') else []
    if not cv2.imwrite(str(output), sheet, options):
        raise RuntimeError(f'Cannot write {output}')
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--seconds', type=float, nargs='+', default=[0, .5, 1, 2, 5, 10])
    parser.add_argument('--jpeg_quality', type=int, default=85, help='QA sheet encoding only; source video is never modified.')
    args = parser.parse_args()
    build(args.video, args.output, args.seconds, args.jpeg_quality)
