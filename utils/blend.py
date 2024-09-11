import sys
import argparse
import numpy as np
import os
import cv2
import glob
import dlib
import subprocess
import shutil

# Add the parent directory to the Python path explicitly
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Now import from the utils module
from utils.common import get_versioned_filename

face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")

def main(samelength_path, pre_blend_path, chin_extension, chin_width, side_point_drop):
    # Extract frames from the samelength video
    print('Extracting frames from samelength video')
    same_length_dir = os.path.join(os.path.dirname(samelength_path), 'samelength')
    if not os.path.exists(same_length_dir):
        os.makedirs(same_length_dir)
    extract_frames_from_video(samelength_path, same_length_dir)

    # Extract frames from the pre_blend video
    print('Extracting frames from pre_blend video')
    pre_blend_dir = os.path.join(os.path.dirname(pre_blend_path), 'pre_blend')
    if not os.path.exists(pre_blend_dir):
        os.makedirs(pre_blend_dir)
    extract_frames_from_video(pre_blend_path, pre_blend_dir)

    print('Tracking Face of Lip-synced (LipSick) video please wait..')

    # Call the blending function
    blend_videos(same_length_dir, pre_blend_dir, samelength_path, pre_blend_path, chin_extension, chin_width, side_point_drop)

def extract_frames_from_video(video_path, save_dir):
    videoCapture = cv2.VideoCapture(video_path)
    frames = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(frames):
        ret, frame = videoCapture.read()
        if not ret:
            break
        result_path = os.path.join(save_dir, str(i).zfill(6) + '.jpg')
        cv2.imwrite(result_path, frame)
    return (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

def load_landmark_dlib(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    if not faces:
        raise ValueError("No faces found in the image.")
    shape = landmark_predictor(gray, faces[0])
    landmarks = np.array([[p.x, p.y] for p in shape.parts()])
    return landmarks

def alpha_blend_face(original_frame, generated_frame, landmarks,
                     chin_extension, chin_width, side_point_drop):
    mask = np.zeros(original_frame.shape[:2], dtype=np.float32)
    
    # Extend the mask by including more points below the chin
    face_outline = landmarks[0:17]  # Jawline
    nose_bridge = landmarks[27:31]  # Nose bridge
    mouth_outline = landmarks[48:61]  # Outer lip
    
    # Add points below the chin
    chin_point = landmarks[8]
    below_chin_points = np.array([
        [chin_point[0], chin_point[1] + chin_extension],
        [chin_point[0] - chin_width, chin_point[1] + side_point_drop],
        [chin_point[0] + chin_width, chin_point[1] + side_point_drop]
    ])
    
    # Combine all points
    points = np.concatenate((face_outline, nose_bridge, mouth_outline, below_chin_points))
    
    # Create convex hull
    hull = cv2.convexHull(points)
    
    if len(hull) < 3:
        raise ValueError("Convex hull could not be formed. Not enough points.")
    
    cv2.fillConvexPoly(mask, hull, 1.0)
    mask = cv2.GaussianBlur(mask, (51, 51), 30)
    mask = mask[..., np.newaxis]
    blended_frame = original_frame * (1 - mask) + generated_frame * mask
    return blended_frame.astype(np.uint8)

def blend_videos(same_length_dir, pre_blend_dir, samelength_path, pre_blend_path, chin_extension, chin_width, side_point_drop):
    # Get frames from both videos
    same_length_frame_path_list = glob.glob(os.path.join(same_length_dir, '*.jpg'))
    same_length_frame_path_list.sort()

    pre_blend_frame_path_list = glob.glob(os.path.join(pre_blend_dir, '*.jpg'))
    pre_blend_frame_path_list.sort()
    pre_blend_landmark_data = np.array([load_landmark_dlib(frame) for frame in pre_blend_frame_path_list])

    # Blend frames from pre_blend video onto samelength video using Alpha blending
    output_video_path = samelength_path.replace('samelength.mp4', '_lipsick_blend.mp4')
    output_video_path = get_versioned_filename(output_video_path)  # Ensure unique filename
    video_size = (int(cv2.VideoCapture(samelength_path).get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(cv2.VideoCapture(samelength_path).get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videowriter = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, video_size)

    for i, (same_length_frame_path, pre_blend_frame_path, landmark_data) in enumerate(zip(same_length_frame_path_list, pre_blend_frame_path_list, pre_blend_landmark_data)):
        sys.stdout.write(f'\rAlpha blending frame {i+1}/{len(same_length_frame_path_list)}')
        sys.stdout.flush()  # Make sure to flush the output buffer

        same_length_frame = cv2.imread(same_length_frame_path)
        pre_blend_frame = cv2.imread(pre_blend_frame_path)

        try:
            blended_frame = alpha_blend_face(same_length_frame, pre_blend_frame, landmark_data, chin_extension, chin_width, side_point_drop)
        except ValueError as e:
            print(f"Skipping frame {i+1} due to error: {e}")
            blended_frame = same_length_frame  # Use original frame if blending fails

        videowriter.write(blended_frame)

    videowriter.release()

    # Store the output video path for later use
    output_video_path = output_video_path

    # Add audio to the blended video
    final_video_path = get_versioned_filename(output_video_path.replace('_lipsick_blend.mp4', 'LipSick_Blend.mp4'))

    cmd = f'ffmpeg -i "{output_video_path}" -i "{pre_blend_path}" -c:v libx264 -crf 23 -c:a aac -strict experimental "{final_video_path}"'
    subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Clean up intermediate files
    os.remove(output_video_path)
    os.remove(samelength_path)
    os.remove(pre_blend_path)
    shutil.rmtree(same_length_dir)
    shutil.rmtree(pre_blend_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Alpha blend two videos based on facial landmarks')
    parser.add_argument('--samelength_video_path', type=str, required=True, help='Path to the samelength.mp4 video')
    parser.add_argument('--pre_blend_video_path', type=str, required=True, help='Path to the pre_blend.mp4 video')
    parser.add_argument("--chin_extension", type=int, default=100, help="Chin extension")
    parser.add_argument("--chin_width", type=int, default=30, help="Chin width")
    parser.add_argument("--side_point_drop", type=int, default=100, help="Side point drop")
    args = parser.parse_args()

    main(args.samelength_video_path, args.pre_blend_video_path, args.chin_extension, args.chin_width, args.side_point_drop)
