import os
import argparse
import time
import tensorflow as tf
import tensorflow_hub as hub
import imageio
import numpy as np
from PIL import Image

SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
OUTPUT_VIDEO_PATH = "output.mp4"

model = hub.load(SAVED_MODEL_PATH)

def preprocess_image(image):
    hr_image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
    hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
    return tf.expand_dims(hr_image, 0)

def process_video(input_video_path, output_video_path):
    video_reader = imageio.get_reader(input_video_path)
    fps = video_reader.get_meta_data()['fps']
    video_writer = imageio.get_writer(output_video_path, fps=fps)

    for i, frame in enumerate(video_reader):
        frame_tensor = preprocess_image(frame)
        sr_frame = model(frame_tensor)
        sr_frame = tf.squeeze(sr_frame)
        
        sr_frame = tf.clip_by_value(sr_frame, 0, 255)
        sr_frame = Image.fromarray(tf.cast(sr_frame, tf.uint8).numpy())
        video_writer.append_data(np.array(sr_frame))

    video_writer.close()

def main():
    parser = argparse.ArgumentParser(description="Apply super resolution on video using TF Hub model.")
    parser.add_argument('--superres', type=str, required=True, help="Super resolution model (e.g., GFPGAN-iv).")
    parser.add_argument('input_video', type=str, help="Input video file path.")
    parser.add_argument('-ia', '--input_audio', type=str, required=True, help="Input audio file path.")
    parser.add_argument('-o', '--output_video', type=str, required=True, help="Output video file path.")
    
    args = parser.parse_args()

    if args.superres != "GFPGAN-iv":
        print(f"Unsupported super resolution model: {args.superres}")
        return

    process_video(args.input_video, args.output_video)

if __name__ == "__main__":
    main()
