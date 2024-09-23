#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 14:23:02 2024

@author: kaab
"""


import argparse
import torch
from CLIP import clip
from PIL import Image
import os
import matplotlib.pyplot as plt
import esrgan_test
import vid_to_fr
import cv2

def main(video_path, prompt):
    vid_to_fr.vid_to_frame(video_path=video_path)
    esrgan_test.highres()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model, preprocess = clip.load("ViT-B/32", device=device)
    img_folder = "results"
    image_paths = [os.path.join(img_folder, filename) for filename in os.listdir(img_folder) if filename.endswith(('.png', '.jpg', '.jpeg'))]
    images = [preprocess(Image.open(image_path)).unsqueeze(0).to(device) for image_path in image_paths]

    image_batch = torch.cat(images)

    text = clip.tokenize([prompt]).to(device)  

    with torch.no_grad():
        image_features = model.encode_image(image_batch)  
        text_features = model.encode_text(text)           
    
        logits_per_image, logits_per_text = model(image_batch, text)

        probs = logits_per_image.softmax(dim=0).cpu().numpy()

    print("Label probs for each image:", probs)

    max_prob_index = probs.argmax()

    print("maximum prob index is:", max_prob_index)

    img_path_max_prob = image_paths[max_prob_index]

    print(img_path_max_prob)
    print("")
    print("timestamp in the video: ",max_prob_index/4)
    img = cv2.imread(img_path_max_prob)

# Display the image in a window
    cv2.imshow('Image', img)

# Wait for a key press to close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # img = Image.open(img_path_max_prob)
    # plt.imshow(img)
    # plt.axis('off')
    # plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a video and a promp.')
    parser.add_argument('--video', type=str, required=True, help='path to the video file')
    parser.add_argument('--prompt', type=str, required=True, help='prompt for the frame to find')

    args = parser.parse_args()

    main(args.video, args.prompt)

    #how to run ----->     python detection.py --video path/to/video.mp4 --prompt "two robbers opening a door"




















