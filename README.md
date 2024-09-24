# Text Prompt Based Frame Detection in a Video

## Overview
This project enables the processing of videos by converting them into frames, enhancing the resolution using **ESRGAN**, and using **CLIP** to identify objects or specific events described by a text prompt. It is particularly useful in applications like security surveillance, video analysis, and content filtering, where identifying specific objects or people in low-resolution videos is crucial.  
The project workflow consists of the following key steps:  
1. `Extract frames` from a video at a reduced frame rate (4 FPS).  
2. `Enhance the resolution` of these frames using a generative model **ESRGAN (Enhanced Super-Resolution GAN)**.  
3. `Detect objects or events` in the frames by matching them with a text prompt using **CLIP**.  
4. `Return the timestamp` and the best matching frame for the given prompt.  

## Applications  
### 1. Security Surveillance  
In security footage, identifying suspects, events, or objects can be challenging due to poor video quality. This project enhances the clarity of frames and helps identify specific instances described by a user.  
### 2. Video Content Analysis
Useful for analyzing video footage in various industries, including video content creation, where users may want to extract specific moments in the video.  
### 3. Forensics and Investigations
Investigators can use this project to search video footage for clues or objects by simply inputting a descriptive prompt, helping speed up manual reviews of large video files.  
### 4. Object and Event Detection in Videos
Detecting specific actions (e.g., “person running” or “car turning”) in video footage can be automated using CLIP, reducing manual work for video review.  

## Project Workflow  

1. **Video to Frames Conversion**  
   - The video is processed to extract individual frames at 4 frames per second. This frame rate balances accuracy with efficiency, ensuring enough frames to detect meaningful actions without overloading the system.
   - The frames are saved in the folder named `frames`.  

2. **Super-Resolution Enhancement using ESRGAN**  
   - Once frames are extracted, they are enhanced using ESRGAN, a `state-of-the-art` deep learning model that increases the resolution of images.  
   - This step is crucial in scenarios where video quality is low, as it enhances clarity and makes object detection more accurate.
   - The Super Resolution frames are saved in the folder named `results`.   

3. **Object Detection using CLIP**  
   - The enhanced frames are processed using CLIP (Contrastive Language-Image Pretraining), which matches the provided text prompt (e.g., "a person wearing a red shirt") with the best corresponding frame.  
   - CLIP generates feature vectors for both the images (frames) and the text, and then finds the frame that has the highest similarity to the text prompt.  

4. **Return Results**  
   - After processing, the project returns:  
     - **Timestamp**: The exact point in the video where the best matching frame occurs.  
     - **Frame Image**: The enhanced frame that most closely matches the user's input prompt.

## Technical Details    
### Tools and Technologies Used    
- **OpenCV**: Used for video processing and converting the video into individual frames.  
- **ESRGAN (Enhanced Super-Resolution Generative Adversarial Networks)**: Used to enhance the resolution of the extracted frames. The model used in this project is the `RRDB_ESRGAN_x4` model. 
- **CLIP (Contrastive Language-Image Pretraining)**: Used for matching a text prompt to a specific frame in the video. 
- **PyTorch**: Deep learning framework used to run both ESRGAN and `CLIP` models.  
- **Numpy**: Utilized for numerical operations in processing images and tensors.  
- **Matplotlib & PIL**: For image visualization (in case you want to display the results in the notebook or GUI).

 
## How to Use  
### 1. Clone the Repository  
```bash  
git clone https://github.com/KAABSHAHID/text-based-frame-detection.git  
cd your-repo-name  
````

### 2. Set Up the Environnment  
- Ensure that you have Python 3.8+ installed.  
- Install the necessary dependencies:

````bash
pip install torch opencv-python pillow numpy matplotlib natsort
````
### 3. Download ESRGAN Model  
- Download the RRDB_ESRGAN_x4.pth model from the official [ESRGAN GitHub Repository](https://github.com/xinntao/ESRGAN) and place it in the `models` folder of the project.

### 4. Run the Script  
You can process the video and detect objects using a text prompt by running the following command:  
command:  
````bash
python detection.py --video path/to/video.mp4 --prompt "your text prompt"
````
For example:  
```bash
python detection.py --video vids/robbing.mp4 --prompt "two robbers opening a door"
````
### Output  
- The project will output the frame that best matches the provided prompt and the corresponding timestamp in the video.

## Acknowledgments  

This project relies on the outstanding contributions made by the creators of [ESRGAN](https://github.com/xinntao/ESRGAN) and [CLIP](https://github.com/openai/CLIP). We would like to acknowledge their impactful research and open-source contributions.  

- **[ESRGAN (Enhanced Super-Resolution Generative Adversarial Networks)](https://github.com/xinntao/ESRGAN)**:  
   - Citation:  
     ```
     @InProceedings{wang2018esrgan,
       author = {Wang, Xintao and Yu, Ke and Wu, Shixiang and Gu, Jinjin and Liu, Yihao and Dong, Chao and Qiao, Yu and Loy, Chen Change},
       title = {ESRGAN: Enhanced super-resolution generative adversarial networks},
       booktitle = {The European Conference on Computer Vision Workshops (ECCVW)},
       month = {September},
       year = {2018}
     }
     ```  
   - ESRGAN has been instrumental in this project by providing high-quality image resolution enhancement, which is essential for improving the clarity of frames extracted from low-quality video inputs.  

- **[CLIP (Contrastive Language-Image Pretraining)](https://github.com/openai/CLIP)**:  
   - Citation:  
     ```
     @inproceedings{radford2021learning,
       title={Learning Transferable Visual Models From Natural Language Supervision},
       author={Radford, Alec and Kim, Jong Wook and Hallacy, Karthik and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and others},
       booktitle={International Conference on Machine Learning (ICML)},
       year={2021}
     }
     ```  
   - CLIP was key in enabling the text-image matching functionality that powers object detection in this project. Special thanks to OpenAI for their work on CLIP and for making it available to the community.  
