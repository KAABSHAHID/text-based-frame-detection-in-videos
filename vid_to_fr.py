import cv2




def vid_to_frame(video_path):

    vidcap = cv2.VideoCapture("vids/robbing.mp4")
    success, image = vidcap.read()


    fps = vidcap.get(cv2.CAP_PROP_FPS)
    print(fps)
    interval = int(fps/4)


    frame_count=1
    count = 0

    while vidcap.isOpened() and success:
        success, image = vidcap.read()
    
        if count % interval == 0:
        
            cv2.imwrite("frames/pic%d.png" % frame_count, image)
            frame_count +=1
    
        count +=1
    

    vidcap.release()
    print("total frames :", frame_count-1)
