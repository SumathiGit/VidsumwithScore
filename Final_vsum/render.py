
"""Adding single text to a video """
# import moviepy.editor as mp



# my_video = mp.VideoFileClip("/home/sumathi/cap/video.mp4")

# w,h = moviesize = my_video.size

# my_text = mp.TextClip("A man and a women walking on a street", font="Amiri-regular", color="white", fontsize=34)

# txt_col = my_text.on_color(size=(my_video.w + my_text.w, my_text.h+5), color=(0,0,0), pos=(6,"center"), col_opacity=0.6)

# txt_mov = txt_col.set_pos( lambda t: (max(w/30,int(w-0.5*w*t)),max(5*h/6,int(100*t))) )

# final = mp.CompositeVideoClip([my_video,txt_mov])

# final.subclip(0,17).write_videofile("final.mp4",fps=24,codec="libx264")

""" ***********************************************************************************************************************************************"""
#Multiple text in a single video
 
# import moviepy.editor as mp


# my_video = mp.VideoFileClip("/home/sumathi/cap/video.mp4").subclip(0,10)
# print(my_video.duration)
# w,h = moviesize = my_video.size

# texts = []
# with open("/home/sumathi/cap/pytorch-tutorial/tutorials/advanced/image_captioning/caption.txt") as file:
#     for line in file: 
#         line = line.strip() #or some other preprocessing
#         texts.append(line)

# print(type(texts))
# print(len(texts))
# starts = [0,2,4,6,8,10,12,14,16,18,20] # or whatever
# durations = [2,2,2,2,2] #Time duraton between captions
# t = 0
# txt_clips = []
# for text,t,duration in zip(texts, starts, durations):
#   txt_clip = mp.TextClip(text,font="Amiri-regular", color="black", fontsize=29)
#   txt_clip = txt_clip.set_start(t)
#   txt_col = txt_clip.on_color(size=(my_video.w + txt_clip.w, txt_clip.h+5), color=(0,0,0), pos=(6,"center"), col_opacity=0.6)
#   txt_clip = txt_clip.set_pos( lambda t: (max(w/30,int(w-0.5*w*t)),max(5*h/6,int(100*t))) ).set_duration(duration)
#   txt_clips.append(txt_clip) 
#   print(txt_clips)
#   print(type(txt_clips))

   
# final_video = mp.CompositeVideoClip([my_video, txt_clips[0],txt_clips[1],txt_clips[2],txt_clips[3],txt_clips[4]])

# final_video.write_videofile("TEXT.mp4",fps=24)

"********************************************************************************************************************************************"

import moviepy.editor as mp
from moviepy.editor import *


my_video = mp.VideoFileClip("/home/sumathi/cap/video.mp4").subclip(0,10)
print(my_video.duration)
print("***********************")
my_video = my_video.fx(vfx.speedx, 0.2)


print(my_video.duration)
w,h = moviesize = my_video.size

texts = []
with open("/home/sumathi/cap/pytorch-tutorial/tutorials/advanced/image_captioning/caption.txt") as file:
    for line in file: 
        line = line.strip() #or some other preprocessing
        texts.append(line)

print(type(texts))
print(len(texts))
# starts = [0,2,4,6,8,10,12,14,16,18,20] # or whatever
# durations = [2,2,2,2,2] #Time duraton between captions
# num_range = len(texts)
# print(num_range)
starts = []
step = len(texts*5)
# for i in range(len(texts)):
#     starts.append(i)
# print(starts)
starts = list(range(0, step, 5))
print(starts)
print(len(starts))
durations = listOfInt2 = [5 for i in range(len(texts))]
print(durations)
t = 0
txt_clips = []
for text,t,duration in zip(texts, starts, durations):
  txt_clip = mp.TextClip(text,font="Amiri-regular", color="black", fontsize=29)
  txt_clip = txt_clip.set_start(t)
  txt_col = txt_clip.on_color(size=(my_video.w + txt_clip.w, txt_clip.h+5), color=(0,0,0), pos=(6,"center"), col_opacity=0.6)
  txt_clip = txt_clip.set_pos( lambda t: (max(w/30,int(w-0.5*w*t)),max(5*h/6,int(100*t))) ).set_duration(duration)
  txt_clips.append(txt_clip) 
print(txt_clips)
print(len(txt_clips))
print(type(txt_clips))


# i= list(range(0, len(txt_clips)))
# print(i)


final_video = mp.CompositeVideoClip([my_video, txt_clips[0],txt_clips[1],txt_clips[2],txt_clips[3],txt_clips[4],txt_clips[5],txt_clips[6],txt_clips[7],txt_clips[8],txt_clips[9],txt_clips[10],txt_clips[11],txt_clips[12],txt_clips[13],txt_clips[14],txt_clips[15],txt_clips[16],txt_clips[17],txt_clips[18],txt_clips[19],txt_clips[20],txt_clips[21],txt_clips[22],txt_clips[23],txt_clips[24]])

final_video.write_videofile("TEXT1.mp4")