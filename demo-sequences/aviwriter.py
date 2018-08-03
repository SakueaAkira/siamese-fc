import os,cv2
fps=24
fourcc=cv2.VideoWriter_fourcc('M','J','P','G')
video_writer=cv2.VideoWriter('result.avi',fourcc,fps,(460,345))
for i in range(1,367):
    img=cv2.imread('%.8d' % i + '.jpg')
    video_writer.write(img)
video_writer.release()
os.listdir()

os.system('pause')