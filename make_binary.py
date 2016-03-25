import cv2
import glob
import os
import numpy as np
import csv
import pandas as pd

from random import randint
from random import uniform

def transform_test():
  img_bytes = 3072 #color 32x32
  filedirs = ['/home/dario/Pictures/DATASET/test/nao/','/home/dario/Pictures/DATASET/test/not_nao/']

  #create a list of files 
  filenames = []

  imageCount = 0
  for f in filedirs:	
    #change the file extensions according to your needs
    png_files_path = glob.glob(os.path.join(f, '*.[pP][nN][gG]'))
    for filename in png_files_path:
      imageCount = imageCount + 1
      #retrieving the label from the csv
      currentFile = os.path.basename(filename)
      i = 0
      for s in df.Filename:
        if str(s) == str(currentFile):
          break
        i = i+1

      classId = classIdCSV[i]
      #storing label file_name
      if classId<10:
        s = str(0)+str(classId)+filename
      else:
        s = str(classId)+filename
      filenames.append(s)
      
  print(str(imageCount) + " images found")
  
  #we set the size of all the images to 24x24
  tempBinaryFile = np.zeros((len(filenames),img_bytes+1), dtype=np.uint8)

  #fill the array
  count = 0

  for j in range(0,len(filenames)):
    #load the image with OpenCV
    #print(filenames[j])
    img = cv2.imread(filenames[j][2:])
 	
 	#squaring the img
    height, width, channels = img.shape
    
    #do not shrink
    if height < 32 or width < 32 :
      continue
 	
    if width > height:
      minSize = height
      diffSize = width - height
    else:
      minSize = width
      diffSize = height - width
    
    if diffSize > minSize/2 :
      continue
      
    img = img[diffSize/2:minSize-diffSize/2, diffSize/2:minSize-diffSize/2]
	
    img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    ch1, ch2, ch3 = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    ch1 = clahe.apply(ch1)
    img = cv2.merge((ch1,ch2,ch3))
    img = cv2.cvtColor(img, cv2.COLOR_Lab2BGR)
    
    #resize the img
    img = cv2.resize(img, (32,32))
		
    b = []
    g = []
    r = []
    
    #cycling over pixels
    for x in range(32):
      for y in range(32):
	    #accessing the intensity of each pixel in the image
        b.append(img[x,y,0])
        g.append(img[x,y,1])
        r.append(img[x,y,2])
     
    tempBinaryFile[count,1:img_bytes+1] = np.concatenate((r,g,b))
    tempBinaryFile[count,0] = int(filenames[j][0:1])
    #count will take care of the file actually inserted in the file
    count = count + 1
  
  print(str(count) + " images created")
  BinaryFile = np.zeros((count,img_bytes+1), dtype=np.uint8)
  
  for j in range(0,count-1):
    BinaryFile[j,:] = tempBinaryFile[j,:]
    
  newFile = open ("/home/dario/Desktop/nao_test.bin", "wb")
  newFileByteArray = bytearray(BinaryFile)
  newFile.write(newFileByteArray)

def transform_train():
  img_bytes = 3072 #color 32x32
  IMAGE_SIZE = [32,32]
  jittered = True
  color = True
  
  #reading the dirs file where all the directories of the images are stored
  filedirs = ['/home/dario/Pictures/data/pos/', '/home/dario/Pictures/data/neg/']

  #create a list of files 
  filenames = []
  classId = 0
  imageCount = 0
  
  for f in filedirs:	
    #change the file extensions according to your needs
    png_files_path = glob.glob(os.path.join(f, '*.[pP][nN][gG]'))
    print len(png_files_path)
    for filename in png_files_path:
      imageCount = imageCount + 1
      #storing label file_name
      if classId<10:
        s = str(0)+str(classId)+filename
      else:
        s = str(classId)+filename
      filenames.append(s)
    classId = classId+1
  print(str(imageCount) + " images found")
  
  #if jittered we have imageCount*4 number of images
  #else the original number 
  #3 is the number of channels when color is True
  if jittered:
    if color:
      tempBinaryFile = np.zeros((imageCount*4*3, img_bytes+1), dtype=np.uint8)
    else:
      tempBinaryFile = np.zeros((imageCount*4, img_bytes+1), dtype=np.uint8)
  else:
    if color:
      tempBinaryFile = np.zeros((imageCount*3, img_bytes+1), dtype=np.uint8)
    else:
      tempBinaryFile = np.zeros((imageCount, img_bytes+1), dtype=np.uint8)
    
  #fill the array
  count = 0

  for j in range(0,len(filenames)):
    #load the image with OpenCV
    #print(filenames[j])
    img = cv2.imread(filenames[j][2:])

 	#img shape
    height, width, channels = img.shape

     #check min size
#    if height < 28 or width< 28 :
#      continue
	
    #normalizing image hist
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    #ch1, ch2, ch3 = cv2.split(img)
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #ch1 = clahe.apply(ch1)
    #img = cv2.merge((ch1,ch2,ch3))
    #img = cv2.cvtColor(img, cv2.COLOR_Lab2BGR)
    if color == False:
      img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      img = cv2.equalizeHist(img)
    imgs = [img]
#    cv2.imshow('images',img)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    if jittered:
      for numOfJitt in range(3):
        #setting up parameters for transformation
        rotation = randint(-15,15)
        centerx = width/2+randint(-2,2)
        centery = height/2+randint(-2,2)
        rotMat = cv2.getRotationMatrix2D((centerx,centery),rotation,1)
        rot = cv2.warpAffine(img,rotMat,(width,height))

        height, width, channels = img.shape
        scale = uniform(1,1.4)
        rotMat = cv2.getRotationMatrix2D((height/2,width/2),0,scale)
        rot = cv2.warpAffine(img,rotMat,(width,height))

        imgs.append(rot)
        
    
    for img in imgs:
      #crop centrally to remove black border caused by rotation
      img = img[4:height-4, 4:width-4]
      if(height<width):
        continue
      img = cv2.resize(img, (IMAGE_SIZE[0], IMAGE_SIZE[1]))
	
      if color:
        b = []
        g = []
        r = []
      else:
        gray = []

      #cycling over pixels
      for x in range(IMAGE_SIZE[1]):
        for y in range(IMAGE_SIZE[0]):
	      if color:
	        b.append(img[x,y,0])
	        g.append(img[x,y,1])
	        r.append(img[x,y,2])
	      else:
	        gray.append(img[x,y])
      
      if color:
        tempBinaryFile[count,1:img_bytes+1] = np.concatenate((r,g,b))
      else:
        tempBinaryFile[count,1:img_bytes+1] = gray
      tempBinaryFile[count,0] = int(filenames[j][0:1])
      #count will take care of the file actually inserted in the file
      count = count + 1
  
  print(str(count) + " images accepted")
  if color:
    BinaryFile = np.zeros((count,img_bytes+1), dtype=np.uint8)
  else:
    BinaryFile = np.zeros((count,img_bytes+1), dtype=np.uint8)
  
  for j in range(0,count-1):
    BinaryFile[j,:] = tempBinaryFile[j,:]
    
  newFile = open ("/home/dario/Desktop/robocup_train.bin", "wb")
  newFileByteArray = bytearray(BinaryFile)
  newFile.write(newFileByteArray)
