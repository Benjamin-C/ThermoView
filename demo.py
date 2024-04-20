from thermocam import P2Pro

import cv2
import argparse
import time
import threading
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0, help="Video Device number e.g. 0, use v4l2-ctl --list-devices")
args = parser.parse_args()

dev = 0
if args.device:
    dev = '/dev/video'+str(args.device)
else:
    # The device to open. This should be correct for any P2 Pro cam, but might be different for others
    # Might not work well if you have multiple cams. Too bad.
    dev = "/dev/v4l/by-id/usb-Generic_USB_Camera_200901010001-video-index0"

#256x192 General settings
width = 256 #Sensor width
height = 192 #sensor height
scale = 3 #scale multiplier
newWidth = 0 
newHeight = 0
alpha = 1.0 # Contrast control (1.0-3.0)
colormap = 0
font=cv2.FONT_HERSHEY_SIMPLEX
dispFullscreen = False
cv2.namedWindow('Thermal',cv2.WINDOW_GUI_NORMAL)
rad = 0 #blur radius
threshold = 2
hud = True
recording = False
elapsed = "00:00:00"
snaptime = "None"

def setDims(w, h):
    global width, height, newWidth, newHeight
    width = w
    height = h
    newWidth = width*scale 
    newHeight = height*scale
    cv2.resizeWindow('Thermal', newWidth,newHeight)

setDims(width, height)

def rec():
    now = time.strftime("%Y%m%d--%H%M%S")
    #do NOT use mp4 here, it is flakey!
    videoOut = cv2.VideoWriter(now+'output.avi', cv2.VideoWriter_fourcc(*'XVID'),25, (newWidth,newHeight))
    return(videoOut)

def snapshot(heatmap):
    #I would put colons in here, but it Win throws a fit if you try and open them!
    now = time.strftime("%Y%m%d-%H%M%S") 
    snaptime = time.strftime("%H:%M:%S")
    cv2.imwrite("TC001"+now+".png", heatmap)
    return snaptime

nextFrame = threading.Event()

with P2Pro(device=dev) as cam:

    cv2.imshow('Thermal',cam.imdata)

    def onFrame():
        nextFrame.set()

    cam.frameListeners.append(onFrame)

    while True:
        nextFrame.wait()
        
        # Get center temperature
        midTemp = round(cam.thdata[96,128],2)

        #find the max temperature in the frame
        lomax = cam.thdata.max()
        posmax = cam.thdata.argmax()
        #since argmax returns a linear index, convert back to row and col
        mcol,mrow = divmod(posmax,width)
        maxtemp = round(cam.thdata[mcol,mrow],2)
        
        #find the lowest temperature in the frame
        lomin = cam.thdata.min()
        posmin = cam.thdata.argmin()
        #since argmax returns a linear index, convert back to row and col
        lcol,lrow = divmod(posmin,width)
        mintemp = round(cam.thdata[lcol,lrow],2)

        #find the average temperature in the frame
        avgtemp = cam.thdata.mean()

        #Contrast
        bgr = cv2.convertScaleAbs(cam.imdata, alpha=alpha) #Contrast
        #bicubic interpolate, upscale and blur
        bgr = cv2.resize(bgr,(newWidth,newHeight),interpolation=cv2.INTER_CUBIC)#Scale up!
        if rad>0:
            bgr = cv2.blur(bgr,(rad,rad))

        #apply colormap
        if colormap == 0:
            heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_JET)
            cmapText = 'Jet'
        if colormap == 1:
            heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_HOT)
            cmapText = 'Hot'
        if colormap == 2:
            heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_MAGMA)
            cmapText = 'Magma'
        if colormap == 3:
            heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_INFERNO)
            cmapText = 'Inferno'
        if colormap == 4:
            heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_PLASMA)
            cmapText = 'Plasma'
        if colormap == 5:
            heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_BONE)
            cmapText = 'Bone'
        if colormap == 6:
            heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_SPRING)
            cmapText = 'Spring'
        if colormap == 7:
            heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_AUTUMN)
            cmapText = 'Autumn'
        if colormap == 8:
            heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_VIRIDIS)
            cmapText = 'Viridis'
        if colormap == 9:
            heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_PARULA)
            cmapText = 'Parula'
        if colormap == 10:
            heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_RAINBOW)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            cmapText = 'Inv Rainbow'

        # draw crosshairs
        cv2.line(heatmap,(int(newWidth/2),int(newHeight/2)+20),\
        (int(newWidth/2),int(newHeight/2)-20),(255,255,255),2) #vline
        cv2.line(heatmap,(int(newWidth/2)+20,int(newHeight/2)),\
        (int(newWidth/2)-20,int(newHeight/2)),(255,255,255),2) #hline

        cv2.line(heatmap,(int(newWidth/2),int(newHeight/2)+20),\
        (int(newWidth/2),int(newHeight/2)-20),(0,0,0),1) #vline
        cv2.line(heatmap,(int(newWidth/2)+20,int(newHeight/2)),\
        (int(newWidth/2)-20,int(newHeight/2)),(0,0,0),1) #hline
        #show temp
        cv2.putText(heatmap,str(midTemp)+' C', (int(newWidth/2)+10, int(newHeight/2)-10),\
        cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(heatmap,str(midTemp)+' C', (int(newWidth/2)+10, int(newHeight/2)-10),\
        cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0, 255, 255), 1, cv2.LINE_AA)

        if hud==True:
            # display black box for our data
            # cv2.rectangle(heatmap, (0, 0),(180, 140), (0,0,0), -1)
            # put text in the box
            cv2.putText(heatmap,'Avg Temp: '+str(avgtemp)+' C', (10, 14),\
            cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 255, 255), 1, cv2.LINE_AA)

            cv2.putText(heatmap,'[SX] Label Threshold: '+str(threshold)+' C', (10, 28),\
            cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 255, 255), 1, cv2.LINE_AA)

            cv2.putText(heatmap,'[M ] Colormap: '+cmapText, (10, 42),\
            cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 255, 255), 1, cv2.LINE_AA)

            cv2.putText(heatmap,'[AZ] Blur: '+str(rad)+' ', (10, 56),\
            cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 255, 255), 1, cv2.LINE_AA)

            cv2.putText(heatmap,'[DC] Scaling: '+str(scale)+' ', (10, 70),\
            cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 255, 255), 1, cv2.LINE_AA)

            cv2.putText(heatmap,'[FV] Contrast: '+str(alpha)+' ', (10, 84),\
            cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 255, 255), 1, cv2.LINE_AA)


            cv2.putText(heatmap,'[P ] Snapshot: '+snaptime+' ', (10, 98),\
            cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 255, 255), 1, cv2.LINE_AA)

            if recording == False:
                cv2.putText(heatmap,'[Y ] Recording: '+elapsed, (10, 112),\
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,(200, 200, 200), 1, cv2.LINE_AA)
            if recording == True:
                cv2.putText(heatmap,'[T ] Recording: '+elapsed, (10, 112),\
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,(40, 40, 255), 1, cv2.LINE_AA)
                
            cv2.putText(heatmap,'[Q ] Quit', (10, 126),\
            cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 255, 255), 1, cv2.LINE_AA)
        
        #Yeah, this looks like we can probably do this next bit more efficiently!
        #display floating max temp
        if maxtemp > avgtemp+threshold:
            cv2.circle(heatmap, (mrow*scale, mcol*scale), 5, (0,0,0), 2)
            cv2.circle(heatmap, (mrow*scale, mcol*scale), 5, (0,0,255), -1)
            cv2.putText(heatmap,str(maxtemp)+' C', ((mrow*scale)+10, (mcol*scale)+5),\
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0,0,0), 2, cv2.LINE_AA)
            cv2.putText(heatmap,str(maxtemp)+' C', ((mrow*scale)+10, (mcol*scale)+5),\
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0, 255, 255), 1, cv2.LINE_AA)

        #display floating min temp
        if mintemp < avgtemp-threshold:
            cv2.circle(heatmap, (lrow*scale, lcol*scale), 5, (0,0,0), 2)
            cv2.circle(heatmap, (lrow*scale, lcol*scale), 5, (255,0,0), -1)
            cv2.putText(heatmap,str(mintemp)+' C', ((lrow*scale)+10, (lcol*scale)+5),\
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0,0,0), 2, cv2.LINE_AA)
            cv2.putText(heatmap,str(mintemp)+' C', ((lrow*scale)+10, (lcol*scale)+5),\
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0, 255, 255), 1, cv2.LINE_AA)

        # Check if the window has been closed, and exit if so
        if cv2.getWindowProperty('Thermal', cv2.WND_PROP_VISIBLE) > 0:
            # display image
            cv2.imshow('Thermal',heatmap)
        else:
            break

        if recording == True:
            elapsed = (time.time() - start)
            elapsed = time.strftime("%H:%M:%S", time.gmtime(elapsed)) 
            #print(elapsed)
            videoOut.write(heaheight*scaletmap)
        
        keyPress = cv2.waitKey(1)
        if keyPress == ord('a'): #Increase blur radius
            rad += 1
        if keyPress == ord('z'): #Decrease blur radius
            rad -= 1
            if rad <= 0:
                rad = 0

        if keyPress == ord('s'): #Increase threshold
            threshold += 1
        if keyPress == ord('x'): #Decrease threashold
            threshold -= 1
            if threshold <= 0:
                threshold = 0

        if keyPress == ord('d'): #Increase scale
            scale += 1
            if scale >=5:
                scale = 5
            newWidth = width*scale
            newHeight = height*scale
        if keyPress == ord('c'): #Decrease scale
            scale -= 1
            if scale <= 1:
                scale = 1
            newWidth = width*scale
            newHeight = height*scale

        if keyPress == ord('q'): #enable fullscreen
            dispFullscreen = True
            cv2.namedWindow('Thermal',cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('Thermal',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        if keyPress == ord('w'): #disable fullscreen
            dispFullscreen = False
            cv2.namedWindow('Thermal',cv2.WINDOW_GUI_NORMAL)
            cv2.setWindowProperty('Thermal',cv2.WND_PROP_AUTOSIZE,cv2.WINDOW_GUI_NORMAL)
            cv2.resizeWindow('Thermal', newWidth,newHeight)

        if keyPress == ord('f'): #contrast+
            alpha += 0.1
            alpha = round(alpha,1)#fix round error
            if alpha >= 3.0:
                alpha=3.0
        if keyPress == ord('v'): #contrast-
            alpha -= 0.1
            alpha = round(alpha,1)#fix round error
            if alpha<=0:
                alpha = 0.0


        if keyPress == ord('h'):
            if hud==True:
                hud=False
            elif hud==False:
                hud=True

        if keyPress == ord('m'): #m to cycle through color maps
            colormap += 1
            if colormap == 11:
                colormap = 0

        if keyPress == ord('y') and recording == False: #r to start reording
            videoOut = rec()
            recording = True
            start = time.time()
        if keyPress == ord('r'): #r to start reording
            cam.rotation += 1
            setDims(height, width)
        if keyPress == ord('t'): #f to finish reording
            recording = False
            elapsed = "00:00:00"

        if keyPress == ord('p'): #f to finish reording
            snaptime = snapshot(heatmap)

        if keyPress == ord('q'):
            break

        nextFrame.clear()