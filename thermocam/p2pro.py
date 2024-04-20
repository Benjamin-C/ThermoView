import cv2
import numpy as np
import threading

class P2Pro():
    def __init__(self, device: str = None, rotation: int = 0, onFrame: callable = None):
        self.__device: str = device
        self.rotation: int = 0
        self.imdata: np.ndarray = None
        self.thdata: np.ndarray = None
        self.__readThread: threading.Thread = None
        self.__readyEvent: threading.Event = threading.Event()
        self.__running: bool = False
        self.frameListeners: list = []
        if onFrame is not None:
            self.frameListeners.append(onFrame)
        self.framenum = 0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exceptionType, exceptionValue, exceptionTraceback):
        # super().__exit__(exceptionType, exceptionValue, exceptionTraceback)
        pass

    def __readThreadWorker(self):
        self.imdata, self.thdata = self.__captureFrame()
        self.__readyEvent.set()
        while self.__running:
            self.imdata, self.thdata = self.__captureFrame()
            self.framenum += 1
            for listener in self.frameListeners:
                listener()
 
    def start(self):
        self._cap = cv2.VideoCapture(self.__device or "/dev/v4l/by-id/usb-Generic_USB_Camera_200901010001-video-index0", cv2.CAP_V4L)
        self._cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)
        self.__running = True
        self.framenum = 0
        self.__readyEvent.clear()
        self.__readThread = threading.Thread(target=self.__readThreadWorker)
        self.__readThread.start()
        self.__readyEvent.wait()

    def stop(self):
        self.__running = False
        self.__readThread.join()
        self._cap.release()
        self._cap = None

    def __captureFrame(self):
        if self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret == True:
                imdata,thdata = np.array_split(frame, 2)

                imdata = np.rot90(imdata, self.rotation)
                imdata = cv2.cvtColor(imdata,  cv2.COLOR_YUV2BGR_YUYV)

                thdata = np.rot90(thdata, self.rotation)
                # now parse the data from the bottom frame and convert to temp!
                # https://www.eevblog.com/forum/thermal-imaging/infiray-and-their-p2-pro-discussion/200/
                # Huge props to LeoDJ for figuring out how the data is stored and how to compute temp from it.
                rawtemp = thdata[:,:,0] + (thdata[:,:,1] * 256)
                temp = (rawtemp/64)-273.15
                
                return imdata, temp
            else:
                raise RuntimeError("There was an error getting a frame from the camera")
        else:
            raise RuntimeError("The camera is not open, so you can't grab a frame")
