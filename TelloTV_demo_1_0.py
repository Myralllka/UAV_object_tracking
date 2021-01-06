from djitellopy import Tello
import cv2
import pygame
import numpy as np
import time

# Speed of the drone
S = 60
# Frames per second of the pygame window display
# A low number also results in input lag, as input information is processed once per frame.
FPS = 120

def dead_band(x, y, w, h, old_x, old_y, old_w, old_h, epsilon):
    if (abs(old_x - x) < epsilon):
        x = old_x;
    if (abs(old_y - y) < epsilon):
        y = old_y;
    if (abs(old_w - w) < epsilon):
        w = old_w;
    if (abs(old_h - h) < epsilon):
        h = old_h;
    
    return x, y, w, h

def find_directions(x, y, center_x, center_y):
    res_x, res_y = "", ""
    if (abs(x - center_x) < 130):
        res_x = "NA"
    elif ( x < center_x):
        res_x = "left"
    elif (x > center_x):
        res_x = "right"
    else:
        res_x = "NA"

    if (abs(y - center_y) < 130):
        res_y = "NA" 
    elif ( y < center_y):
        res_y = "up"
    elif (y > center_y):
        res_y = "down"
    else:
        res_y = "NA"
    return res_y, res_x
    

class FrontEnd(object):
    """ Maintains the Tello display and moves it through the keyboard keys.
        Press escape key to quit.
        The controls are:
            - T: Takeoff
            - L: Land
            - Arrow keys: Forward, backward, left and right.
            - A and D: Counter clockwise and clockwise rotations (yaw)
            - W and S: Up and down.
    """

    def __init__(self):
        # Init pygame
        pygame.init()

        # Creat pygame window
        pygame.display.set_caption("Tello video stream")
        self.screen = pygame.display.set_mode([960, 720])

        # Init Tello object that interacts with the Tello drone
        self.tello = Tello()

        # Drone velocities between -100~100
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 10

        self.send_rc_control = False

        # create update timer
        pygame.time.set_timer(pygame.USEREVENT + 1, 1000 // FPS)

    def run(self):
        ###
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('trainer/trainer.yml')
        cascadePath = "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(cascadePath);
        font = cv2.FONT_HERSHEY_SIMPLEX
        id = 0
        names = ["None", "front", "top_view", "right", "left"] 
        minW = 0.05
        minH = 0.05
      
        ###
        self.tello.connect()
        self.tello.set_speed(self.speed)

        # In case streaming is on. This happens when we quit this program without the escape key.
        self.tello.streamoff()
        self.tello.streamon()

        frame_read = self.tello.get_frame_read()
        
        ###
        old_x, old_y, old_w, old_h = 0, 0, 0, 0
        ###
        
        should_stop = False
        while not should_stop:
            for event in pygame.event.get():
                if event.type == pygame.USEREVENT + 1:
                    self.update()
                elif event.type == pygame.QUIT:
                    should_stop = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        should_stop = True
                    else:
                        self.keydown(event.key)
                elif event.type == pygame.KEYUP:
                    self.keyup(event.key)

            if frame_read.stopped:
                break

            self.screen.fill([0, 0, 0])
            
            frame = frame_read.frame
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            text = "Battery: {}%".format(self.tello.get_battery())
            
            cv2.putText(frame, text, (5, 720 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            ###
    
            faces = faceCascade.detectMultiScale(gray,scaleFactor = 1.2,minNeighbors = 5,minSize = (int(minW), int(minH)),)

            for(x,y,w,h) in faces:
                x, y, w, h = dead_band(x, y, w, h, old_x, old_y, old_w, old_h, 8);

                id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
                if not (confidence < 70):
                    continue
                cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
                # If confidence is less them 100 ==> "0" : perfect match 
                if (confidence < 100):
                    id = names[id]
                    confidence = "  {0}%".format(round(100 - confidence))
                else:
                    id = "unknown"
                    confidence = "  {0}%".format(round(100 - confidence))
                cx, cy = x+w//2, y+h//2

                cv2.line(frame, (cx, cy), (frame.shape[1]//2, frame.shape[0]//2), (0, 255, 0) , 5) 
                cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
                cv2.putText(frame, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
                cv2.putText(frame, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  

                res_x, res_y = find_directions(cx, cy, frame.shape[1]//2, frame.shape[0]//2)
                cv2.putText(frame, "move {} and {} ".format(res_x, res_y), (30,460), font, 1, (0,0,0), 1)  
                cv2.putText(frame, "w={}, h={}".format(str(h), str(w)), (30,30), font, 1, (0,0,0), 1)  
                old_x, old_y, old_w, old_h = x, y, w, h
                
                self.update_geometry((res_x, res_y))
                break
            ###
            frame = np.rot90(frame)
            frame = np.flipud(frame)
            frame = pygame.surfarray.make_surface(frame)
            self.screen.blit(frame, (0, 0))
            pygame.display.update()
            
            time.sleep(1 / FPS)

        # Call it always before finishing. To deallocate resources.
        self.tello.end()

    def keydown(self, key):
        """ Update velocities based on key pressed
        Arguments:
            key: pygame key
        """
        if key == pygame.K_UP:  # set forward velocity
            self.for_back_velocity = S
        elif key == pygame.K_DOWN:  # set backward velocity
            self.for_back_velocity = -S
        elif key == pygame.K_LEFT:  # set left velocity
            self.left_right_velocity = -S
        elif key == pygame.K_RIGHT:  # set right velocity
            self.left_right_velocity = S
        elif key == pygame.K_w:  # set up velocity
            self.up_down_velocity = S
        elif key == pygame.K_s:  # set down velocity
            self.up_down_velocity = -S
        elif key == pygame.K_a:  # set yaw counter clockwise velocity
            self.yaw_velocity = -S
        elif key == pygame.K_d:  # set yaw clockwise velocity
            self.yaw_velocity = S

    def keyup(self, key):
        """ Update velocities based on key released
        Arguments:
            key: pygame key
        """
        if key == pygame.K_UP or key == pygame.K_DOWN:  # set zero forward/backward velocity
            self.for_back_velocity = 0
        elif key == pygame.K_LEFT or key == pygame.K_RIGHT:  # set zero left/right velocity
            self.left_right_velocity = 0
        elif key == pygame.K_w or key == pygame.K_s:  # set zero up/down velocity
            self.up_down_velocity = 0
        elif key == pygame.K_a or key == pygame.K_d:  # set zero yaw velocity
            self.yaw_velocity = 0
        elif key == pygame.K_t:  # takeoff
            self.tello.takeoff()
            self.send_rc_control = True
        elif key == pygame.K_l:  # land
            not self.tello.land()
            self.send_rc_control = False

    def update_geometry(self, parameters):
        """ Update velocities based on position
        Arguments:
            parameters: parameters to update (one per time)
        """
        if (parameters[0] == "up"):
            self.up_down_velocity = 15
        elif (parameters[0] == "down"):
            self.up_down_velocity = -15
        else:
            self.up_down_velocity = 0
        if (parameters[1] == "right"):
            self.left_right_velocity = 15
        elif (parameters[1] == "left"):
            self.left_right_velocity = -15
        else:
            self.left_right_velocity = 0
        

    def update(self):
        """ Update routine. Send velocities to Tello."""
        if self.send_rc_control:
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity,
                self.up_down_velocity, self.yaw_velocity)


def main():
    frontend = FrontEnd()

    # run frontend
    frontend.run()


if __name__ == '__main__':
    main()