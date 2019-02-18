
# =============================================================================
# LIBRARIES AND DEPENDENCIES - LIBRARIES AND DEPENDENCIES - LIBRARIES AND DEPEN
# =============================================================================
from __future__ import print_function
from imutils import face_utils, rotate_bound
from datetime import datetime, date, time, timedelta
import os.path, time
import numpy as np
import calendar
import math
import dlib
import sys
import cv2 
import os

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# =============================================================================
# GLOBAL VARIABLES - GLOBAL VARIABLES - GLOBAL VARIABLES - GLOBAL VARIABLES - G
# =============================================================================
cord_target = [-1, -1]

# =============================================================================
# CLASSES - CLASSES - CLASSES - CLASSES - CLASSES - CLASSES  - CLASSES - CLASSE
# =============================================================================

# Dataset class for operations
class Dataset():

    def __init__(self, path, video_size, idx = 0):
        
        """ Loads all dataset in a given path and sets all information to create
            an every day video. Program will search picture in all folders and
            sub folders in the given path
        Args:
            path: `string` absolute path where pictures are located
            video_size: `tuple` desired video size
            idx: `int` picture index

        Returns:
            None: `None`  No returns
        """

        # Process variables and descriptors
        self.NumberofSamples = 0 # Number of samples
        self.pictures = [] # List of pictures
        self.idx = idx # Index of pictures

        # Video process variables
        self.video_size = video_size # Desired video size
        self.current_img_src = None  # Current picture cv2.mat
        self.fourcc = 0x00000020 # Desired Codec H264
        self.fps = 15. # Desired Video frame rate
        
        # Face shape descriptors
        self.face_width  = 0.15 # Face descriptor width
        self.face_heihgt = 0.25 # Face descriptor heihgt
        # Face descriptor rectangle
        self.face_rect   = ((0.5 - self.face_width, 0.5 - self.face_heihgt), 
                            (0.5 + self.face_width, 0.5 + self.face_heihgt))

        # Face descriptor models
        self.face_detector = FaceDetector()
        self.face_marks = FacialLandMarks()

        # Load over mask
        self.over_mask = cv2.imread("./over_mask_1.png" ,cv2.IMREAD_UNCHANGED)
        self.over_mask = cv2.resize(self.over_mask, self.video_size)

        # Check if specified folder exits
        if not os.path.exists(path):
            print("Specified folder does not exit")

        else:
            subfolders = []
            # Read all sub folders with png and jpg files in 'path'
            print("\nFolders in path:")
            for root, dirs, files in os.walk(path):
                subfolders.append(root)
                print("\t",root)
            if len(subfolders) < 2:
                print("\t\tthere's no sub folders in path")

            print("\nReading dataset, please wait ... \n")
            for fold_idx, folder in enumerate(subfolders):
                dataset = [os.path.join(path, folder, scr) for scr in os.listdir(os.path.join(path, folder)) if scr.endswith(".png") or scr.endswith(".PNG")  or scr.endswith(".jpg") or scr.endswith(".JPG") ]  
                date_file = "None"; 
                if os.path.isfile(os.path.join(folder, "date_format.txt")): date_file = "Yes";
                print("\n\t{} - folder: {} \n\t\tDate File: {}\n\t\tPictures:{}".format(fold_idx, folder, date_file, len(dataset)))

                # List pictures names in folder
                # dataset_scr = [scr for scr in os.listdir(os.path.join(path, folder)) if scr.endswith(".png") or scr.endswith(".PNG") or scr.endswith(".jpg") or scr.endswith(".JPG")]  
                # self.list_dataset(dataset_scr)
    
                for data in dataset:
                    self.pictures.append(Picture(str(data)))
                    self.NumberofSamples += 1

            # Print gallery information
            print("\nGallery information:\n")
            print("\tEstimated Duration ..... {} [seg]".format(round(self.NumberofSamples/self.fps, 2)))
            print("\tTotal of pictures ...... {}".format(self.NumberofSamples))
            print("\tTotal of Years ......... {}".format(round(self.NumberofSamples/365., 2)))
            
            # If there's not any image to process
            if not self.NumberofSamples:
                self.idx = -1
                print("Note: No images to process, process finished\n")
                return

            # Re-assign index if the specified one is superior to Number of samples
            if self.idx > self.NumberofSamples:
                print("Warning: Current index {} is superior to number of samples".format(self.idx))
                print("\tNote: Index will be reassigned as zero\n")
                self.idx = 0
            elif self.idx == -1:
                self.idx = self.NumberofSamples - 1 

            # Sort picture samples
            self.sorted_idx_list, self.timestamp_list =  self.list_time_stamps(False)

            # Load picture in specified index
            self.get_idx_sample(self.sorted_idx_list[self.idx])

    def list_time_stamps(self, report_secuence):
        
        """ get a list with indices sorted of files list by date 
        Args:
            report_secuence: `boolean` enable/disable print complete report of files
        Returns:
            None: `None`  No returns
        """

        # Variables 
        timestamp_list = []     # List of timestamps
        str_timestamp_list = [] # List of str files times
        no_included = 0         # FIles to no include in video

        # Get date str and timestamps list of picture files
        for picture in self.pictures:
            timestamp_list.append(int(picture.date))
            str_timestamp_list.append(datetime.fromtimestamp(int(picture.date)).strftime('%Y_%m_%d'))
            if not picture.include:
                no_included += 1

        # Get start and finish time bewteen all pictures files
        start_time = min(timestamp_list)
        finish_time = max(timestamp_list)

        # Print report
        print("\tStart time ............. {}".format(time.ctime(start_time)))
        print("\tCurrent time ........... {}".format(time.ctime(finish_time)))

        # Counters
        rep_dates = no_dates = 0

        if report_secuence: print()

        # Variables
        date_current_time = datetime.fromtimestamp(start_time)
        str_date_current_time = date_current_time.strftime('%Y_%m_%d')
        str_date_stop_time = datetime.fromtimestamp(finish_time).strftime('%Y_%m_%d')

        # Sort and get indices list
        idx = 0
        while str_date_current_time != str_date_stop_time:

            if not str_date_current_time in str_timestamp_list:
                no_dates += 1
                if report_secuence:
                    pass
                    print(bcolors.FAIL + "\t{}".format(str_date_current_time) + bcolors.ENDC)
            else:
                if report_secuence:
                    idx += 1
                    print("\t{} - {}".format(str_date_current_time, idx))
                    
            count = str_timestamp_list.count(str_date_current_time)
            if count > 1:
                idx += 1
                for i in range(1, count):
                    rep_dates += 1
                    if report_secuence:
                        print(bcolors.WARNING + "\t{} - {}".format(str_date_current_time, idx) + bcolors.ENDC)

            date_current_time = date_current_time + timedelta(days=1)
            str_date_current_time = date_current_time.strftime('%Y_%m_%d')
            
        # Print report
        if report_secuence: print()
        print("\n\tRepeated dates  ........ {}".format(rep_dates))
        print("\tNo picture dates ....... {}".format(no_dates))
        print("\tCompensation ........... {}".format(rep_dates-no_dates))
        print("\n\tPictures No included ... {}".format(no_included))

        # Get index list of sorted timestamp list
        sorted_idx_list = np.argsort(timestamp_list)

        # Sort list of timestamps
        timestamp_list = sorted(timestamp_list)

        # Return results
        return sorted_idx_list, timestamp_list

    def include_current_picture(self):
        
        """ Cange the state of include picture file in final video
        Args:
            None: `None`  No returns
        Returns:
            None: `None`  No returns
        """

        self.pictures[self.sorted_idx_list[self.idx]].include = not self.pictures[self.sorted_idx_list[self.idx]].include

    def get_idx_sample(self, idx):
        
        """ Load the picture and its information for a given index 
        Args:
            idx: `int` index to read sample
        Returns:
            None: `None`  No returns
        """

        # Get picture cv2.mat in list index   
        img = self.pictures[idx].get_image()

        # Get image shape
        h, w, _ = img.shape

        # Adjust image if height is superior to width
        if h >= w:
            img = image_resize(img, width = None, height = self.video_size[1], inter = cv2.INTER_AREA)
            img_mask = np.zeros((self.video_size[1], self.video_size[0], 3), np.uint8)

            x_offset = int(self.video_size[0]*0.5 - img.shape[0]*0.5); y_offset = 0
            img_mask[y_offset:y_offset+img.shape[0], x_offset:x_offset+img.shape[1]] = img
            img = img_mask
        
        else:
            if float(h)/float(w) != float(self.video_size[1])/float(self.video_size[0]):
                img_mask = np.zeros((self.video_size[1], self.video_size[0], 3), np.uint8)
                img = image_resize(img, width = self.video_size[0], height = None, inter = cv2.INTER_AREA)

                if img.shape[0] > self.video_size[1]:
                    img = cv2.resize(img, self.video_size)

                y_offset = int(self.video_size[1]*0.5 - img.shape[0]*0.5); x_offset = 0
                img_mask[y_offset:y_offset+img.shape[0], x_offset:x_offset+img.shape[1]] = img
                img = img_mask

        # Resize picture to desired video size
        img = cv2.resize(img, self.video_size)

        # Assign picture to current cv2.mat variable
        self.current_img_src = img

        # Load features of current picture
        self.load_features_current_sample()

    def draw_desired_geometric(self, img_src):
        
        """ Draws the desired face shape and information in an image
        Args:
            img_src: `cv2.math` picture to draw desire face shape and information
        Returns:
            None: `None`  No returns
        """

        # ---------------------------------------------------------------------
        # Video dimensions
        width = self.video_size[0]
        height = self.video_size[1]

        # Center of image variable
        center_img = (int(width/2), 
                      int(height/2))

        # ---------------------------------------------------------------------
        # Draw desired face geometric
        cv2.rectangle(  img_src, 
                        (int(self.face_rect[0][0]*self.video_size[0]), 
                        int(self.face_rect[0][1]*self.video_size[1])), 
                        (int(self.face_rect[1][0]*self.video_size[0]),
                        int(self.face_rect[1][1]*self.video_size[1])), 
                        (0, 255, 0), 
                        2)
        
        # Draw face shape
        cv2.ellipse(img = img_src, 
                    center = center_img, 
                    axes = (int(width*0.15), 
                            int(height*0.3)), 
                    angle = 0, 
                    startAngle = 0, 
                    endAngle = 360, 
                    color = (0, 255, 0), 
                    thickness = 2)
        cv2.line(img = img_src, 
                 pt1 = (int(center_img[0]), 
                        int(center_img[1] - height*0.3)),
                 pt2 = (int(center_img[0]), 
                        int(center_img[1] + height*0.3)),
                 color = (0, 255, 0), 
                 thickness = 2)

        # ---------------------------------------------------------------------
        # Draw eyes 

        # Draw left eye
        cv2.ellipse(img = img_src, 
                    center = (int(center_img[0] + width*0.0655),
                              int(center_img[0] - height*0.185)), 
                    axes = (int(width*0.04), 
                            int(height*0.02)), 
                    angle = 0, 
                    startAngle = 0, 
                    endAngle = 360, 
                    color = (0, 255, 0), 
                    thickness = 2)

        # Draw right eye
        cv2.ellipse(img = img_src, 
                    center = (int(center_img[0] - width*0.0655),
                              int(center_img[0] - height*0.185)), 
                    axes = (int(width*0.04), 
                            int(height*0.02)), 
                    angle = 0, 
                    startAngle = 0, 
                    endAngle = 360, 
                    color = (0, 255, 0), 
                    thickness = 2)

        # ---------------------------------------------------------------------
        # Print picture information
        str_list = ["Rotation ... {} [deg]".format(self.get_current_rotation()),
                    "Zoom ....... {} [%]".format(self.get_current_Zoom()),
                    "X_Offset ... {} [%]".format(self.get_current_X_Offset()),
                    "Y_Offset ... {} [%]".format(self.get_current_Y_Offset())]
        for idx, str_ in enumerate(str_list):
            cv2.putText(img = img_src, 
                        text = str_, 
                        org = (10, 20 + 18*idx), 
                        fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale = 0.4, 
                        color = (0, 0, 0), 
                        thickness = 3, 
                        lineType = cv2.LINE_AA)
            cv2.putText(img = img_src, 
                        text = str_, 
                        org = (10, 20 + 18*idx), 
                        fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale = 0.4, 
                        color = (0, 255, 255), 
                        thickness = 1, 
                        lineType = cv2.LINE_AA)

        # ---------------------------------------------------------------------

        # Print current picture information
        str_list = [
                    "Index ... {}/{}".format(self.idx + 1, len(self.pictures)),
                    "Data .... {} % ".format(self.idx*100/len(self.pictures)),
                    "Path .... {} ".format(os.path.basename(str(self.pictures[self.sorted_idx_list[self.idx]].path)) ),
                    "Name .... {} ".format(self.pictures[self.sorted_idx_list[self.idx]].file_name),
                    "Date .... {} ".format(time.ctime(self.pictures[self.sorted_idx_list[self.idx]].date))
                    ]
        y_pos = 390
        for idx, str_ in enumerate(str_list):
            cv2.putText(img = img_src, 
                        text = str_, 
                        org = (10, y_pos + 18*idx), 
                        fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale = 0.4, 
                        color = (0, 0, 0), 
                        thickness = 3, 
                        lineType = cv2.LINE_AA)
            cv2.putText(img = img_src, 
                        text = str_, 
                        org = (10, y_pos + 18*idx), 
                        fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale = 0.4, 
                        color = (255, 255, 0), 
                        thickness = 1, 
                        lineType = cv2.LINE_AA)

        if self.pictures[self.sorted_idx_list[self.idx]].auto_parameters:
            str_ = " Auto mode"
            color = (0, 255, 0)
        else:
            str_ = " Manual mode"
            color = (0, 69, 255)

        cv2.putText(img = img_src, 
                        text = str_, 
                        org = (self.video_size[0] - 130, 20), 
                        fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale = 0.5, 
                        color = (0, 0, 0), 
                        thickness = 3, 
                        lineType = cv2.LINE_AA)
        cv2.putText(img = img_src, 
                        text = str_, 
                        org = (self.video_size[0] - 130, 20), 
                        fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale = 0.5, 
                        color = color, 
                        thickness = 1, 
                        lineType = cv2.LINE_AA)

        if self.pictures[self.sorted_idx_list[self.idx]].include:
            str_ = " include"
            color = (0, 255, 0)
        else:
            str_ = " No include"
            color = (0, 69, 255)

        cv2.putText(img = img_src, 
                        text = str_, 
                        org = (self.video_size[0] - 130, 40), 
                        fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale = 0.5, 
                        color = (0, 0, 0), 
                        thickness = 3, 
                        lineType = cv2.LINE_AA)
        cv2.putText(img = img_src, 
                        text = str_, 
                        org = (self.video_size[0] - 130, 40), 
                        fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale = 0.5, 
                        color = color, 
                        thickness = 1, 
                        lineType = cv2.LINE_AA)


        # ---------------------------------------------------------------------
        # Return result
        return img_src

    def list_dataset(self, dataset):
            
        """ Lists all dataset absolute paths of pictures
        Args:
            dataset: `list`  list of datasets absolute paths to files
        Returns:
            None: `None`  No returns
        """

        # Print absolute paths to every file in list
        for data in dataset:
            print("\t", data)

    def create_video(self, filename, print_age = False, print_date = False, print_mask = False):
        
        """ Creates video with current pictures and settings
        Args:
            filename: `string`  file name for video file
        Returns:
            None: `None`  No returns
        """

        print("Creating video, please wait:\n")

        # Create video out variable
        video_out = cv2.VideoWriter(filename = filename, # format MP4
                                    fourcc = self.fourcc , # Codec H264
                                    fps = self.fps, 
                                    frameSize = tuple(self.video_size))

        # variables to print process progression
        total_progress = len(self.pictures)
        progress_porce = 0

        # Process variables
        idx_aux = self.idx - 1
        self.idx = -1

        # Iterate over all samples
        for idx in range(0, len(self.pictures) - 1): 

            # Read picture in index
            self.next_sample()

            if not self.pictures[self.sorted_idx_list[self.idx]].include:
                print("\tOmitted: {} - {}".format(idx, self.pictures[self.sorted_idx_list[self.idx]].file_name))
                continue

            img = self.get_edit_picture()

            # Resize picture to desired size
            img = cv2.resize(img, VIDEO_SIZE)

            # Overlay mask if option
            if print_mask:
                img = overlay_image(img, self.over_mask, (0, 0), 1)

            if print_age:
                pass

            if print_date:
                date = self.pictures[self.sorted_idx_list[self.idx]].date
                date = datetime.fromtimestamp(date)
                date = date.strftime('%Y-%m-%d')
                cv2.putText(img = img, 
                            text = date, 
                            org = (int(self.video_size[0]*0.42), 
                                   int(self.video_size[1]*0.95)), 
                            fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                            fontScale = 0.55, 
                            color = (0, 0, 0), 
                            thickness = 3, 
                            lineType = cv2.LINE_AA)
                cv2.putText(img = img, 
                            text = date, 
                            org = (int(self.video_size[0]*0.42), 
                                   int(self.video_size[1]*0.95)), 
                            fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                            fontScale = 0.55, 
                            color = (0, 255, 0), 
                            thickness = 1, 
                            lineType = cv2.LINE_AA)
 
            # Add margins
            img[0: self.video_size[1], 0:2] = (0, 0, 0) 
            img[0:2, 0: self.video_size[0]] = (0, 0, 0)
            img[0: self.video_size[1], self.video_size[0] - 2:self.video_size[0]] = (0, 0, 0) 
            img[self.video_size[1]-2: self.video_size[1], 0: self.video_size[0]] = (0, 0, 0)

            # Write current frame/picture in video out
            video_out.write(img)

            # Calculate progress and print it
            progress = float(idx*100./total_progress)
            if progress > progress_porce:
                progress_porce += 5
                print("{} %".format(round(progress, 2)))

        # Re-assign again index to its initial value
        self.idx = idx_aux

        # Get the sample in the initial index
        self.next_sample()

        # Release memory of video output
        video_out.release()

        # Merge audio
        audio_file = "every_day_audio.mp3"
        file_destination = "./"+os.path.splitext(filename)[0] + ".avi"
        if os.path.isfile(file_destination): os.remove(file_destination)
        merge_audio_command = "ffmpeg -hide_banner -loglevel panic -y -i {} -i {} -af apad -map 0:v -map 1:a -c:v copy -shortest {}".format(filename, audio_file, file_destination)
        os.system(merge_audio_command)

        # If process done successfully then print
        print("100 %\n")
        print("Video done, saved as: {}\n".format(filename))

    def next_sample(self):
                 
        """ Loads the next picture in dataset's list
        Args:
            None: `None`  No input arguments
        Returns:
             None: `None`  No returns
        """
 
        self.pictures[self.sorted_idx_list[self.idx]].save_information()

        if not self.idx == self.NumberofSamples - 1:
            self.idx += 1
            self.get_idx_sample(self.sorted_idx_list[self.idx])
        else:
            print("Warning: Maximum of pictures reached")

    def previous_sample(self):

        """ Loads the previous picture in data set's list
        Args:
            None: `None`  No input arguments
        Returns:
             None: `None`  No returns
        """
        
        if not self.idx == 0:
            self.pictures[self.sorted_idx_list[self.idx]].save_information()
            self.idx -= 1
            self.get_idx_sample(self.sorted_idx_list[self.idx])
            self.pictures[self.sorted_idx_list[self.idx]].load_information()
        else:
            print("Warning: Minimum of pictures reached")

    def rotate(self, img_src, angle, rotation_point):
         
        """ Rotates and image in its center
        Args:
            img_src: `cv2.math` image to rotate
            angle: `float` angle to rotate image
            rotation_point" `int` point to rotate image
        Returns:
            img_dst: `cv2.math` rotated image
        """

        # Calculate rotation matrix
        M = cv2.getRotationMatrix2D(rotation_point, 
                                    angle, 
                                    1)
                                
        # Apply rotation to matrix
        img_dst = cv2.warpAffine(img_src, M, self.video_size)

        # Return result 
        return img_dst

    def desplace_XY(self, img_src, X_Offset, Y_Offset):
        
        """ Moves in X and Y axis an image
        Args:
            img_src: `cv2.math` image to move in X and Y axis
            X_Offset: `int` value in pixels to move image in X axis
            Y_Offset: `int` value in pixels to move image in Y axis
        Returns:
            img_dst: `cv2.math` rotated image
        """

        # Calculate rotation matrix
        M = np.float32([[1, 0, X_Offset*self.video_size[0]],
                        [0, 1, Y_Offset*self.video_size[1]]])

        # Move image in X and Y axis
        img_dst = cv2.warpAffine(img_src, M, self.video_size)

        # Return result
        return img_dst

    def zoom_img(self, img_src, zoom):
                
        """ Zooms in or Zooms out an source image
        Args:
            img_src: `cv2.mat`  image to zoom out or zoom in
            zoom: `float` value of zoom to zoom in or zoom out [-1 to 1]
            
        Returns:
            img_src: `cv2.mat` zoomed image
        """

        # Get image dimensions
        height, width = img_src.shape[:2]

        # To zoom in
        if zoom > 0:

            zoom = 0.01 if not zoom else 1.0 - zoom
            if not zoom:
                zoom = zoom + 0.01 
        
            height_zoom = int(height*zoom)
            width_zoom  = int(width*zoom)
            
            Y = int((height - height_zoom)/2)
            X = int((width - width_zoom)/2)

            img_src = img_src[Y: Y + height_zoom, X: X+ width_zoom]        
            img_src = cv2.resize(img_src, (width, height))

        # To zoom out
        elif zoom < 0:

            height_zoom = height + abs(int(height*zoom))
            width_zoom  = width  + abs(int(width*zoom))
            
            # Create background mask
            black_image = np.zeros((height_zoom, width_zoom, 3), np.uint8)

            # Apply zoom out process
            x_offset = int((width_zoom - width)/2) 
            y_offset = int((height_zoom - height)/2) 

            black_image[y_offset:y_offset+img_src.shape[0], x_offset:x_offset+img_src.shape[1]] = img_src

            img_src = cv2.resize(black_image, (width, height))

        # Return result
        return img_src

    def save_features_current_sample(self):
               
        """ Save the current picture's features
        Args:
        Returns:
        """
        
        self.pictures[self.sorted_idx_list[self.idx]].save_information()

    def load_features_current_sample(self):
                
        """ Loads the current picture's features
        Args:
        Returns:
        """
        
        self.pictures[self.sorted_idx_list[self.idx]].load_information()

    def open_file_folder(self):
               
        """ Function description
        Args:
            variable_name: `type`  description
        Returns:
            variable_name: `type`  description
        """

        os.system('xdg-open "%s"' % self.pictures[self.sorted_idx_list[self.idx]].path)

    def set_parameters_picture(self, X_Offset, Y_Offset, Zoom, angle):

        """ Sets parameters or features to the current picture
        Args:
            X_Offset: `float` value in pixels to move image in X axis
            Y_Offset: `float` value in pixels to move image in Y axis
            Zoom: `float` zoom's value of current image
            angle: `float` angle's value of current image
        Returns:
        """

        self.pictures[self.sorted_idx_list[self.idx]].X_Offset = X_Offset
        self.pictures[self.sorted_idx_list[self.idx]].Y_Offset = Y_Offset
        self.pictures[self.sorted_idx_list[self.idx]].rotation = angle
        self.pictures[self.sorted_idx_list[self.idx]].zoom = Zoom 

    def get_edit_picture(self):

        """ Returns the current picture with all its associated modifications:
            rotations, displacement in X and Y axis, Zoom, and others
        Args:
        Returns:
            img: `cv2.math` current image with modifications
        """

        # Copy the current image to a new variable
        img = self.current_img_src.copy()

        if not self.pictures[self.sorted_idx_list[self.idx]].auto_parameters:

            # Get values for operations/modifications
            X_Offset = self.pictures[self.sorted_idx_list[self.idx]].X_Offset
            Y_Offset = self.pictures[self.sorted_idx_list[self.idx]].Y_Offset
            angle = self.pictures[self.sorted_idx_list[self.idx]].rotation
            Zoom = self.pictures[self.sorted_idx_list[self.idx]].zoom

            # Apply displacement in X and Y Axis
            img = self.desplace_XY(img, 
                                float(-(1000 - X_Offset)/1000.), 
                                float((1000 - Y_Offset)/1000.))

            # Apply rotations
            img = self.rotate(img, float((3600 - angle)/10.),
                            (int(self.video_size[0]/2), int(self.video_size[1]/2)))

            # Apply zoom
            img = self.zoom_img(img, float((Zoom - 1000)/1000. ))
            
        else:

            # Apply auto rotation
            self.predict_shapes(img_src = img)
            angle = self.pictures[self.sorted_idx_list[self.idx]].auto_rotation
            center = self.pictures[self.sorted_idx_list[self.idx]].eyes_center
            # print("\tAuto Angle : {}".format(angle))
            # print("\tAuto center: {}".format(center))
            img = self.rotate(img, angle, center)

            # Apply displacement in X and Y Axis
            self.predict_shapes(img_src = img)
            X_Offset = self.pictures[self.sorted_idx_list[self.idx]].auto_X_Offset
            Y_Offset = self.pictures[self.sorted_idx_list[self.idx]].auto_Y_Offset
            # print("\tAuto X_Offset : {}".format(X_Offset))
            # print("\tAuto Y_Offset: {}".format(Y_Offset))
            img = self.desplace_XY(img, X_Offset, Y_Offset)

            # Apply zoom
            self.predict_shapes(img_src = img)
            Zoom = self.pictures[self.sorted_idx_list[self.idx]].auto_zoom
            # print("\tAuto Zoom : {}".format(Zoom))
            img = self.zoom_img(img, Zoom)

            # Draw face's marks
            self.predict_shapes(img_src = img)
            img = self.face_marks.draw_shapes(
                img_src = img, 
                shape = self.pictures[self.sorted_idx_list[self.idx]].Face_Marks, 
                rects = self.pictures[self.sorted_idx_list[self.idx]].FaceShape_Rect)
            center = self.pictures[self.sorted_idx_list[self.idx]].eyes_center
            cv2.circle(img, center, 2, (0, 255, 0), -1)

        # Return result
        return img

    def get_current_parameters(self):

        """ Returns the current GUI parameters for the current picture 
        Args:
        Returns:
            Features: `list`  list of GUI features and their values for
                              the current picture
        """

        # First assignment
        X_Offset = self.pictures[self.sorted_idx_list[self.idx]].X_Offset
        Y_Offset = self.pictures[self.sorted_idx_list[self.idx]].Y_Offset
        rotation = self.pictures[self.sorted_idx_list[self.idx]].rotation
        __Zoom__ = self.pictures[self.sorted_idx_list[self.idx]].zoom
        
        # Features assignation
        Features = {"X_Offset":(X_Offset  , None), 
                    "Y_Offset":(Y_Offset  , None),
                    "Rotation":(rotation , None),
                    "__Zoom__":(__Zoom__ , None)}

        return Features

    def get_current_rotation(self):

        """ Gets the current rotation_value's value for the current picture 
        Args:
            rotation_value: `float` current rotation_value's value for the current picture 
        Returns:
        """

        rotation_value = self.pictures[self.sorted_idx_list[self.idx]].get_rotation()

        return rotation_value

    def get_current_X_Offset(self):

        """ Returns the current X_Offset's value for the current picture 
        Args:
            X_Offset_value: `float` current X_Offset's value for the current picture 
        Returns:
        """

        X_Offset_value = self.pictures[self.sorted_idx_list[self.idx]].get_X_Offset()

        return X_Offset_value

    def get_current_Y_Offset(self):

        """ Returns the current Y_Offset's value for the current picture 
        Args:
            Y_Offset_value: `float` current Y_Offset's value for the current picture 
        Returns:
        """

        Y_Offset_value = self.pictures[self.sorted_idx_list[self.idx]].get_Y_Offset()

        return Y_Offset_value

    def get_current_Zoom(self):

        """ Returns the current zoom's value for the current picture 
        Args:
            zoom_value: `float` current zoom's value for the current picture 
        Returns:
        """

        zoom_value = self.pictures[self.sorted_idx_list[self.idx]].get_Zoom()

        return zoom_value

    def assing_face_shape_features(self, marks, rects):


        """ Assign face shape features for the current picture file
        Args:
            marks: `list` list of face's marks
            rects: `list` list of face's rectangles
        Returns:
        """

        if len(marks) and len(rects):

            # Face
            self.pictures[self.sorted_idx_list[self.idx]].FaceShape = marks[0:17]
            self.pictures[self.sorted_idx_list[self.idx]].FaceShape_Rect = rects
            
            # left eyebrow
            self.pictures[self.sorted_idx_list[self.idx]].eyebow_left_rect = self.face_marks.get_face_boundbox(marks, 1)
            self.pictures[self.sorted_idx_list[self.idx]].eyebow_left_shape = marks[17:22]

            # right eyebrow
            self.pictures[self.sorted_idx_list[self.idx]].eyebow_right_rect = self.face_marks.get_face_boundbox(marks, 2)
            self.pictures[self.sorted_idx_list[self.idx]].eyebow_right_shape = marks[22:27]

            # left eye
            self.pictures[self.sorted_idx_list[self.idx]].eye_left_rect = self.face_marks.get_face_boundbox(marks, 3)
            self.pictures[self.sorted_idx_list[self.idx]].eye_left_shape = marks[36:42]

            #right eye
            self.pictures[self.sorted_idx_list[self.idx]].eye_right_rect = self.face_marks.get_face_boundbox(marks, 4)
            self.pictures[self.sorted_idx_list[self.idx]].eye_right_shape = marks[42:48]

            # nose
            self.pictures[self.sorted_idx_list[self.idx]].nose_rect = self.face_marks.get_face_boundbox(marks, 5)
            self.pictures[self.sorted_idx_list[self.idx]].nose_shape = marks[29:36]

            # mouth
            self.pictures[self.sorted_idx_list[self.idx]].mouth_rect = self.face_marks.get_face_boundbox(marks, 6)
            self.pictures[self.sorted_idx_list[self.idx]].mouth_shape = marks[48:68]

            eye_l = self.pictures[self.sorted_idx_list[self.idx]].eye_left_rect
            eye_r = self.pictures[self.sorted_idx_list[self.idx]].eye_right_rect

            # Calcualte eyes inclination0
            (elx, ely, elw, elh) = eye_l
            (erx, ery, erw, erh) = eye_r

            eye_l_center = (int(elx + elw * 0.5), int(ely + elh * 0.5))
            eye_r_center = (int(erx + erw * 0.5), int(ery + erh * 0.5))
            
            x = eye_r_center[0] - eye_l_center[0]
            y = eye_r_center[1] - eye_l_center[1]

            x1 = eye_l_center[0]; x2 = eye_r_center[0]
            y1 = eye_l_center[1]; y2 = eye_r_center[1]
            xc = int(x1 + (x2 - x1)*0.5)
            yc = int(y1 + (y2 - y1)*0.5)
    
            self.pictures[self.sorted_idx_list[self.idx]].eyes_inclination = \
                math.degrees(math.atan2(y, x))

            self.pictures[self.sorted_idx_list[self.idx]].eyes_center = (xc, yc)

            self.pictures[self.sorted_idx_list[self.idx]].Face_Marks = marks

            # -----------------------------------------------------------------
            # Assign auto parameters

            # For auto rotation
            self.pictures[self.sorted_idx_list[self.idx]].auto_rotation = \
                self.pictures[self.sorted_idx_list[self.idx]].eyes_inclination

            # For auto displacement
            xd = -float(xc - self.video_size[0]*.5)/float(self.video_size[0])
            yd = -float(yc - self.video_size[1]*.5)/float(self.video_size[1]) - 0.015
            self.pictures[self.sorted_idx_list[self.idx]].auto_X_Offset = xd
            self.pictures[self.sorted_idx_list[self.idx]].auto_Y_Offset = yd

            # For auto zoom
            x1 = self.pictures[self.sorted_idx_list[self.idx]].eye_left_shape[0][0]
            y1 = self.pictures[self.sorted_idx_list[self.idx]].eye_left_shape[0][1]

            x2 = self.pictures[self.sorted_idx_list[self.idx]].eye_right_shape[3][0]
            y2 = self.pictures[self.sorted_idx_list[self.idx]].eye_right_shape[3][1]

            dist = float(math.sqrt((x1 - x2)**2 + (y1 - y2)**2))
            dist_eyes = 135
            d = -(dist - dist_eyes)*0.005

            self.pictures[self.sorted_idx_list[self.idx]].auto_zoom = d

        else:
            print("Warning: No face shapes deteced")

    def predict_shapes(self, img_src):

        """ Draws face's shapes in image
        Args:
            img_src: `cv2.math` image to draw face's shapes
        Returns:
        """

        # Predict face's shapes
        shape, rects = self.face_marks.predict(img_src)

        # Assign face's features to current image
        self.assing_face_shape_features(shape, rects)

    def switch_to_auto_parameters(self):

        """ Switch current picture file to auto parameters
        Args:
        Returns:
        """

        self.pictures[self.sorted_idx_list[self.idx]].auto_parameters = \
            not self.pictures[self.sorted_idx_list[self.idx]].auto_parameters

        if self.pictures[self.sorted_idx_list[self.idx]].auto_parameters:
            print("Switched to auto mode")
        else:
            print("Switched to manual mode")

# Picture or sample class
class Picture():
    
    def __init__(self, abspath2img):
         
        """ Initializes variable class
        Args:
            abspath2img: `string`  absolute path to file/picture
        Returns:
        """

        head, tail = os.path.split(abspath2img)
        name, extension = os.path.splitext(tail)

        # ---------------------------------------------------------------------
        # Parameters of picture
        self.absolute_path = abspath2img
        self.extension = extension
        self.file_name = name
        self.path = head
        self.width = 0
        self.height = 0
        self.channels = 0

        self.include = True
        # ---------------------------------------------------------------------
        # Manual parameters
        self.rotation = 3600
        self.X_Offset = 1000
        self.Y_Offset = 1000
        self.zoom = 1000

        # ---------------------------------------------------------------------
        # Auto parameters
        self.auto_parameters = False
        self.auto_rotation = 0
        self.auto_X_Offset = 0
        self.auto_Y_Offset = 0
        self.auto_zoom = 0
        
        # ---------------------------------------------------------------------
        # Face's shapes from predictor
        self.Face_Marks = None

        self.FaceShape = None
        self.FaceShape_Rect = None

        self.eyebow_left_rect = None
        self.eyebow_left_shape = None

        self.eyebow_right_rect = None
        self.eyebow_right_shape = None

        self.eye_left_rect = None
        self.eye_left_shape = None

        self.eye_right_rect = None
        self.eye_right_shape = None

        self.nose_rect = None
        self.nose_shape = None

        self.mouth_rect = None
        self.mouth_shape = None

        self.eyes_inclination = 0
        self.eyes_center = None

        # ---------------------------------------------------------------------
        # File's dates
        self.date_modification = 0 # [time_stamp]
        self.date_file_format = 0 # [time_stamp]
        self.date_creation = 0 # [time_stamp]
        self.date = 0 # [time_stamp]

        # Get file's modification date
        MoDate = datetime.fromtimestamp(os.path.getmtime(abspath2img))
        self.date_modification = int(calendar.timegm(MoDate.timetuple()))

        # Get file's creation date
        CrDate = datetime.fromtimestamp(os.path.getctime(abspath2img)) 
        self.date_creation = int(calendar.timegm(CrDate.timetuple()))

        self.date_file_format = int(min([int(self.date_modification), int(self.date_creation)]))

        path_to_date_format = os.path.join(head, "date_format.txt")
        if os.path.isfile(path_to_date_format):
            with open (path_to_date_format, "r") as myfile:
                data = myfile.readlines()
                # print(data[0])
                if len(data):
                    data = data[0]

                    if data == "XXX_YYYYMMDD_HHMMSS\n":
                        if len(self.file_name) == 19:
                            Year    = int(self.file_name[4:8])
                            Month   = int(self.file_name[8:10])
                            Day     = int(self.file_name[10:12])
                            Hour    = int(self.file_name[13:15])
                            Minute  = int(self.file_name[15:17])
                            Second  = int(self.file_name[17:19])

                            date = datetime(Year, Month, Day, Hour, Minute, Second)
                            timestamp = calendar.timegm(date.timetuple())
                            self.date_file_format = int(timestamp)
                            self.date = int(timestamp)
                            return

                        else:
                            print("\t\tWarning: {} - invalid name date format".format(self.file_name))

                    if data == "XX_YYYYMMDD_HH_MM_SS_XXX\n":
                        if len(self.file_name) == 24:
                            Year    = int(self.file_name[3:7])
                            Month   = int(self.file_name[7:9])
                            Day     = int(self.file_name[9:11])
                            Hour    = int(self.file_name[12:14])
                            Minute  = int(self.file_name[15:17])
                            Second  = int(self.file_name[18:20])

                            date = datetime(Year, Month, Day, Hour, Minute, Second)
                            timestamp = calendar.timegm(date.timetuple())
                            self.date_file_format = int(timestamp)
                            self.date = int(timestamp)
                            return

                        else:
                            print("\t\tWarning: {} - invalid name date format".format(self.file_name))

                    if data == "XXX_YYYYMMDD_XXX\n":
                        if len(self.file_name) == 16:
                            Year    = int(self.file_name[4:8])
                            Month   = int(self.file_name[8:10])
                            Day     = int(self.file_name[10:12])

                            date = datetime(Year, Month, Day, 0, 0, 0)
                            timestamp = calendar.timegm(date.timetuple())
                            self.date_file_format = int(timestamp)
                            self.date = int(timestamp)
                            return

                        else:
                            print("\t\tWarning: {} - invalid name date format".format(self.file_name))

                    if data == "XX_YYYYMMDD_HH_MM_SS_XXXXX\n":
                        if len(self.file_name) == 26:
                            Year    = int(self.file_name[3:7])
                            Month   = int(self.file_name[7:9])
                            Day     = int(self.file_name[9:11])
                            Hour    = int(self.file_name[12:14])
                            Minute  = int(self.file_name[15:17])
                            Second  = int(self.file_name[18:20])

                            date = datetime(Year, Month, Day, Hour, Minute, Second)
                            timestamp = calendar.timegm(date.timetuple())
                            self.date_file_format = int(timestamp)
                            self.date = int(timestamp)
                            return

                        else:
                            print("\t\tWarning: {} - invalid name date format".format(self.file_name))

                    if data == "XX_YYYYMMDD_XXX\n":
                        if len(self.file_name) == 15:
                            Year    = int(self.file_name[3:7])
                            Month   = int(self.file_name[7:9])
                            Day     = int(self.file_name[9:11])

                            date = datetime(Year, Month, Day, 0, 0, 0)
                            timestamp = calendar.timegm(date.timetuple())
                            self.date_file_format = int(timestamp)
                            self.date = int(timestamp)
                            return

                        else:
                            print("\t\tWarning: {} - invalid name date format".format(self.file_name))

        self.date = int(min([int(self.date_modification), int(self.date_file_format), int(self.date_creation)]))

        self.load_information()

    def get_image(self):

        """ loads and adjusts the current image 
        Args:
        Returns:
            img: `cv2.math` source image read
        """

        # Read the original image in path
        img = cv2.imread(str(self.absolute_path))

        # If no image read then report and return
        if not len(img):
            print("Error: can not open file:")
            print("\t{}".format(self.absolute_path))
            return []

        # Get image shape
        h, w, c = img.shape

        # Assign parameters
        self.width = w
        self.height = h
        self.channels = c

        # Return result
        return img

    def save_information(self):

        """ Saves information of current picture/sample to a npz file
        Args:
        Returns:
        """

        # absolute path to save npz file
        file_path = os.path.join(str(self.path), str(self.file_name) + ".npz")

        # Remove any previous file
        if os.path.isfile(file_path):
            os.remove(file_path)

        try:
            # Save information in file
            np.savez_compressed(file_path, 
                absolute_path  = self.absolute_path,
                extension = self.extension,
                file_name = self.file_name,
                path = self.path,

                width  = self.width,
                height = self.height,
                channels  = self.channels,

                include = self.include,

                rotation  = self.rotation,
                X_Offset  = self.X_Offset,
                Y_Offset  = self.Y_Offset,
                zoom  = self.zoom,

                Auto_params = self.auto_parameters,
                auto_rotation =  self.auto_rotation,
                auto_X_Offset = self.auto_X_Offset,
                auto_Y_Offset = self.auto_Y_Offset,
                auto_zoom = self.auto_zoom,

                Face_Marks = self.Face_Marks,

                FaceShape = self.FaceShape,
                FaceShape_Rect = self.FaceShape_Rect,

                eyebow_left_rect = self.eyebow_left_rect,
                eyebow_left_shape = self.eyebow_left_shape,

                eyebow_right_rect = self.eyebow_right_rect,
                eyebow_right_shape = self.eyebow_right_shape,

                eye_left_rect = self.eye_left_rect,
                eye_left_shape = self.eye_left_shape,

                eye_right_rect = self.eye_right_rect,
                eye_right_shape = self.eye_right_shape,
                
                nose_rect = self.nose_rect,
                nose_shape = self.nose_shape,

                mouth_rect = self.mouth_rect,
                mouth_shape = self.mouth_shape,

                eyes_inclination = self.eyes_inclination,
                eyes_center = self.eyes_center,

                date_modification = self.date_modification,
                date_file_format = self.date_file_format,
                date_creation = self.date_creation,
                date = self.date

                )

        # Print exception
        except ValueError:
            print("Oops!  something went wrong loading parameters file for:")
            print("\t {}.{}".format(self.file_name, self.extension))

    def load_information(self):
        
        """ Loads information of current picture/sample form its npz file
        Args:
        Returns:
        """

        # path to npz file path
        file_path = os.path.join(str(self.path), str(self.file_name) + ".npz")

        # If there's a file for picture parameters then do ...
        if os.path.isfile(file_path):
            try:
                # Load file in variable
                npzfile = np.load(file_path)

                # self.absolute_path = npzfile['absolute_path']  
                # self.extension = npzfile['extension']  
                # self.file_name = npzfile['file_name']  
                # self.channels = npzfile['channels']  

                self.rotation = npzfile['rotation']  
                self.X_Offset = npzfile['X_Offset']  
                self.Y_Offset = npzfile['Y_Offset']  
                self.zoom = npzfile['zoom']  

                self.height = npzfile['height']  
                self.width = npzfile['width']  

                self.include = npzfile['include']  

                # self.path = npzfile['path']  
                # self.date = npzfile['date']  
                
            # Print exception
            except ValueError:
                print("Oops!  something went wrong loading parameters file for:")
                print("\t {}.{}".format(self.file_name, self.extension))

        else:
            print("{}: No parameters file".format(self.file_name))
                
        # Read picture's date
        if not self.date:
            try:
                mtime = os.path.getmtime(self.absolute_path)
            except OSError:
                mtime = 0
            self.date = datetime.fromtimestamp(mtime)

    def get_rotation(self):

        """ Returns current picture rotation angle value
        Args:
            None: `None`  No input arguments 
        Returns:
            rotation: `float` rotations's value
        """

        rotation = (3600 - self.rotation)/10.

        return rotation

    def get_X_Offset(self):

        """ Returns current picture X offset value
        Args:
            None: `None`  No input arguments 
        Returns:
            X_Offset: `float` X axis offset value
        """

        X_Offset = - (1000 - self.X_Offset)/10.

        return X_Offset

    def get_Y_Offset(self):

        """ Returns current picture Y offset value
        Args:
            None: `None`  No input arguments 
        Returns:
            Y_Offset: `float` Y axis offset value
        """

        Y_Offset = - (1000 - self.Y_Offset)/10.

        return Y_Offset

    def get_Zoom(self):
        
        """ Returns current picture zoom value
        Args:
            None: `None`  No input arguments 
        Returns:
            zoom: `float` zoom's value
        """

        zoom = - (1000 - self.zoom)/10.

        return zoom

    def reset_parameters(self):

        """ Resets parameters to default values
        Args:
            None: `None`  No input arguments 
        Returns:
            None: `None`  No returns 
        """

        # Re-assign variables to default values
        self.rotation = 3600
        self.X_Offset = 1000
        self.Y_Offset = 1000
        self.zoom = 1000

# Face and eyes detector models
class FaceDetector():
    
    def __init__(self):
        
        """ Create face detector class
        Args:
            None: `None`  No input arguments 
        Returns:
            None: `None`  No returns 
        """

        self.face_cascade = None
        self.eye_cascade = None

        # Absolute path to face model
        face_cascade_path = 'haarcascade_frontalface_default.xml'

        # Absolute path to eyes model
        eyecascade_path = 'haarcascade_eye.xml'

        # Load face model
        if not os.path.isfile(face_cascade_path):
            print("Error:\tFace Haarcascade file does not exist")
        else:
            print("Face detector:\t", face_cascade_path)
            self.face_cascade = cv2.CascadeClassifier(face_cascade_path)

        # Load eyes model
        if not os.path.isfile(face_cascade_path):
            print("Error:\tEye Haarcascade file does not exist")
        else:
            print("Eye detector:\t", eyecascade_path)
            self.eye_cascade = cv2.CascadeClassifier(eyecascade_path)

    def detected_face(self, img_src):

        """ Detects faces and eyes in an image
        Args:
            img_src: `cv2.math`  image to detect faces and eyes
        Returns:
            face_detections: `list`  list of detected faces
            eyes_detections: `list`  list of detected eyes
        """

        # Detection variables
        face_detections = []
        eyes_detections = []

        # Critical conditions
        if isinstance(self.face_cascade, type(None)):
            print("Face model has not been loaded")
            return face_detections, eyes_detections
        if isinstance(self.eye_cascade, type(None)):
            print("Eye model has not been loaded")
            return face_detections, eyes_detections

        # Image source dimensions
        _, _, channels = img_src.shape()

        if channels == 3: # Convert to gay scale
            img_src = cv2.cvtColor(img_src, cv2.COLOR_RGB2GRAY)

        # Get detected faces with haar cascades
        face_detections = self.face_cascade.detectMultiScale(img_src, 1.3, 5)

        # Get detected eyes with haar cascades in every face detected
        for (x,y,w,h) in face_detections:
            eyes_detections = self.eye_cascade.detectMultiScale(img_src[y:y+h, x:x+w])

        # Return detections
        return face_detections, eyes_detections
    
    def draw_detection(self, img_src, face_detections, eyes_detections):
        
        """ Draws detected faces and detected eyes in an image
        Args:
            img_src: `cv2.math`  image to draw eyes and face detections
            face_detections: `list`  list of detected faces
            eyes_detections: `type`  list of detected eyes
        Returns:
            img_src: `cv2.math` image with detections drawn
        """

        # Draw face detections
        for (x,y,w,h) in face_detections:
            cv2.rectangle(img_src, (x,y), (x+w,y+h), (255,0,0), 2)
        
        # Draw eyes detections
        for (ex,ey,ew,eh) in eyes_detections:
            cv2.rectangle(img_src,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    
        # Return result image
        return img_src

# Face land marks model
class FacialLandMarks():

    def __init__(self):

        """ initializes the variable class, loading the model to detect face shape
        Args:
        Returns:
        """

        # initialize dlib's face detector (HOG-based) and then create
        # the facial landmark predictor
        print("\nLoading facial landmark predictor...")

        # link to model: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        model = "snapchat-filters-opencv/filters/shape_predictor_68_face_landmarks.dat"
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(model) 
        
        print("\tFace marks detector loaded, Done ...")

    def predict(self, img_src):

        """ Predicts face shape on a given source image
        Args:
            img_src: `cv2.mat` source image to predict face shapes
        Returns:
            shape: `list` list of (X, Y) Coordinates of face shape
        """

        # Change color space to gray scale
        img_src_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)

        # detect faces in the gray scale frame
        rects = self.detector(img_src_gray, 0)

        shape = []

        # loop over the face detections
        for rect in rects:

            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy array
            shape = self.predictor(img_src_gray, rect)
            shape = face_utils.shape_to_np(shape)

        return shape, rects

    def draw_shapes(self, img_src, shape, rects):

        """ Predicts face shape on a given source image
        Args:
            img_src: `cv2.mat` input image to draw face's shapes on it
            marks: `list` list of face's marks
            rects: `list` list of face's rectangles
        Returns:
            img_src: `cv2.mat` input image with face's shapes drawn on it
        """

        if (isinstance(shape, type(None)) or isinstance(rects, type(None))):
            return img_src

        for idx_rect, rect in enumerate(rects):

            x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()

            # cv2.rectangle(img_src, (x, y), (x+w, y+h), (0, 0, 255), 2)
            # cv2.putText(img = img_src, 
            #             text = str(idx_rect), 
            #             org = (x, y), 
            #             fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
            #             fontScale = 0.4, 
            #             color = (0, 255, 255), 
            #             thickness = 2, 
            #             lineType = cv2.LINE_AA)

            for i in range(1,7):

                (x, y, w, h) = self.get_face_boundbox(shape, i)
                
                # cv2.rectangle(img_src, (x, y), (x + w, y + h), (0, 255, 0), 1)
                # cv2.putText(img = img_src, 
                #             text = str(i), 
                #             org = (x, y), 
                #             fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                #             fontScale = 0.4, 
                #             color = (255, 255, 255), 
                #             thickness = 2, 
                #             lineType = cv2.LINE_AA)

                # loop over the (x, y)-coordinates for the facial landmarks
                # and draw them on the image
                for idx, (x, y) in enumerate(shape):
                    cv2.circle(img_src, (x, y), 1, (0, 0, 255), -1)

                    # # Draw marks enumeration
                    # cv2.putText(img = img_src, 
                    #             text = str(idx), 
                    #             org = (x, y), 
                    #             fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                    #             fontScale = 0.2, 
                    #             color = (0, 0, 0), 
                    #             thickness = 3, 
                    #             lineType = cv2.LINE_AA)
                    # cv2.putText(img = img_src, 
                    #             text = str(idx), 
                    #             org = (x, y), 
                    #             fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                    #             fontScale = 0.2, 
                    #             color = (0, 255, 255), 
                    #             thickness = 1, 
                    #             lineType = cv2.LINE_AA)

        return img_src

    def calculate_inclination(self, point1, point2):

        """ Calculates the inclination in degrees of two given points
        Args:
            point1: `tuple` (X, Y) coordinate of first point
            point2: `tuple` (X, Y) coordinate of second point
        Returns:
            incl: `float` angle of inclination between point1 and point2 [deg]
        """

        x1,x2,y1,y2 = point1[0], point2[0], point1[1], point2[1]
        incl = -180/math.pi*math.atan((float(y2-y1))/(x2-x1))

        return incl

    def calculate_boundbox(self, list_coordinates):

        """ Calculates the bounding box given a list of points
        Args:
            list_coordinates: `list`  list of (X, Y) coordinates of points
                                      to calculate bounding box
        Returns:
            x: `int`  superior left X coordinate point of bounding box
            y: `int`  superior left Y coordinate point of bounding box
            w: `int`  Width of bounding box
            h: `int`  Height of bounding box
        """

        x = min(list_coordinates[:,0])
        y = min(list_coordinates[:,1])
        w = max(list_coordinates[:,0]) - x
        h = max(list_coordinates[:,1]) - y

        return (x,y,w,h)

    def get_face_boundbox(self, points, face_part):
        
        """ Calculates the bounding box of a component of the face
            with the face land marks
        Args:
            points: `list`  list of (X, Y) coordinates of face land marks
            face_part: `int`  part of face to calculate bounding box and select
                              land marks
        Returns:
            x: `int`  superior left X coordinate point of bounding box
            y: `int`  superior left Y coordinate point of bounding box
            w: `int`  Width of bounding box
            h: `int`  Height of bounding box
        """

        if face_part == 1: #left eyebrow
            (x, y, w, h) = self.calculate_boundbox(points[17:22]) 

        elif face_part == 2: #right eyebrow
            (x, y, w, h) = self.calculate_boundbox(points[22:27]) 

        elif face_part == 3: #left eye
            (x, y, w, h) = self.calculate_boundbox(points[36:42]) 

        elif face_part == 4: #right eye
            (x, y, w, h) = self.calculate_boundbox(points[42:48]) 

        elif face_part == 5: #nose
            (x, y, w, h) = self.calculate_boundbox(points[29:36]) 

        elif face_part == 6: #mouth
            (x, y, w, h) = self.calculate_boundbox(points[48:68]) 

        return (x, y, w, h)

# =============================================================================
# FUNCTIONS - FUNCTIONS - FUNCTIONS - FUNCTIONS - FUNCTIONS  - FUNCTIONS - FUNC
# =============================================================================
def mouse_callback(event, x, y, flags, params):

    """ Sets a mouse event in mouse_callback 
    Args:
        x: `int` X axis coordinate of click
        y: `int` Y axis coordinate of click
    Returns:
        None: `None`  No returns 
    """

    global cord_target
    global refresh

    # this function will be called whenever the mouse is right-clicked
    # right-click event value is 2
    if event == 2:
        if x == cord_target[0] and y == cord_target[1]:
            cord_target = (-1, -1)
            print("Target cord deleted")
        else:
            cord_target = (x, y)
            print("X:{}/Y:{}".format(x, y))
        
        refresh = True

def update_GUI(x):
            
    """ Change the state to refresh or not the GUI
    Args:
        None: `None`  No input arguments 
    Returns:
        None: `None`  No returns 
    """
    
    # Change the state of GUI
    global refresh
    refresh = True

def create_GUI(GUI_WINDOW_NAME):

    """ Creates GUI with given features
    Args:
        GUI_WINDOW_NAME: `string`  GUI's window name
    Returns:
        None: `None`  No returns   
    """

    # default parameters and features
    Features = {"X_Offset":(1000 , 2000), 
                "Y_Offset":(1000 , 2000),
                "Rotation":(3600 , 3600*2),
                "__Zoom__":(1000 , 2000)}

    # Window parameters
    cv2.namedWindow(GUI_WINDOW_NAME)            # Show tune window
    cv2.resizeWindow(GUI_WINDOW_NAME, 500, 100) # Resize Window

    # Create a track bar for each specified featured
    for Param in Features:
        cv2.createTrackbar(Param, 
                           GUI_WINDOW_NAME,
                           Features[Param][0], 
                           Features[Param][1], 
                           update_GUI)

def get_gui_features(GUI_WINDOW_NAME, Params):
    
    """ Returns GUI's parameters and their values
    Args:
        GUI_WINDOW_NAME: `string`  GUI's window name
        Params: `list`  list of GUI's parameters, features and values
    Returns:
        None: `None`  No returns   
    """

    parameters = np.zeros(len(Params))
    for idx, Param in enumerate(Params):
        parameters[idx] = cv2.getTrackbarPos(Param, GUI_WINDOW_NAME)

    # Return parameters and their values
    return parameters

def set_gui_features(GUI_WINDOW_NAME, Params):

    """ Sets GUI's parameters and values in GUI's widgets
    Args:
        GUI_WINDOW_NAME: `string`  GUI's window name
        Params: `list`  list of GUI's parameters, features and values
    Returns:
        None: `None`  No returns   
    """

    # Assign value for GUI's track bars
    for Param in Params:
        cv2.setTrackbarPos(Param, GUI_WINDOW_NAME, int(Params[Param][0]))

def save_gui_features(file_path, VIDEO_SIZE, Index):

    """ Saves GUI's features
        Args:
            file_path: `string`  path to save GUI settings
            VIDEO_SIZE: `tuple`  (width, height) desired video size or dimensions
            Index: `int` current picture gallery index

        Returns:
            None: `None`  No returns
    """

    # Remove previous file if exits 
    if os.path.isfile(file_path):
        os.remove(file_path)

    # Save in file parameters
    np.savez_compressed(file_path, 
        VIDEO_SIZE  = VIDEO_SIZE,
        Index = Index)

    # Print info
    print("GUI parameters saved\n")

def load_gui_features(file_path):

    """ Loads GUI features
        Args:
            file_path: `string`  path where GUI settings are located
        Returns:
            None: `None`  No returns
    """

    # If there's not GUI parameters to load then
    if not os.path.isfile(file_path):
        print("No GUI parameters file to load\n")
        return None, None

    else:
        try:
            # Load GUI settings file
            npzfile = np.load(file_path)

            # Load parameters
            VIDEO_SIZE = npzfile['VIDEO_SIZE']  
            Index = npzfile['Index']  
  
            # Return parameters
            return VIDEO_SIZE, Index

        except ValueError:
            print("Oops!  something went wrong loading GUI parameters\n")

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):

    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def overlay_image(l_img, s_img, pos, transparency):

    """ Overlies 's_img on' top of 'l_img' at the position specified by
        pos and blend using 'alpha_mask' and 'transparency'.
    Args:
        l_img: `cv2.mat` inferior image to overlay superior image
        s_img: `cv2.mat` superior image to overlay
        pos: `tuple`  position to overlay superior image
        transparency: `float`  variable description
    Returns:
        l_img: `cv2.mat` original image with s_img overlayed
    """

    # Get superior image dimensions
    s_img_height, s_img_width, s_img_channels = s_img.shape

    if s_img_channels == 3 and transparency != 1:
        s_img = cv2.cvtColor(s_img, cv2.COLOR_BGR2BGRA)
        s_img_channels = 4

    # Take 3rd channel of 'img_overlay' image to get shapes
    img_overlay= s_img[:, :, 0:4]

    # cords assignation to overlay image 
    x, y = pos

    # Image ranges
    y1, y2 = max(0, y), min(l_img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(l_img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], l_img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], l_img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return l_img

    if s_img_channels == 4:
        # Get alphas channel
        alpha_mask = (s_img[:, :, 3] / 255.0) * transparency
        alpha_s = alpha_mask[y1o:y2o, x1o:x2o]
        alpha_l = (1.0 - alpha_s)

        # Do the overlay with alpha channel
        for c in range(0, l_img.shape[2]):
            l_img[y1:y2, x1:x2, c] = (alpha_s * img_overlay[y1o:y2o, x1o:x2o, c] +
                                    alpha_l * l_img[y1:y2, x1:x2, c])

    elif s_img_channels < 4:
        # Do the overlay with no alpha channel
        if l_img.shape[2] == s_img.shape[2]:
            l_img[y1:y2, x1:x2] = s_img[y1o:y2o, x1o:x2o]
        else:
            print("Error: to overlay images should have the same color channels")
            return l_img

    # Return results
    return l_img

# =============================================================================
# MAIN FUNCTION - MAIN FUNCTION - MAIN FUNCTION - MA[-IN FUNCTION - MAIN FUNCTION
# IMPLEMENTATION EXAMPLE - IMPLEMENTATION EXAMPLE - IMPLEMENTATION EXAMPLE - IM
# =============================================================================
if __name__ == "__main__":

    """ With this program you can create a every day video, just set the "data_Set_path"
        variable to the path where you have you picture gallery located, and then
        just run the script and follow the options shown in console 
    Args:
    Returns:
    """

    # Print some process information
    info_str =  "\nMenu Options:\n"+\
                "\n\tQ:\t Quit"+\
                "\n\tN:\t Next picture"+\
                "\n\tB:\t Previous picture"+\
                "\n\tH:\t Show help\n"+\
                "\nPicture Options:\n"+\
                "\n\tS:\t Save parameters"+\
                "\n\tL:\t Load parameters"+\
                "\n\tR:\t Reset parameters"+\
                "\n\tI:\t Include to video"+\
                "\n\tA:\t Picture auto adjustment\n"+\
                "\n\t7:\t Zoom in"+\
                "\n\t4:\t Zoom out"+\
                "\n\t8:\t Rotate to left"+\
                "\n\t9:\t Rotate to right"+\
                "\n\t5:\t Move to left  X axis "+\
                "\n\t6:\t Move to right X axis"+\
                "\n\t3:\t Move to up    Y axis"+\
                "\n\t2:\t Move to down  Y axis"+\
                "\n\n\tC:\t Create video\n"
                
    # Process variables
    GUI_WINDOW_NAME = "Every_day_maker"
    VIDEO_SIZE = (640, 480)
    refresh = True

    # Create window to show results and user interface
    cv2.namedWindow(GUI_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    # Set mouse action in created window
    cv2.setMouseCallback(GUI_WINDOW_NAME, mouse_callback)

    data_idx = 0
    gui_file = "gui_settings.npz"
    VIDEO_SIZE_, idx_dataset = load_gui_features(gui_file)

    if (not isinstance(VIDEO_SIZE_, type(None)) and 
        not isinstance(idx_dataset, type(None))): 
        VIDEO_SIZE = tuple(VIDEO_SIZE_)
        data_idx = idx_dataset

    if len(sys.argv)>1: # Assign with user input
        data_idx =  int(sys.argv[1])

    data_Set_path = '/media/john/SHDEXP/JOHN_DATA/Personales/Proyectos/EVERY_DAY/Original' 

    # Read data set
    data_Set = Dataset(data_Set_path, 
                       VIDEO_SIZE, 
                       data_idx)

    previous_user_input = None
    input_factor = 1

    # If there's nothing to process then quit
    if not data_Set.NumberofSamples:
        exit()
    
    # Create GUI
    create_GUI(GUI_WINDOW_NAME)
    set_gui_features(GUI_WINDOW_NAME, data_Set.get_current_parameters())

    # Print info and options
    print(info_str)

    # Start loop for operations in pictures
    while True and data_Set.NumberofSamples:

        if refresh:

            # Change refresh state
            refresh = False

            # Get and assign current picture features
            features = data_Set.get_current_parameters()
            Zoom, angle, Y_Offset, X_Offset = get_gui_features(GUI_WINDOW_NAME, 
                                                                features)

            # Show current picture with operations
            data_Set.set_parameters_picture(X_Offset, Y_Offset, Zoom, angle)
            img = data_Set.get_edit_picture()
            img = data_Set.draw_desired_geometric(img)
            cv2.imshow(GUI_WINDOW_NAME, img)

        # ---------------------------------------------------------------------
        # Read user input
        User_input = cv2.waitKey(40) & 0xFF
        if User_input == previous_user_input:
            input_factor += 0.3
        else:
            input_factor = 1
            previous_user_input = User_input

        # Press 'Q' to quit
        if User_input == ord('q') or User_input == ord('Q'):
            break

        # Press 'N' to next sample
        if User_input == ord('n') or User_input == ord('N'):
            data_Set.next_sample()
            set_gui_features(GUI_WINDOW_NAME, 
                             data_Set.get_current_parameters())
            refresh = True
            continue

        # Press 'B' to previous sample
        if User_input == ord('b') or User_input == ord('B'):
            data_Set.previous_sample()
            set_gui_features(GUI_WINDOW_NAME, 
                             data_Set.get_current_parameters())
            refresh = True
            continue

        # Press 'S' to save sample features
        if User_input == ord('s') or User_input == ord('S'):
            data_Set.save_features_current_sample()
            print("{}: parameters saved".format(data_Set.pictures[data_Set.idx].file_name))
            continue

        # Press 'O' open current picture folder
        if User_input == ord('o') or User_input == ord('O'):
            data_Set.open_file_folder()
            continue

        # Press 'L' to Load sample features
        if User_input == ord('l') or User_input == ord('L'):
            data_Set.pictures[data_Set.idx].load_information()
            set_gui_features(GUI_WINDOW_NAME, 
                             data_Set.get_current_parameters())
            print("{}: parameters loaded".format(data_Set.pictures[data_Set.idx].file_name))
            refresh = True
            continue

        # Press 'R' to Reset
        if User_input == ord('r') or User_input == ord('R'):
            data_Set.pictures[data_Set.idx].reset_parameters()
            set_gui_features(GUI_WINDOW_NAME, 
                             data_Set.get_current_parameters())
            print("{}: parameters reseted".format(data_Set.pictures[data_Set.idx].file_name))
            continue

        # Press 'A' to auto adjustment
        if User_input == ord('a') or User_input == ord('A'):
            data_Set.switch_to_auto_parameters()
            refresh = True
            continue

        # Press 'I' to include current picture
        if User_input == ord('i') or User_input == ord('I'):
            data_Set.include_current_picture()
            refresh = True
            continue

        # Press 'H' to show options
        if User_input == ord('h') or User_input == ord('H'):
            print(info_str)
            continue

        # Press 'C' to create video
        if User_input == ord('c') or User_input == ord('C'):
            data_Set.create_video(filename = "every_day.mp4", 
                                  print_age = True, 
                                  print_date = True, 
                                  print_mask = True)
            pass

        # ---------------------------------------------------------------------
        # Press '4' to small adjustment in zoom
        if User_input == 183:
            cv2.setTrackbarPos('__Zoom__', 
                               GUI_WINDOW_NAME, 
                               int((Zoom + 1*input_factor)))
            refresh = True
            continue
  
        # Press '7' to small adjustment in zoom
        if User_input == 180:
            cv2.setTrackbarPos('__Zoom__', 
                               GUI_WINDOW_NAME, 
                               int((Zoom - 1*input_factor)))
            refresh = True
            continue
        
       # Press '6' to small adjustment in Y_Offset
        if User_input == 178:
            cv2.setTrackbarPos('Y_Offset', 
                               GUI_WINDOW_NAME, 
                               int(Y_Offset - 1*input_factor))
            refresh = True
            continue
  
        # Press '5' to small adjustment in Y_Offset
        if User_input == 179:
            cv2.setTrackbarPos('Y_Offset', 
                               GUI_WINDOW_NAME, 
                               int(Y_Offset + 1*input_factor))
            refresh = True
            continue
  
        # Press '8' to small adjustment in X_Offset
        if User_input == 182:
            cv2.setTrackbarPos('X_Offset', 
                               GUI_WINDOW_NAME, 
                               int(X_Offset + 1*input_factor))
            refresh = True
            continue
  
        # Press '9' to small adjustment in X_Offset
        if User_input == 181:
            cv2.setTrackbarPos('X_Offset', 
                               GUI_WINDOW_NAME, 
                               int(X_Offset - 1*input_factor))
            refresh = True
            continue

        # Press '/' to small adjustment in Rotation
        if User_input == 185:
            cv2.setTrackbarPos('Rotation', 
                               GUI_WINDOW_NAME, 
                               int(angle + 1*input_factor))
            refresh = True
            continue
  
        # Press '*' to small adjustment in Rotation
        if User_input == 184:
            cv2.setTrackbarPos('Rotation', 
                               GUI_WINDOW_NAME, 
                               int(angle - 1*input_factor))
            refresh = True
            continue
        # ---------------------------------------------------------------------
    
    # Destroy any created windows
    cv2.destroyAllWindows()

    # Save GUI parameters
    save_gui_features(gui_file, VIDEO_SIZE, data_Set.idx)
        
# =============================================================================