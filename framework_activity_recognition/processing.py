import os
import shutil

import math
import imageio
import numpy as np
from random import randint
import cv2
import skvideo
from PIL import Image



def get_entity_by_module_path(entity_path):
    """
    This method returns instance of a class given in the entity path
    Arguments:
        entity_path: path to the class
    """
    #location = config_file["architecture"]["location"]
    parts = entity_path.split('.')
    module = ".".join(parts[:-1])
    entity = __import__(module)
    for part in parts[1:]:
        entity = getattr(entity, part)
    return entity


def extractFilesFromDirWhichMatchList(folder, string_match_filenames, string_not_match_filenames=[], full_path=False):
    """
    This methods return files, whose name match string_math_filenames and exclude string_not_match_filenames.
    Arguments:
        folder: folder in which the file should be filtered
        string_match_filenames: list of keywords to be included
        string_not_match_filenames: list of keywords to be excluded
        full_path: if True, then use full path for matching, else use only file name
    """
    result_files = []
    print (folder)
    for (dirpath, dirnames, filenames) in os.walk(folder):
        for filename in filenames:
            ok = True

            if full_path:

                check = os.path.join(dirpath, filename)
                # Only include if matches
                for s in string_match_filenames:
                    if (check.find(s) == -1):  # if not found
                        ok = False

                # Exclude if matches
                for s in string_not_match_filenames:
                    if (check.find(s) != -1):  # if found
                        ok = False

            else:
                # Only include if matches
                for s in string_match_filenames:
                    if (filename.find(s) == -1):  # if not found
                        ok = False

                # Exclude if matches
                for s in string_not_match_filenames:
                    if (filename.find(s) != -1):  # if found
                        ok = False

            if ok:
                file_path = os.path.join(dirpath, filename)
                result_files.append(file_path)

    return result_files


#Todo: proper documentation
def normalize_color_input_zero_center_unit_range(frames, max_val = 255.0):
    """
    This method normalizes all pixel in the frame to unit range and mean of 0 based on the given max_val
    Arguments:
        frames: frames to be normalized
        max_val: maximum value of a pixel
    """

    frames = (frames / max_val) * 2 - 1
    return(frames)

#Todo: proper documentation
def normalize_color_input_zero_center_unit_range_per_channel(frames):
    """
       Takes multiple frames as ndarray with shape
       (frame id, height, width, channels) and normalizes to unit range and mean of 0 channel-wise.
       frames: numpy
           all frames (e.g. video) with shape
           (frame id, height, width, channels)
       Returns
       -------
       Numpy: frames
           normalized frames (channel-wise)
       """
    last_dimension_ind = len(np.shape(frames))-1 # last dimension is the channel dimension

    for curr_channel in range(np.shape(frames)[last_dimension_ind]):
        #For each channel (probably there is three of them)
        curr_max = np.max(frames[...,curr_channel])
        curr_min = np.min(frames[..., curr_channel])
        curr_range = curr_max-curr_min

        frames[..., curr_channel] = ((frames[..., curr_channel]-curr_min)/curr_range)* 2 - 1

        return (frames)

def unit_range_zero_center_to_unit_range_zero_min(frames):

    frames = (frames + 1)/2
    return frames

def random_select(frames, n):
    """
    Takes multiple frames as ndarray with shape
    (frame id, height, width, channels) and selects
    randomly n-frames. If n is greater than the number
    of overall frames, placeholder frames (zeros) will
    be added.

    frames: numpy
        all frames (e.g. video) with shape
        (frame id, height, width, channels)
    n: int
        number of desired randomly picked frames

    Returns
    -------
    Numpy: frames
        randomly picked frames with shape
        (frame id, height, width, channels)
    """
    number_of_frames = np.shape(frames)[0]
    if number_of_frames < n:
        # Add all frames
        selected_frames = []
        for i in range(number_of_frames):
            frame = frames[i, :, :, :]
            selected_frames.append(frame)

        # Fill up with 'placeholder' images
        #frame = np.zeros(frame.shape)
        if len(np.shape(frames)) > 1:
            frame = np.zeros(frames[0,:,:,:].shape)
        else:
            frame = np.zeros((224,224,3))
        for i in range(n - number_of_frames):
            selected_frames.append(frame)

        return np.array(selected_frames)

    # Selected random frame ids
    frame_ids = set([])
    while len(frame_ids) < n:
        frame_ids.add(randint(0, number_of_frames - 1))

    # Sort the frame ids
    frame_ids = sorted(frame_ids)

    # Select frames
    selected_frames = []
    for id in frame_ids:
        frame = frames[id, :, :, :]
        selected_frames.append(frame)

    return np.array(selected_frames)

#Todo: add zeros if too small
def center_crop(frames, height, width, pad_zeros_if_too_small = True):
    """
    Takes multiple frames as ndarray with shape
    (frame id, height, width, channels) and crops all
    frames centered to desired width and height.

    frames: numpy
        all frames (e.g. video) with shape
        (frame id, height, width, channels)
    height: int
        height of the resulting crop
    width: int
        width of the resulting crop

    Returns
    -------
    Numpy: frames
        centered cropped frames with shape
        (frame id, height, width, channels)
    """

    frame_height = np.shape(frames)[1]
    frame_width = np.shape(frames)[2]

    t = np.shape(frames)[0]
    channels = np.shape(frames)[3]

    if pad_zeros_if_too_small and (height > frame_height or width > frame_width):
        # desired width
        frames_new = np.zeros((t, max(frame_height, height), max(frame_width, width), channels))
        # fill with the old data
        frames_new[0:t, 0:frame_height, 0:frame_width, 0:channels] = frames
        frames = frames_new
        frame_height = np.shape(frames)[1]
        frame_width = np.shape(frames)[2]


    origin_x = (frame_width - width) / 2
    origin_y = (frame_height - height) / 2

    # Floor origin (miss matching input sizes)
    # E.g. input width of 171 and crop width 112
    # would result in a float.
    origin_x = math.floor(origin_x)
    origin_y = math.floor(origin_y)

    return frames[:,
                  origin_y: origin_y + height,
                  origin_x: origin_x + width,
                  :]


def random_crop(frames, height, width, pad_zeros_if_too_small = True):
    """
    Takes multiple frames as ndarray with shape
    (frame id, height, width, channels) and crops all
    frames randomly to desired width and height.

    frames: numpy
        all frames (e.g. video) with shape
        (frame id, height, width, channels)
    height: int
        height of the resulting crop
    width: int
        width of the resulting crop

    Returns
    -------
    Numpy: frames
        randomly cropped frames with shape
        (frame id, height, width, channels)
    """

    frame_height = np.shape(frames)[1]
    frame_width = np.shape(frames)[2]
    t = np.shape(frames)[0]
    channels = np.shape(frames)[3]
    
    if pad_zeros_if_too_small and (height>frame_height or width > frame_width):
        #desired width
        frames_new = np.zeros((t,max(frame_height,height),max(frame_width,width),channels))
        #fill with the old data
        frames_new[0:t,0:frame_height,0:frame_width,0:channels] =frames
        frames = frames_new
        frame_height = np.shape(frames)[1]
        frame_width = np.shape(frames)[2]


    #Pad with zeros if too small
    
    origin_range_x = frame_width - width
    origin_range_y = frame_height - height

    origin_x = randint(0, origin_range_x)
    origin_y = randint(0, origin_range_y)

    return frames[:,
                  origin_y: origin_y + height,
                  origin_x: origin_x + width,
                  :]


def horizontal_flip(frames):
    """
    Flips all frames horizontally.Keeps the same shape and
    order of the frames. Expects a ndarray with
    shape (frame id, height, width, channels)

    Parameters
    ----------
    frames: numpy
        all frames (e.g. video) with shape
        (frame id, height, width, channels)

    Returns
    -------
    Numpy: frames
        all flipped frames of the video
        with shape (frame id, height, width, channels)
    """
    return np.flip(frames, axis=2)


def random_horizontal_flip(frames):
    """
    Randomly flips all frames horizontally with a chance of 50%.
    Keeps the same shape and order of the frames. Expects a
    ndarray with shape (frame id, height, width, channels)

    Parameters
    ----------
    frames: numpy
        all frames (e.g. video) with shape
        (frame id, height, width, channels)

    Returns
    -------
    Numpy: frames
        all flipped or original frames of the video
        with shape (frame id, height, width, channels)
    """
    flip = bool(randint(0, 1))

    if flip:
        return horizontal_flip(frames)

    return frames

def loadVideo(filepath, rescale=None, verbose=False, start_frame = 0, n_frames = 0):
    """
    Extracts all frames of a video and combines them to a ndarray with
    shape (frame id, height, width, channels)

    Parameters
    ----------
    filepath: str
        path to video file including the video name
        (e.g '/your/file/video.avi')
    rescale: str
        rescale input video to desired resolution (e.g. rescale='160x120')
    verbose: bool
        hide or display debug information

    Returns
    -------
    Numpy: frames
        all frames of the video with (frame id, height, width, channels)
    """
    # Opens video file with imageio.
    # Returns an 'empty' numpy with shape (1,1) if the file can not be opened.



    #print (filepath, start_frame, n_frames)
    new_dimensions = None
    if rescale:
        #Old: rescaling with ffmpeg. Does not work anymore
        #kwargs = {'size': rescale, "ffmpeg_params":['-loglevel', '-8']}
        #Dimensions for rescaling with PIL
        new_dimensions = list(map(int, rescale.split('x')))


    #print (filepath)

    cap = cv2.VideoCapture(filepath)

    if (start_frame > 0):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Interate over frames in video
    images = []
    count = 0
    max_video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - start_frame-1

    if (n_frames > max_video_length):
        #print(filepath, start_frame, n_frames)
        #print ("n_frames >= max_video_length")
        pass

    if (n_frames > 0 and n_frames <= max_video_length):
        video_length = n_frames
    else:
        video_length = max_video_length


    while cap.isOpened():
        # Extract the frame
        ret, image = cap.read()
        if rescale:
            #Convert to PIL image for rescaling
            image = Image.fromarray(image)
            image = image.resize(new_dimensions, resample=Image.BILINEAR)
            # Convert back to numpy array
            image = np.array(image)



        count = count + 1

        #print(np.shape(images))

        # If there are no more frames left
        #print ("len(np.shape(image)): "+str(len(np.shape(image))))
        if (count > video_length -1 or (len(np.shape(image))<2)):

            cap.release()
            # Print stats
            if (verbose):
                print("Done extracting frames.\n%d frames extracted" % count)
                print("-----")
                print(np.shape(image))
                print(np.shape(images))
            break

        images.append(image)



    images = np.array(images)

    # Print debug information
    if verbose:
        print(np.shape(images))

        print(filepath)
        print(np.shape(images))
        print (start_frame)
        print (n_frames)
        print ("---")

    return images


def loadVideoSequence(filepath, frame_start, frame_end, resize=""):
    """
    Extracts video sequence based on given filepath from frame_start until frame_end and resize the frames to a size given by resize string
    Arguments:
        filepath: path to the video data
        frame_start: frame index of video sequence start
        frame_end: frame index of video sequence end
        resize: string denoting the size of transformed frames
    """
    frame_count = frame_end - frame_start
    counter = 0
    frames = []
    dim = (0,0)

    if not os.path.isfile(filepath):
        print("no file with path ", filepath)
        return
    
    if resize:
        dim = list(map(int, resize.lower().split('x')))
        if len(dim) != 2:
            print("wrong rescale dimension ", str(len(dim)))

    cap = cv2.VideoCapture(filepath)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print("can't receive frame nr ", str(counter + frame_start), "from video ", filepath)
            print("the video has ", str(max_frames), " frames")
            break

        if resize:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = frame.resize(dim)
            frame = np.array(frame)

        if counter >= frame_count or len(np.array(frame)) < 2:
            break

        counter += 1
        frames.append(frame)

    frames = np.array(frames)
    cap.release()
    cv2.destroyAllWindows()
    
    return frames

#Todo: move image resizing to Transforms!
def loadVideoOld(filepath, rescale=None, verbose=False):
    """
    Extracts all frames of a video and combines them to a ndarray with
    shape (frame id, height, width, channels)

    Parameters
    ----------
    filepath: str
        path to video file including the video name
        (e.g '/your/file/video.avi')
    rescale: str
        rescale input video to desired resolution (e.g. rescale='160x120')
    verbose: bool
        hide or display debug information

    Returns
    -------
    Numpy: frames
        all frames of the video with (frame id, height, width, channels)
    """
    # Opens video file with imageio.
    # Returns an 'empty' numpy with shape (1,1) if the file can not be opened.

    kwargs = {}
    new_dimensions = None
    if rescale:
        #Old: rescaling with ffmpeg. Does not work anymore
        #kwargs = {'size': rescale, "ffmpeg_params":['-loglevel', '-8']}
        #Dimensions for rescaling with PIL
        new_dimensions = list(map(int, rescale.split('x')))



    reader = imageio.get_reader(filepath, 'ffmpeg', **kwargs)


    # Interate over frames in video
    images = []
    for image in reader:

        if rescale:
            #Convert to PIL image for rescaling
            image = Image.fromarray(image)
            image = image.resize(new_dimensions, resample=Image.BILINEAR)
            # Convert back to numpy array
            image = np.array(image)

        images.append(image)


    images = np.array(images)

    # Print debug information
    if verbose:
        print(np.shape(images))


    return images



def visualizeNumpyVideoTensor( input , waitKey = 100):
    """
      Visualizes the video frames in with opencv
       given a numpy tensor of shape (frame id, height, width, channels)

       Parameters
       ----------
       input: numpy array
           all frames of the video with (frame id, height, width, channels)
       waitKey: if >0 : interval to wait for the next frame (in ms)
                if <= 0: waits until the next key is pressed to proceed to the next frame
        waitKey
    """
    for frame_id in range(np.shape(input)[0]):
        frame = input[frame_id,:,:,:]
        cv2.imshow('Video Frame', frame)
        cv2.waitKey(waitKey)

    cv2.destroyAllWindows()


def convertImagesToVideoFFMPEG(input_dir, output_file, regexp = "%*.jpg", framerate =30, ignore_if_exists = True):
    """
    Converts images in a folder specified by input_dir to a video with specified framerate and named output_file
    #ffmpeg -r 30 -i %*.jpg -vcodec mpeg4 -y ~/workspace/movie.mp4
    Arguments:
        input_dir: folder containing images to be converted
        output_file: name of output file
        regexp: regular expression to select all images to be converted
        framerate: framerate of the FFMPEG video
        ignore_if_exists: If True and there is a file in directory with name output_file, skip this method
    """
    if (ignore_if_exists and os.path.isfile(output_file)):
        "File {} already exists. doing nothing".format(output_file)
        return

    if  not os.path.isdir(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print ("Converting images from directory {}".format(input_dir))
    command = "cd {}; ffmpeg -r {} -i {} -vcodec mpeg4 -y {}".format(input_dir, framerate, regexp, output_file)
    os.system(command)
    print("Done. Saved video file: {}".format(output_file))



# TODO CLEAN THIS FUNCTION
def downsampleVideo(videoName, resizedName, x=320, y=240, r=None):
    """
    Downsample a video
    Arguments:
        videoName: name of video to be downsampled
        resizedName: name of downsampled video
        x: x axis resolution
        y: y axis resolution
        r: resulting frame rate
    """
    # ffmpeg -i movie.mp4 -vf scale=640:360 movie_360p.mp4

    if not os.path.isfile(resizedName):
        command = "/home/aroitberg/workspace/libs/fastvideofeat-master/bin/dependencies/bin/ffmpeg -i " + videoName + " -vf " + "scale=" + str(
            x) + ":" + str(y) + " " + resizedName
        if r is not None:
            command = command + " -r " + str(r)
        print ("Executing command: \n" + command)

        command = command.replace("(", "\(")
        command = command.replace(")", "\)")
        command = command.replace(";", "\;")
        command = command.replace("&", "\&")
        os.system(command)
        # subprocess.check_call(command, shell=True)
        print ("Converted video " + videoName + " to " + resizedName);

    else:
        "File " + resizedName + "already exists."


"""

"""

# TODO CLEAN THIS FUNCTION
def downsampleVideoList(files_to_be_processed, output_path, suffix_to_be_added="_downsampled.avi",
                        video_file_ending=".avi", x=320, y=240, copy_dir_structure_level=0, r=None):
    """
    Downsample a list of videos
    Arguments:
        files_to_be_processed: a list of video name to be downsampled
        output_path: path to the directory to store downsampled videos
        suffix_to_be_added: suffix to be added to the video names to denoted downsampled videos
        video_file_ending: video extension
        x: resolution of x axis
        y: resolution of y axis
        copy_dir_structure_level: If 0 then output directly to the output path, if not 0 then copy the directory structure in the end of output path 
        r: resulting frame rate
    """
    nfiles = len(files_to_be_processed)

    if (copy_dir_structure_level == 0):
        for video_file in files_to_be_processed:

            print (str(nfiles) + " files left.")

            new_filename = os.path.basename(video_file).replace(video_file_ending, suffix_to_be_added)

            if output_path == "":
                output_path = os.path.dirname(os.path.realpath(video_file))
            file_output_path = os.path.join(output_path, new_filename)
            downsampleVideo(video_file, file_output_path, x=x, y=y, r=r)

            nfiles = nfiles - 1

    else:

        for video_file in files_to_be_processed:

            print (str(nfiles) + " files left.")
            dirname = os.path.basename(os.path.dirname(video_file))

            dirname = os.path.join(output_path, dirname)
            if not (os.path.isdir(dirname)):  # Create directory if it does not exist
                print ("Creating directory " + dirname)
                os.mkdir(dirname)

            new_filename = os.path.basename(video_file).replace(video_file_ending, suffix_to_be_added)

            file_output_path = os.path.join(dirname, new_filename)
            downsampleVideo(video_file, file_output_path, x=x, y=y)

            nfiles = nfiles - 1

