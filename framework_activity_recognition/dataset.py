import os
import torch
import random as rn
import numpy as np
from torch.utils.data import Dataset
from framework_activity_recognition.processing import loadVideo, random_select, loadVideoSequence


class FileBasedDataset(Dataset):
    """An abstract class representing a Dataset, with provided list of files
    and corresponding annotations.

        All subclasses should override the  ``__getitem__`` method.
     Arguments:

        data_file_list (List of strings (file paths)): contains list of Video-File paths.

        ground_truth_label_list: list containing ground truth annotation

        target_tensor (Tensor): contains sample targets (labels).

        annotation_converter: an array to convert original annotation string to the

        integer-annotations, used for training. If not specified, a new converter is
        created based on found classes.

        CAUTION: create the annotation converter (created when the argument  annotation_converter is set to NONE)
        only once, at the beginning, with one of your Dataset objects. Use/set this converter (retrieved by __get_annotation_converter__)
        for each of the following Datasets. Training and Test datasets should have the same annotation_converter.
        We need the annotation converter, so that the resulting labels are 0, 1, 2, ... N (for pytorch)

    """
    #Todo: save the annotation converter. Currently done in the network wrapper. This should probably change

    def __init__(self, data_file_list, ground_truth_label_list,
                 annotation_converter = None, transform = None,
                 dynamicLoading = True, return_file_path = False, return_triplet = False, w2w_embedding = False, model = None):

        self.data_file_list = data_file_list
        self.annotations_original = ground_truth_label_list  # annotations as original stri
        self.transform = transform
        self.data_lookup = {}
        self.return_file_path = return_file_path
        self.return_triplet = return_triplet
        self.model = model
        self.w2w_embedding = w2w_embedding

        if (self.w2w_embedding):
            assert self.model is not None
            self.w2w_embeddings = word2vector_utils.get_word2vec_matrix_for_string_list(model=model, class_names= self.annotations_original)
            print ("Computed W2W embeddings, Shape:")
            print (np.shape(self.w2w_embeddings))

        assert len(self.data_file_list) == len(self.annotations_original)

        # IDs become new annotations, can be mapped with this list
        # If the annotation mapping list was NOT provided:
        if annotation_converter is None:
            self.annotation_converter = list(sorted(set(self.annotations_original)))
        else:
            self.annotation_converter = annotation_converter

        self.annotations_transformed = [self.annotation_converter.index(a) for a in self.annotations_original]  # annotations as int (mapping stored in self.annotation_converter)

        self.nClasses = len(self.annotation_converter)

        if not dynamicLoading:

            for i in range(len(self.data_file_list)):

                self.dataLookup[self.data_file_list[i]] = self.__loaditem__(i)  # Load whole video


    def __len__(self):
        return len(self.data_file_list)

    def __get_samples_weight__(self):
        class_sample_count = np.array(
            [len(np.where(self.annotations_transformed == t)[0]) for t in np.unique(self.annotations_transformed)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in self.annotations_transformed])

        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()

        return(samples_weight)

    def __get_annotation_converter__(self):
        return self.annotation_converter


    def __loaditem__(self, index):
        raise NotImplementedError

    #Todo: check what it was doing here and if it should be deleted or re-written
    """
    def __get_annotations__(self):
        return self.annotations

    def __get_files__(self):
        return self.files
    """

    def __getitem__(self, index):

        sample, label = self.__loaditem__(index)

        # Data augmentation: apply transform
        if self.transform:
            sample = self.transform(sample)

        if (self.return_triplet):

            ind_neg = random.sample(
                [i for i in range(len(self.annotations_transformed)) if self.annotations_transformed[i] != label], 1)[0]
            ind_pos = random.sample(
                [i for i in range(len(self.annotations_transformed)) if self.annotations_transformed[i] == label], 1)[0]

            sample_neg, label_neg = self.__loaditem__(ind_neg)
            sample_pos, label_pos = self.__loaditem__(ind_pos)
            #print (label)
            #print(label_pos)
            #print(label_neg)
            #print ("-----")

            if self.transform:
                sample_neg = self.transform(sample_neg)
                sample_pos = self.transform(sample_pos)

            if (self.return_file_path):
                return sample, label,  sample_pos,label_pos, sample_neg,label_neg, self.data_file_list[index]
            else:
                return sample, label,  sample_pos,label_pos, sample_neg,label_neg

            #self.annotations_original =
            #annotations_transformed[index]

        #No Triplet needed

        if (self.w2w_embedding):
            if (self.return_file_path):
                return sample, label, self.transform(self.w2w_embeddings[index]),self.data_file_list[index]
            else:
                return sample, label, self.transform(self.w2w_embeddings[index])

        if (self.return_file_path ):

            return sample, label, self.data_file_list[index]

        else:

            return sample, label


class VideoFileBasedDataset(FileBasedDataset):
    """Dataset reading videos with imageio utility.

    Arguments:
        dir_path (String): Path to the top level directory
        split (String): "train" or "test"- split
        annotations (list): List with classes
        resize_string: string or None
            None if use the original size
            "widthxheight" if the videos should be resized (e.g. "160x120")
        selectFramesNr: int or None
            None if the whole video should be returned
            number of frames to sample, if the frames should be uniformly sampled along the video
            in this case the video will be evenly divided in N snippets and one frame will be sampled from each snippet

    """

    def __init__(self, data_file_list, ground_truth_label_list, annotation_converter = None, transform = None, resize_string = None, selectFramesNr = None):

        super(VideoFileBasedDataset, self).__init__(data_file_list, ground_truth_label_list, annotation_converter, transform)

        self.resize_string = resize_string
        self.selectFramesNr = selectFramesNr


    #Loads the data item, without applying the transform!
    def __loaditem__ (self, index):
        path = self.data_file_list[index]
        _, classname = os.path.split(os.path.split(path)[0])

        #Load data

        data = loadVideo(path, rescale = self.resize_string)

        if self.selectFramesNr is not None:
            data = random_select(data, self.selectFramesNr)


        label = self.annotations_transformed[index]  # Get ground truth label

        return data, label

class DriveNActDataset(FileBasedDataset):
    
    """
    Arguments:
        df (pandas.DataFrame):
            df should contain full path to the videos (/cvhci/data/activity/Pakos/final_dataset/pakos_videos_ids_1/vp1/..mp4)
            (column 'file_id'), frame where the annotation starts (column 'frame_start'), frame where the annotation ends (column 'frame_end'),
            annotation/activity performed in the respective frame (column 'activity') and chunks (column 'chunk_id')
        labels (list):
            contains ground truth labels
        annotation_converter (np.ndarray/list):
            array to convert original annotation (String) into integer annotation
        transform:
            transformation performed to the video
    """
    def __init__(self, df, labels, annotation_converter = None, transform = None, resize = None):
        #TODO INHERIT FROM FileBasedDataset
        super().__init__(df, labels, annotation_converter, transform)
        self.resize = resize
    
    def __loaditem__(self, index):
        if torch.is_tensor(index):
            index = index.toList()

        file_path = self.data_file_list.loc[index, 'file_id']
        frame_start = self.data_file_list.loc[index, 'frame_start']
        frame_end = self.data_file_list.loc[index, 'frame_end']
        #label = self.data_file_list.loc[index, "activity"]  # String label
        #label = self.annotation_converter.index(label)  # Convert to Integer label

        frames = loadVideoSequence(file_path, frame_start, frame_end, self.resize)

        label = self.annotations_transformed[index]
        return frames, label

    def __len__(self):
        return len(self.data_file_list.index)