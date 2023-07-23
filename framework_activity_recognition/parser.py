import os
import pandas as pd
import sys
from torch.functional import split
from framework_activity_recognition.processing import extractFilesFromDirWhichMatchList


def parseDriveNActSplitFiles(split_nr=0, folder_splits="/cvhci/data/activity/Pakos/vpfaefflin/activities_3s/", \
    data_path="/cvhci/data/activity/Pakos/final_dataset/pakos_videos_ids_1/", \
        views=["a_column_co_driver"], new_suffix=".mp4"):
    """
    This method reads the path to Drive&Act train and validation data with given split, path and views and convert the paths into pandas DataFrame instance
    Arguments:
        split_nr: split of the data
        folder_splits: path to folder containing splits
        data_path: path to video data
        views: list of type of view
        new_suffix: new suffix for the videos
    """

    train_df = pd.DataFrame()
    test_df = pd.DataFrame()

    for view in views:
        train_csv_path = folder_splits + view + "/midlevel.chunks_90.split_" + str(split_nr) + ".train.csv"
        test_csv_path = folder_splits + view + "/midlevel.chunks_90.split_" + str(split_nr) + ".val.csv"

        train_file = pd.read_csv(train_csv_path, usecols=['file_id', 'annotation_id', 'frame_start', 'frame_end', 'activity', 'chunk_id'])
        test_file = pd.read_csv(test_csv_path, usecols=['file_id', 'annotation_id', 'frame_start', 'frame_end', 'activity', 'chunk_id'])

        train_df = pd.concat([train_df, replacePathInDf(train_file, data_path, new_suffix)], ignore_index=True, copy=False)
        test_df = pd.concat([test_df, replacePathInDf(test_file, data_path, new_suffix)], ignore_index=True, copy=False)

    print("get ", str(len(train_df.index)), " training data from ", str(len(train_file['file_id'].unique())), " video files")
    print("get ", str(len(test_df.index)), " validation data from ", str(len(test_file['file_id'].unique())), " video files")

    return train_df, test_df

def parseDriveNActTestSplitFiles(split_nr=0, folder_splits="/cvhci/data/activity/Pakos/vpfaefflin/activities_3s/", \
    data_path="/cvhci/data/activity/Pakos/final_dataset/pakos_videos_ids_1/", \
        views=["a_column_co_driver"], new_suffix=".mp4"):
    """
    This method reads the path to Drive&Act test data with given split, path and views and convert the paths into pandas DataFrame instance
    Arguments:
        split_nr: split of the data
        folder_splits: path to folder containing splits
        data_path: path to video data
        views: list of type of view
        new_suffix: new suffix for the videos
    """
    
    test_df = pd.DataFrame()

    for view in views:
        test_csv_path = folder_splits + view + "/midlevel.chunks_90.split_" + str(split_nr) + ".test.csv"

        test_file = pd.read_csv(test_csv_path, usecols=['file_id', 'annotation_id', 'frame_start', 'frame_end', 'activity', 'chunk_id'])

        test_df = pd.concat([test_df, replacePathInDf(test_file, data_path, new_suffix)], ignore_index=True, copy=False)

    print("get ", str(len(test_df.index)), " test data from ", str(len(test_file['file_id'].unique())), " video files")

    return test_df

def replacePathInDf(df, data_path, new_suffix):
    """
    This method replaces path in the DataFrame df with path to the data and add new suffix in the end of the path
    Arguments:
        df: DataFrame instance with paths to be replaced
        data_path: path to the video data
        new_suffix: new suffix of the video data
    """
    df_output = pd.DataFrame()
    unique_files = df['file_id'].unique()
    for video_path in unique_files:
        new_file_path = data_path + video_path + new_suffix
        if os.path.isfile(new_file_path):
            file_rows = df[df['file_id'] == video_path].copy()
            file_rows['file_id'] = new_file_path
            df_output = pd.concat([df_output, file_rows], ignore_index=True, copy=False)
        else:
            error_message = "File " + new_file_path + " does not exist!"
            sys.exit(error_message)

    return df_output