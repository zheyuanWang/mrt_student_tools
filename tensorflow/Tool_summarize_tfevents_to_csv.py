#!/usr/bin/env python3
"""
This is a enhanced version of https://gist.github.com/ptschandl/ef67bbaa93ec67aba2cab0a7af47700b
This script exctracts variables from all logs from tensorflow event files ("event*"),
writes them to Pandas and finally stores them a csv-file or pickle-file including all (readable) runs of the logging directory.

# from https://github.com/theRealSuperMario/supermariopy/blob/master/scripts/tflogs2pandas.py
"""

#import tensorflow as tf
import glob
import os
import pandas as pd
import traceback
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pprint
import time
from random import randrange



# Extraction function
def tflog2pandas(path: str) -> pd.DataFrame:
    """convert single tensorflow log file to pandas DataFrame
    
    Parameters
    ----------
    path : str
        path to tensorflow log file
    
    Returns
    -------
    pd.DataFrame
        converted dataframe
    """
    DEFAULT_SIZE_GUIDANCE = {
        "compressedHistograms": 1,
        "images": 1,
        "scalars": 0,  # 0 means load all
        "histograms": 1,
    }
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]  # extract all tags in TFevents
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))  # event_list.value -> list
            step = list(map(lambda x: x.step, event_list))  # event_list.step -> list
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    print(runlog_data)
    return runlog_data


def tflog2pandas_record_folder(path: str) -> pd.DataFrame:
    """convert single tensorflow log file to pandas DataFrame

    Parameters
    ----------
    path : str
        path to tensorflow log file

    Returns
    -------
    pd.DataFrame
        converted dataframe
    """
    DEFAULT_SIZE_GUIDANCE = {
        "compressedHistograms": 1,
        "images": 1,
        "scalars": 0,  # 0 means load all
        "histograms": 1,
    }
    runlog_data = pd.DataFrame({"expriment_name": [], "metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]  # extract all tags in TFevents
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))  # event_list.value -> list
            step = list(map(lambda x: x.step, event_list))  # event_list.step -> list
            #r = {"metric": [tag] * len(step), "value": values, "step": step}
            expriment_name = path.split(os.sep)[-3]
            r = {"expriment_name": [expriment_name]* len(step), "metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data


def many_logs2pandas(event_paths):
    all_logs = pd.DataFrame()
    for path in event_paths:
        #log = tflog2pandas(path)
        log= tflog2pandas_record_folder(path)
        if log is not None:
            if all_logs.shape[0] == 0:
                all_logs = log
            else:
                all_logs = all_logs.append(log, ignore_index=True)
    return all_logs

"""
import click

@click.command()
@click.argument("logdir-or-logfile")
@click.option(
    "--write-pkl/--no-write-pkl", help="save to pickle file or not", default=False
)
@click.option(
    "--write-csv/--no-write-csv", help="save to csv file or not", default=True
)
@click.option("--out-dir", "-o", help="output directory", default="/tmp/")
def main(logdir_or_logfile: str, write_pkl: bool, write_csv: bool, out_dir: str):
"""
# this following function was main, changed name in order to call it without flags
def extrac_tfevent_folder(logdir_or_logfile: str, write_pkl=False, write_csv=True, out_dir="/tmp/"):
    """
    Example usage:
    # create csv file from all tensorflow logs in provided directory (.) and write it to folder "./converted"
    Tool_summarize_tfevents_to_csv.py . --write-csv --no-write-pkl --o converted
    # creaste csv file from tensorflow logfile only and write into and write it to folder "./converted"
    Tool_summarize_tfevents_to_csv.py tflog.hostname.12345 --write-csv --no-write-pkl --o converted
    """
    pp = pprint.PrettyPrinter(indent=4)
    logdir_or_logfile=logdir_or_logfile
    if os.path.isdir(logdir_or_logfile):
        # Get all event* runs from logging_dir subdirectories
        event_paths = set(glob.glob(os.path.join(logdir_or_logfile, "event*")))
        event_paths.update(glob.glob(os.path.join(logdir_or_logfile, "*/event*")))
        event_paths.update(glob.glob(os.path.join(logdir_or_logfile, "*/*/event*")))
        event_paths.update(glob.glob(os.path.join(logdir_or_logfile, "*/*/*/event*")))

    elif os.path.isfile(logdir_or_logfile):
        event_paths = [logdir_or_logfile]
    else:
        raise ValueError(
            "input argument {} has to be a file or a directory".format(
                logdir_or_logfile
            )
        )
    # Call & append
    if event_paths:
        pp.pprint("Found tensorflow logs to process:")
        pp.pprint(event_paths)
        all_logs = many_logs2pandas(event_paths)
        pp.pprint("Head of created dataframe")
        pp.pprint(all_logs.head())

        os.makedirs(out_dir, exist_ok=True)
        if write_csv:
            save_to_name= "tflogs_" + logdir_or_logfile.split(os.sep)[-1] + "__"+str(randrange(99))
            out_file = os.path.join(out_dir, save_to_name + ".csv")
            print("saving to csv file {}".format(out_file))
            all_logs.to_csv(out_file, index=None)
        if write_pkl:
            out_file = os.path.join(out_dir, "all_training_logs_in_one_file.pkl")
            print("saving to pickle file {}".format(out_file))
            all_logs.to_pickle(out_file)
    else:
        print("No event paths have been found.")


if __name__ == "__main__":
    NOCUT = "/mrtstorage/users/zwang/github_zheyuan/mrt_experiments/m3l_xyr_cutnet2_centerFZ"
    IMPORTANT = "/mrtstorage/users/zwang/github_zheyuan/mrt_experiments/important" # baseline
    TESTS = "/mrtstorage/users/zwang/github_zheyuan/mrt_experiments/no_reuse_tests"
    REUSE = "/mrtstorage/users/zwang/github_zheyuan/mrt_experiments/m3l_range1_conv1x1_interval1_dr10"
    CENTER = "/mrtstorage/users/zwang/github_zheyuan/mrt_experiments/m3l_xyr_cutnet2_centerFZ"

    date = time.strftime("%m%d")
    out_dir_today="/tmp/tfevents_extract_" + date +"/"

    extrac_tfevent_folder(NOCUT, out_dir=out_dir_today)
    extrac_tfevent_folder(CENTER, out_dir=out_dir_today)
    extrac_tfevent_folder(REUSE, out_dir=out_dir_today)
    extrac_tfevent_folder(TESTS, out_dir=out_dir_today)
    extrac_tfevent_folder(IMPORTANT, out_dir=out_dir_today)

