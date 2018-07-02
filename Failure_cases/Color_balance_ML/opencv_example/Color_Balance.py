#!/usr/bin/env python
from __future__ import print_function
import os, sys, argparse, json
import numpy as np
import scipy.io
import cv2 as cv
import timeit


def load_json(path):
    f = open(path, "r")
    data = json.load(f)
    return data


def save_json(obj, path):
    tmp_file = path + ".bak"
    f = open(tmp_file, "w")
    json.dump(obj, f, indent=2)
    f.flush()
    os.fsync(f.fileno())
    f.close()
    try:
        os.rename(tmp_file, path)
    except:
        os.remove(path)
        os.rename(tmp_file, path)


def parse_sequence(input_str):
    if len(input_str) == 0:
        return []
    else:
        return [o.strip() for o in input_str.split(",") if o]


def stretch_to_8bit(arr, clip_percentile = 2.5):
    arr = np.clip(arr * (255.0 / np.percentile(arr, 100 - clip_percentile)), 0, 255)
    return arr.astype(np.uint8)


def evaluate(im, algo, gt_illuminant, i, range_thresh, bin_num, dst_folder, model_folder):
    new_im = None
    start_time = timeit.default_timer()
    if algo=="grayworld":
        inst = cv.xphoto.createGrayworldWB()
        inst.setSaturationThreshold(0.95)
        new_im = inst.balanceWhite(im)
    elif algo=="nothing":
        new_im = im
    elif algo.split(":")[0]=="learning_based":
        model_path = ""
        if len(algo.split(":"))>1:
            model_path = os.path.join(model_folder, algo.split(":")[1])
        inst = cv.xphoto.createLearningBasedWB(model_path)
        inst.setRangeMaxVal(range_thresh)
        inst.setSaturationThreshold(0.98)
        inst.setHistBinNum(bin_num)
        new_im = inst.balanceWhite(im)
    elif algo=="GT":
        gains = gt_illuminant / min(gt_illuminant)
        g1 = float(1.0 / gains[2])
        g2 = float(1.0 / gains[1])
        g3 = float(1.0 / gains[0])
        new_im = cv.xphoto.applyChannelGains(im, g1, g2, g3)
    time = 1000*(timeit.default_timer() - start_time) #time in ms

    if len(dst_folder)>0:
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)
        im_name = ("%04d_" % i) + algo.replace(":","_") + ".jpg"
        cv.imwrite(os.path.join(dst_folder, im_name), stretch_to_8bit(new_im))

    #recover the illuminant from the color balancing result, assuming the standard model:
    estimated_illuminant = [0, 0, 0]
    eps = 0.01
    estimated_illuminant[2] = np.percentile((im[:,:,0] + eps) / (new_im[:,:,0] + eps), 50)
    estimated_illuminant[1] = np.percentile((im[:,:,1] + eps) / (new_im[:,:,1] + eps), 50)
    estimated_illuminant[0] = np.percentile((im[:,:,2] + eps) / (new_im[:,:,2] + eps), 50)

    res = np.arccos(np.dot(gt_illuminant,estimated_illuminant)/
                   (np.linalg.norm(gt_illuminant) * np.linalg.norm(estimated_illuminant)))
    return (time, (res / np.pi) * 180)


def build_html_table(out, state, stat_list, img_range):
    stat_dict = {'mean': ('Mean error', lambda arr: np.mean(arr)),
                 'median': ('Median error',lambda arr: np.percentile(arr, 50)),
                 'p05': ('5<sup>th</sup> percentile',lambda arr: np.percentile(arr, 5)),
                 'p20': ('20<sup>th</sup> percentile',lambda arr: np.percentile(arr, 20)),
                 'p80': ('80<sup>th</sup> percentile',lambda arr: np.percentile(arr, 80)),
                 'p95': ('95<sup>th</sup> percentile',lambda arr: np.percentile(arr, 95))
                }
    html_out = ['<style type="text/css">\n',
                '  html, body {font-family: Lucida Console, Courier New, Courier;font-size: 16px;color:#3e4758;}\n',
                '  .tbl{background:none repeat scroll 0 0 #FFFFFF;border-collapse:collapse;font-family:"Lucida Sans Unicode","Lucida Grande",Sans-Serif;font-size:14px;margin:20px;text-align:left;width:480px;margin-left: auto;margin-right: auto;white-space:nowrap;}\n',
                '  .tbl span{display:block;white-space:nowrap;}\n',
                '  .tbl thead tr:last-child th {padding-bottom:5px;}\n',
                '  .tbl tbody tr:first-child td {border-top:3px solid #6678B1;}\n',
                '  .tbl th{border:none;color:#003399;font-size:16px;font-weight:normal;white-space:nowrap;padding:3px 10px;}\n',
                '  .tbl td{border:none;border-bottom:1px solid #CCCCCC;color:#666699;padding:6px 8px;white-space:nowrap;}\n',
                '  .tbl tbody tr:hover td{color:#000099;}\n',
                '  .tbl caption{font:italic 16px "Trebuchet MS",Verdana,Arial,Helvetica,sans-serif;padding:0 0 5px;text-align:right;white-space:normal;}\n',
                '  .firstingroup {border-top:2px solid #6678B1;}\n',
                '</style>\n\n']

    html_out += ['<table class="tbl">\n',
                 '  <thead>\n',
                 '    <tr>\n',
                 '      <th align="center" valign="top"> Algorithm Name </th>\n',
                 '      <th align="center" valign="top"> Average Time </th>\n']
    for stat in stat_list:
        if stat not in stat_dict.keys():
            print("Error: unsupported statistic " + stat)
            sys.exit(1)
        html_out += ['      <th align="center" valign="top"> ' +
                             stat_dict[stat][0] +
                          ' </th>\n']
    html_out += ['    </tr>\n',
                 '  </thead>\n',
                 '  <tbody>\n']

    for algorithm in state.keys():
        arr = [state[algorithm][file]["angular_error"] for file in state[algorithm].keys() if file>=img_range[0] and file<=img_range[1]]
        average_time = "%.2f ms" % np.mean([state[algorithm][file]["time"] for file in state[algorithm].keys()
                                                                           if file>=img_range[0] and file<=img_range[1]])
        html_out += ['    <tr>\n',
                     '      <td>' + algorithm + '</td>\n',
                     '      <td>' + average_time + '</td>\n']
        for stat in stat_list:
            html_out += ['      <td> ' +
                                 "%.2f&deg" % stat_dict[stat][1](arr) +
                             ' </td>\n']
        html_out += ['    </tr>\n']
    html_out += ['  </tbody>\n',
                 '</table>\n']
    f = open(out, 'w')
    f.writelines(html_out)
    f.close()

cur_path = os.path.join("./", "z.jpg")
im = cv.imread(cur_path, -1).astype(np.float32)
#im -= black_levels[i]
range_thresh = 255
im = stretch_to_8bit(im)

model_path = "./color_balance_model.yml"
inst = cv.xphoto.createLearningBasedWB(model_path)
inst.setRangeMaxVal(range_thresh)
inst.setSaturationThreshold(0.98)
inst.setHistBinNum(64)
new_im = inst.balanceWhite(im)


cv.imwrite(os.path.join("./", "zz.jpg"), stretch_to_8bit(new_im))
#new_im.save(".\zz.jpg")