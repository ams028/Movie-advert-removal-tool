import sys

import numpy as np
from numpy import uint8 as uint8
from scipy import ndimage
import glob
import os.path
import os
import cv2
import subprocess
import re
import datetime
import argparse
import tempfile
import enum

class ConvertMode:
   FAST = 1
   FINE = 2
   HEVC = 3

tmpdir = os.path.join(os.getcwd(),"")

def logger(txt):    
  print(txt)
  print(txt, file=open(tmpdir + '_log.txt', 'a'))

def timestr2fractsec(str):
    h,m,s,fract = int(re.split("[:.]", str))
    h,m,s,fract = int(h), int(m), int(s), int(fract)
    return fract + s * 100 + m * 100 * 60 + h * 100 * 60 * 60

def fname2fractsec(str):   
    ar = re.split("[_.bB]+", os.path.split(str)[-1])
    if len(ar) == 4:
        pref,hms,fract,suffix = ar
        hms = int(hms)
        h = int(hms/10000)
        hms -= h * 10000
        m = int(hms/100)
        hms -= m * 100
        s = int(hms)
        fract = int(fract)
        return fract + s * 100 + m * 100 * 60 + h * 100 * 60 * 60
    else:
        if len(ar) == 3:
            ar = pref,m,suffix = ar
            m = int(m)
            return m * 100 * 60
        else:
            return 0               

def fracsec2timestr(fract, ffmpeg = True):
    h = (int) (fract / (100 * 60 * 60))
    fract -= h * (100 * 60 * 60)
    m = (int) (fract / (100 * 60))
    fract -= m * (100 * 60)
    s = (int) (fract / 100)
    fract -= s * (100)
    if ffmpeg:
        return "{:02d}:{:02d}:{:02d}.{:02d}".format(h,m,s,fract)
    else:
        return "{:02d}{:02d}{:02d}_{:02d}".format(h,m,s,fract)

def getmoviefend(moviein):
    cmdar = ["ffprobe", "-v", "panic", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", moviein]
    if os.path.isfile(moviein):
        retval = subprocess.run(cmdar, capture_output = True, text = True)
        if "N/A" in retval.stdout:
            return 5*3600*100
        else:
            return int(float(retval.stdout) * 100)
    else:
        return 0    

def movie2png(moviein, timefstart = 0, timefend = -1, delta = 60*100):
    imgtimes = []
    imgfiles = []
    eof = False
    last = False
    if timefend < 0:
        timefend = getmoviefend(moviein)
    while timefstart <= timefend:
        timestart = fracsec2timestr(timefstart)
        pngout = tmpdir + "thumb_{:s}.png".format(fracsec2timestr(timefstart, False))
        logger("%7d %7d %s %s"%(timefstart, timefend, timestart, os.path.split(pngout)[-1]))
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "panic", "-ss", timestart, "-i", moviein, "-update", "1", "-frames:v", "1", pngout])
        if os.path.isfile(pngout):
            imgfiles.append(pngout)
            imgtimes.append(timefstart)
        else:
            break;
        if timefstart + delta <= timefend or timefstart == timefend:
            timefstart += delta
        else:
            timefstart = timefend # last extra frame if not aligned delta
    return imgfiles, imgtimes

def dump_single_ok_range(inputmovie, ok_start_time, ok_end_time, part, mode, delogofilter, execute):
    prefix = os.path.splitext(os.path.split(inputmovie)[-1])[-2]
    outputmovie = prefix + "_p{:02d}.mp4".format(part)
    outfilelist = prefix + ".txt"
    outbatfile = prefix + ".bat"
    fileWriteMode = "a" if part > 0 else "w"
    cmd = ""
    cmdar1 = ["ffmpeg", "-hide_banner", "-ss", fracsec2timestr(ok_start_time), "-to", fracsec2timestr(ok_end_time), "-i", inputmovie] 
    cmdar2 = []
    if mode == ConvertMode.FINE:
        cmdar2 = ["-c:v", "libx264", "-c:a", "aac", outputmovie]
    elif mode == ConvertMode.HEVC:
        cmdar2 = ["-c:v", "libx265", "-c:a", "aac", outputmovie]
    elif mode == ConvertMode.FAST:
        cmdar2 = ["-c", "copy", outputmovie]
    cmdar = cmdar1 + delogofilter + cmdar2
    cmd = " ".join(cmdar)
    logger(cmd)
    print(cmd, file=open(outbatfile, fileWriteMode))
    print("file ", outputmovie, file=open(outfilelist, fileWriteMode))
    if execute:
        subprocess.run(cmdar)
        
def dump_concat_command(inputmovie, part, execute):
    if part > 1:
        prefix = os.path.splitext(os.path.split(inputmovie)[-1])[-2]
        movie_part0 = prefix + "_p00.mp4"
        outputmovie = prefix + "_out.mp4"
        outfilelist = prefix + ".txt"
        outbatfile = prefix + ".bat"
        cmdar = []
        cmdar = ["ffmpeg", "-hide_banner", "-f", "concat", "-i", outfilelist, "-c", "copy", outputmovie]
        cmd = " ".join(cmdar)        
        logger(cmd)
        print(cmd, file=open(outbatfile, "a"))
        if execute:
            if os.path.isfile(outputmovie):
                os.remove(outputmovie)
            subprocess.run(cmdar)

def to_grayscale(img):
    "If arr is a color image (3D array), convert it to grayscale (2D array)."
    if len(img.shape) == 3:
        grayarr = cv2.cvtColor(img.astype(uint8), cv2.COLOR_BGR2GRAY)
        return grayarr
    else:
        return img

def to_edges(img):   
    edges = cv2.Canny(img.astype(uint8), 100, 125)   
    return edges

def locate_roi_img_area(img):
    h,w, ch = img.shape    
    mask = np.zeros(shape = (h, w), dtype=uint8)
    margin = 10
    mask = cv2.rectangle(mask, (margin, margin), (w-margin, h-margin), 1, -1)
    return mask

def delogo_command():    
    l_mask_logo = (to_grayscale(cv2.imread(tmpdir + "_mask_logo_l.png")) / 255).astype(uint8)
    ((minX, minY), (maxX, maxY)) = mask_img_ranges(l_mask_logo, margin = 2)
    cmdL = "delogo=x={:d}:y={:d}:w={:d}:h={:d}".format(minX,minY,maxX-minX,maxY-minY)
    r_mask_logo = (to_grayscale(cv2.imread(tmpdir + "_mask_logo_r.png")) / 255).astype(uint8)
    ((minX, minY), (maxX, maxY)) = mask_img_ranges(r_mask_logo, margin = 2)
    cmdR = "delogo=x={:d}:y={:d}:w={:d}:h={:d}".format(minX,minY,maxX-minX,maxY-minY)
    return ["-vf", '"{:s}, {:s}"'.format(cmdL, cmdR)]
    
def remlogo_command():
    return ["-vf", '"removelogo={:s}"'.format(os.path.join(tmpdir, "_mask_logo.png").replace('\\','/'))]

def parse_args():
    parser = argparse.ArgumentParser(description='Movie advert removal tool.')
    parser.add_argument('-i', help='Path to input .mpeg/.mp4 movie', default=None)
    parser.add_argument('-fast', help='Movie cutting mode', action="store_true")
    parser.add_argument('-hevc', help='Movie cutting mode', action="store_true")
    parser.add_argument('-execute', help='Execute movie cutting', action="store_true")
    parser.add_argument('-min_ok_time', help='Minimal movie time detection [s]', type=int, default='120')
    parser.add_argument('-min_bad_time', help='Minimal movie time detection [s]', type=int, default='1')
    parser.add_argument('-delogo', help='Remove station logo from input file. Method#1 - rectangle', action="store_true")
    parser.add_argument('-remlogo', help='Remove station logo from input file. Method#2 - accurate', action="store_true")
    parser.add_argument('-stdhistbin', help='Logo detection average image percentile [0..19]', type=int, default='-1')
    parser.add_argument('-avghistbin', help='Logo detection standard deviation image percentile [0..19]', type=int, default='-1')
    parser.add_argument('-t', help='Path to directory from previous execution to be reused', default=None)
    parser.add_argument('-skip_phase1', help='Skipping phase 1 - calculation of average image and standard deviation image.', action="store_true")
    parser.add_argument('-skip_phase2', help='Skipping phase 2 - calculation of logo/edge masks.', action="store_true")
    parser.add_argument('-rendebug', help='Enable debug functionality of renaming thumbnails from thumb*.* to thumB*.* for advert thumbnails', action="store_true")
    parser.add_argument('-maskdebug', help='Enable debug mask generation', action="store_true")
    parser.add_argument('-searchtreedebug', help='Enable debug of search tree', action="store_true")
    parser.add_argument('-hystdebug', help='Enable debug for movie time detection', action="store_true")
    args = parser.parse_args()
    if (args.delogo and args.fast) or (args.delogo and args.remlogo) or (args.remlogo and args.fast):
        print("Options: -delogo -remlogo -fast cannot work together")
        exit(1)
    if (args.i != None):
        if "mpeg" in args.i or "mp4" in args.i:
            return (args.i, args)
        else:
            print("Only .mp4 and .mpeg movies supported.")
            exit(1)
    else:
        print("Please provide input movie for adverts removal option -i")
        exit(1)

def learn_avg_std_img(imgfiles, skip_phase):
    logger("learn_avg_std_img skip_phase1=%d" % (skip_phase))

    if skip_phase and os.path.isfile(tmpdir + "_avg_image_color.png") and os.path.isfile(tmpdir + "_std_image.png"):
        logger("-- LEARNING AVG of all images - skipped")
        avg_image_color = cv2.imread(tmpdir + "_avg_image_color.png").astype(float)
        avg_image = to_grayscale(avg_image_color).astype(float)
        std_image = to_grayscale(cv2.imread(tmpdir + "_std_image.png")).astype(float)
    else:
        logger("-- LEARNING AVG of all images")
        if len(imgfiles) == 0:
            logger("No files to analyze STD AVG")
            exit(1)
        sum_image_color = np.empty_like(cv2.imread(imgfiles[0]).astype(float))
        for file in imgfiles:
            im_color = cv2.imread(file).astype(float)
            cv2.add(sum_image_color, im_color, sum_image_color)
        avg_image_color = (sum_image_color / len(imgfiles))
        avg_image = to_grayscale(avg_image_color).astype(float)
        cv2.imwrite(tmpdir + "_avg_image_color.png", avg_image_color)

        logger("-- LEARNING STD of all images")
        var_image = diff = np.zeros_like(avg_image)
        for file in imgfiles:         
            im_color = cv2.imread(file).astype(float)
            im = to_grayscale(im_color).astype(float)
            cv2.subtract(im, avg_image, diff)
            diff = diff * diff
            var_image += diff            
        std_image = (np.sqrt(var_image / len(imgfiles)))   
        cv2.imwrite(tmpdir + "_std_image.png", (std_image).astype(uint8))
    return(avg_image, avg_image_color, std_image)

def print_histogram(histogram, heading):
    logger(heading)    
    counts, bins = histogram
    logger(counts)
    logger(bins)

def rmtmpfiles(imgfiles):
    for f in imgfiles:
        if os.path.isfile(f.replace(".png", "_e.png")):
            os.remove(f.replace(".png", "_e.png"))
        if os.path.isfile(f.replace(".png", "_de.png")):
            os.remove(f.replace(".png", "_de.png"))
        if os.path.isfile(f.replace(".png", "_p.png")):
            os.remove(f.replace(".png", "_p.png"))

def mask_img_ranges(img, margin = 0):
    h,w = img.shape
    idxs = np.argwhere(img != 0)
    minX = max(0, np.min(idxs[:,1]) - margin)
    minY = max(0, np.min(idxs[:,0]) - margin)
    maxX = min(w, np.max(idxs[:,1]) + margin)
    maxY = min(h, np.max(idxs[:,0]) + margin)
    return ((minX, minY), (maxX, maxY))
    
def get_rectange_mask(inputmask, margin):
    mask_out = np.zeros_like(inputmask)
    if cv2.countNonZero(inputmask) > 0:    
        ((minX, minY), (maxX, maxY)) = mask_img_ranges(inputmask, margin);
        logger("Mask edges in range: (%d,%d)..(%d,%d)" % (minX, minY, maxX, maxY))
        cv2.rectangle(mask_out, (minX, minY), (maxX+1, maxY+1), 1, -1)
    return mask_out

def calc_threshold(prefix, subr, histbin=-1):
    (_hist_cnt, _hist_rng) = _hist = np.histogram(subr.astype(uint8).ravel(), bins=20)
    _threshold = _hist_rng[histbin]             
    logger("\n  %s BIN: %5d => THR: %5d" % (prefix, histbin, _threshold))
    print_histogram(_hist, "  "+prefix+" HISTOGRAM")
    return (_threshold)

def calc_thresholds(avg_image, std_image, avg_histbin=-1, std_histbin=-1):
    h,w = avg_image.shape
    half = w//2
    if avg_histbin == -1: 
        avg_histbin = 13   # STD: 8..14th out of 21 bins (~66 working), 8th for mecz, 11th for tvn 
    if std_histbin == -1:
        std_histbin = 15   # AVG: 13th out of 21 bits (~145 working) # 13th stare (mecz), 11th for pegi18 (red)
        
    l_threshold_avg = calc_threshold("L AVG", avg_image[:,:half], avg_histbin)
    r_threshold_avg = calc_threshold("R AVG", avg_image[:,half:], avg_histbin)
    l_threshold_std = calc_threshold("L STD", std_image[:,:half], std_histbin)
    r_threshold_std = calc_threshold("R STD", std_image[:,half:], std_histbin)
    return (l_threshold_avg, r_threshold_avg, l_threshold_std, r_threshold_std)

def calc_treshold_mask(img, trhL, thrR, mode):
    h,w = img.shape
    half = w//2
    ret,maskL = cv2.threshold(img[:,:half], trhL, 1, mode)
    ret,maskR = cv2.threshold(img[:,half:], thrR, 1, mode)
    return cv2.hconcat([maskL, maskR]).astype(uint8)
      
def learn_masks(avg_image, avg_image_color, std_image, skip_phase, avg_histbin=-1, std_histbin=-1):       
    if skip_phase and os.path.isfile(tmpdir + "_mask_edge.png") and os.path.isfile(tmpdir + "_mask_final.png"):
        logger("-- LEARNING MASKS of all images - skipped")
        mask_edge = (to_grayscale(cv2.imread(tmpdir + "_mask_edge.png")) / 255).astype(uint8)
        mask_final = (to_grayscale(cv2.imread(tmpdir + "_mask_final.png")) / 255).astype(uint8)
    else:
        logger("-- LEARNING MASKS of all images")
        
        h,w = avg_image.shape
        half = w//2
        (l_threshold_avg, r_threshold_avg, l_threshold_std, r_threshold_std) = calc_thresholds(avg_image, std_image, avg_histbin, std_histbin)
        mask_avg = calc_treshold_mask(avg_image, l_threshold_avg, r_threshold_avg, cv2.THRESH_BINARY)
        mask_std = calc_treshold_mask(std_image, l_threshold_std, r_threshold_std, cv2.THRESH_BINARY_INV)        
        mask_roi = locate_roi_img_area(avg_image_color)
        mask_logo = cv2.bitwise_and(mask_std, mask_avg, mask_roi)
        mask_final = cv2.hconcat([get_rectange_mask(mask_logo[:,:half], 3), get_rectange_mask(mask_logo[:,half:], 3)]).astype(uint8)
        mask_edge = to_edges(avg_image)
        mask_edge = cv2.bitwise_and(mask_edge, mask_final, mask_edge)

        cv2.imwrite(tmpdir + "_mask_logo.png", (mask_logo * 255).astype(uint8))
        cv2.imwrite(tmpdir + "_mask_edge.png", (mask_edge * 255).astype(uint8))    
        cv2.imwrite(tmpdir + "_mask_final.png", (mask_final * 255).astype(uint8))
        cv2.imwrite(tmpdir + "_mask_std.png", (mask_std * 255).astype(uint8))
        cv2.imwrite(tmpdir + "_mask_avg.png", (mask_avg * 255).astype(uint8))
        
    return (mask_edge, mask_final)

def evaluate_files(imgfiles, skip_phase1, skip_phase2, std_histbin, avg_histbin, rendebug, maskdebug):
    rmtmpfiles(imgfiles)
    logger("evaluate_files skip_phase1=%d skip_phase2=%d" % (skip_phase1, skip_phase2))

    avg_image, avg_image_color, std_image = learn_avg_std_img(imgfiles, skip_phase1)
    (mask_edge, mask_final) = learn_masks(avg_image, avg_image_color, std_image, skip_phase2, avg_histbin, std_histbin)
 
    h,w = avg_image.shape
    half = w//2 
    edge_thr_L = cv2.countNonZero(mask_edge[:,:half]) * 0.6
    edge_thr_R = cv2.countNonZero(mask_edge[:,half:]) * 0.6
    
    fmt = "%-15s %6.0f %6.0f %1s %ls"   
    logger(fmt % ("== THRESHOLD ==", edge_thr_L, edge_thr_R, "", "\n==============="))

    imgs_okay = []
    for file in imgfiles:
        imcolor = cv2.imread(file)
        im_edg = to_edges(imcolor)   
        cv2.bitwise_and(im_edg, im_edg, mask=mask_final)
        diffe=np.zeros_like(mask_edge)
        cv2.subtract(mask_edge, im_edg, diffe, mask_edge)       
        sumL = cv2.countNonZero(diffe[:,:half])
        sumR = cv2.countNonZero(diffe[:,half:])
        edge_ind = " " if sumL < edge_thr_L or sumR < edge_thr_R else "x"
        imgs_okay.append(edge_ind)
        if rendebug: 
            if edge_ind == "x":
                newfname = file.replace("thumb","thumB")
            else:
                newfname = file.replace("thumB","thumb")
            os.rename(file, newfname)
            file = newfname
            bad_ind = " "
        else:
            if ("thumB" in file):        
                bad_ind = "M" if (edge_ind == " ") else " "
            else:
                bad_ind = "F" if (edge_ind == "x") else " "

        if (edge_ind != " " and maskdebug):
            cv2.imwrite(file.replace(".png", "_e.png"), (im_edg * 128).astype(uint8))
            cv2.imwrite(file.replace(".png", "_de.png"), (diffe * 128).astype(uint8))

        logger(fmt % (os.path.split(file)[-1], sumL, sumR, edge_ind, bad_ind))
        
    return imgs_okay

def get_convert_mode(fastMode, hevcMode):
    convertMode = ConvertMode.FINE
    if hevcMode:
        convertMode = ConvertMode.HEVC
    elif fastMode:
        convertMode = ConvertMode.FA
    return convertMode

def dump_merged_ok_ranges(imgs_okay, imgtimes, inputmovie, min_ok_time, min_bad_time, hystdebug, fastMode, hevcMode, delogo, remlogo, execute):
    convertMode = get_convert_mode(fastMode, hevcMode)
    ok_start_time = ok_end_time = -1
    bad_start_time = imgtimes[0] if imgs_okay[0] == "x" else -2000*100
    bad_end_time = imgtimes[0] if imgs_okay[0] == "x" else -1
    part = 0
    inMovie = False
    
    delogofilter = []
    if delogo:
        delogofilter = delogo_command()
    elif remlogo:
        delogofilter = remlogo_command()

    for i in range(1, len(imgs_okay)):   
        if hystdebug:
            logger("[dump] i=%3d inMovie=%d part=%2d okay=[%s..%s]"%(i, inMovie, part, imgs_okay[i-1], imgs_okay[i]))
            logger("[dump]   time=   [%s..%s]"%(fracsec2timestr(imgtimes[i-1]), fracsec2timestr(imgtimes[i])))
            logger("[dump]   oktime= [%s..%s] diff=%d"%(fracsec2timestr(ok_start_time), fracsec2timestr(ok_end_time), ok_end_time-ok_start_time))
            logger("[dump]   badtime=[%s..%s] diff=%d"%(fracsec2timestr(bad_start_time), fracsec2timestr(bad_end_time), bad_end_time-bad_start_time))
        if imgs_okay[i-1] == imgs_okay[i]:
            if imgs_okay[i] == " ":
                # OK -> OK
                ok_end_time = imgtimes[i]
                if ok_start_time < 0:
                    ok_start_time = imgtimes[i-1]
                if ok_end_time - ok_start_time >= min_ok_time * 100:
                    inMovie = True 
            else:
                # BAD -> BAD
                bad_end_time = imgtimes[i]
                if bad_start_time < 0:
                    bad_start_time = imgtimes[i-1]
                if (bad_end_time - bad_start_time > min_bad_time * 100 and inMovie):
                    dump_single_ok_range(inputmovie, ok_start_time, ok_end_time, part, convertMode, delogofilter, execute)
                    part += 1
                    inMovie = False
        else: 
            if imgs_okay[i] == "x":
                # OK -> BAD
                bad_start_time = bad_end_time = imgtimes[i]                                 
                ok_end_time = imgtimes[i-1]
                if ok_start_time >= 0 and ok_end_time - ok_start_time > min_ok_time * 100:
                    inMovie = True # till now inMovie for sure
                else:
                    ok_start_time = ok_end_time -1 # too short, drop range
            else:
                # BAD -> OK
                bad_end_time = imgtimes[i-1]
                if (bad_end_time - bad_start_time > min_bad_time * 100 and inMovie):
                    dump_single_ok_range(inputmovie, ok_start_time, ok_end_time, part, convertMode, delogofilter, execute)
                    part += 1
                    inMovie = False # not sure yet
                if not inMovie:
                    ok_start_time = imgtimes[i]
                ok_end_time = imgtimes[i]
                bad_start_time = bad_end_time = -1
    if inMovie:
        dump_single_ok_range(inputmovie, ok_start_time, ok_end_time, part, convertMode, delogofilter, execute)
        part += 1
    dump_concat_command(inputmovie, part, execute)

def dump3tables(heading, imgfiles, imgs_okay, imgtimes):
    logger(heading)
    if len(imgtimes) == 0:
        return
    t_prev = imgtimes[0] - 1
    for i in range(len(imgfiles)):
        error = " " if imgtimes[i] > t_prev else "E"
        logger("%20s %2s %d %s"%(os.path.split(imgfiles[i])[-1], imgs_okay[i], imgtimes[i], error))
        t_prev = imgtimes[i]


def main():
    nextdelta = { 6000:500, 500:100, 100:4, 4:0, 0:0 } 
    timedelta = 0
    imgtimes = []
    
    # ct stores current time
    ct1 = datetime.datetime.now()
    moviefile, args = parse_args()

    global tmpdir
    if args.t == None:
        deletetmp = not (args.maskdebug or args.rendebug or args.searchtreedebug or args.hystdebug)
        prefix = os.path.join(os.getcwd(), "") + os.path.splitext(os.path.split(moviefile)[-1])[-2] + "_"
        tmpdirobj = tempfile.TemporaryDirectory(prefix=prefix, delete=deletetmp)
        tmpdir = os.path.join(tmpdirobj.name, "")
    else:
        tmpdir = os.path.join(args.t, "")

    logger("START === %s ===========================" % ct1)
    if args.t == None:
        timedelta = 6000 # 1 minute
        imgfiles, imgtimes = movie2png(moviefile, delta = timedelta)
    else:
        imgfiles = glob.glob(os.path.join(tmpdir, "thumb_??????_??.png"))
        if len(imgfiles) == 0:
            logger("No files to analyze")
            exit(1)        
        for i in imgfiles:
            imgtimes.append(fname2fractsec(i))

    imgs_okay = evaluate_files(imgfiles, args.skip_phase1, args.skip_phase2, args.stdhistbin, args.avghistbin, args.rendebug, args.maskdebug)
    if len(imgs_okay) >= 1: 
        imgs_okay.append("x") # artificial end - "sentinel"
        imgtimes.append(imgtimes[-1] + timedelta)
        imgfiles.append("_nosuchfile_.png")
        
    while timedelta > 0 and args.t == None:
        prevdelta = timedelta        
        imgtimes[-1] = imgtimes[-2] + timedelta # update end time of sentinel to speed up
        timedelta = nextdelta.get(timedelta)
        
        logger("== DELTA == %f"%timedelta)
        i = 0
        while i < len(imgfiles)-1 and timedelta > 0:            
            if imgs_okay[i] != imgs_okay[i+1]:
                time1 = imgtimes[i] + timedelta
                time2 = imgtimes[i+1] - timedelta
                if (time2 - time1 > prevdelta):
                    logger("ERROR: i=%d time %d-%d > %d prevdelta (delta = %d)" % (i, time2, time1, prevdelta, timedelta))
                    if args.searchtreedebug:
                        dump3tables("ALL with ERROR", imgfiles, imgs_okay, imgtimes)
                    return 
                new_imgfiles, new_imgtimes = movie2png(moviefile, timefstart = time1, timefend = time2, delta = timedelta)
                new_imgs_okay = evaluate_files(new_imgfiles, True, True, args.stdhistbin, args.avghistbin, args.rendebug, args.maskdebug)                                
                imgfiles = imgfiles[0:i+1] + new_imgfiles + imgfiles[i+1:]
                imgs_okay = imgs_okay[0:i+1] + new_imgs_okay + imgs_okay[i+1:]
                imgtimes = imgtimes[0:i+1] + new_imgtimes + imgtimes[i+1:]
                if args.searchtreedebug:
                    dump3tables("NEW", new_imgfiles, new_imgs_okay, new_imgtimes)
                    dump3tables("ALL", imgfiles, imgs_okay, imgtimes)                                
                i += len(new_imgs_okay) + 1
            else:
                i += 1    
       
    dump_merged_ok_ranges(imgs_okay, imgtimes, moviefile, args.min_ok_time, args.min_bad_time, args.hystdebug, args.fast, args.hevc, args.delogo, args.remlogo, args.execute)
    ct2 = datetime.datetime.now()
    logger("END   === %s =========================== ELAPSED: %s" % (ct2, ct2-ct1))

# Run the main function:

if __name__ == "__main__":
    main()
