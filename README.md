# Tool for automatic removal of advertisments from recorded TV programmes.

## How it works:
The logo of the TV station and/or PEGI sign are detected as the most stable part of the picture within the movie and used to distiguish block of advertisments.
The parts of the movie are converted to mp4 using preinstalled ffmpeg and if more than 1 piece, merged into single .mp4 output file.

## Movie advert removal tool.
`adve_cut.py --help` \
usage: adve_cut.py [-h] [-i I] [-fast] [-hevc] [-execute] [-min_ok_time MIN_OK_TIME] [-min_bad_time MIN_BAD_TIME]
                   [-delogo] [-remlogo] [-stdhistbin STDHISTBIN] [-avghistbin AVGHISTBIN] [-t T] [-skip_phase1]
                   [-skip_phase2] [-rendebug] [-maskdebug] [-searchtreedebug] [-hystdebug]
### Options:
#### Main:
  `-h, --help`            show this help message and exit \
  `-i I`                  Path to input .mpeg/.mp4 movie \
  `-fast`                 Fast movie cutting mode \
  `-hevc`                 Best movie cutting mode \
  `-execute`              Execute movie cutting
#### Additional tuning/debug:
  `-min_ok_time MIN_OK_TIME` 
                        Minimal movie time detection [s] \
  `-min_bad_time MIN_BAD_TIME` 
                        Minimal movie time detection [s] \
  `-stdhistbin STDHISTBIN`
                        Logo detection average image percentile [0..19] \
  `-avghistbin AVGHISTBIN`
                        Logo detection standard deviation image percentile [0..19] \
  `-delogo`               Remove station logo from input file. Method#1 - rectangle \
  `-remlogo`              Remove station logo from input file. Method#2 - accurate \
  `-t T`                  Path to directory from previous execution to be reused \
  `-skip_phase1`          Skipping phase 1 - calculation of average image and standard deviation image. \
  `-skip_phase2`          Skipping phase 2 - calculation of logo/edge masks. \
  `-rendebug`             Enable debug functionality of renaming thumbnails from thumb*.* to thumB*.* for advert
                        thumbnails \
  `-maskdebug`            Enable debug mask generation \
  `-searchtreedebug`      Enable debug of search tree \
  `-hystdebug`            Enable debug for movie time detection

## Dependecies
- Python
- FFMPEG
  
## Examples .bat files:

### Max compression, generate covertion script only - recommended:
`for %%a in ("*.mpeg") do (python adve_cut.py -i %%a -hevc -rendebug)` 

### Normal compression, generate conversion script and execute immediately
`for %%a in ("*.mpeg") do (python adve_cut.py -i %%a -execute)` 

### Fastest, no compression, generate conversion script only
`for %%a in ("*.mpeg") do (python adve_cut.py -i %%a -fast -rendebug)` 
