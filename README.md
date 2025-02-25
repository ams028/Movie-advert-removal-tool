Tool for automatic removal of advertisments from recorded TV programmes.

adve_cut.py --help
usage: adve_cut.py [-h] [-i I] [-fast] [-hevc] [-execute] [-min_ok_time MIN_OK_TIME] [-min_bad_time MIN_BAD_TIME]
                   [-delogo] [-remlogo] [-stdhistbin STDHISTBIN] [-avghistbin AVGHISTBIN] [-t T] [-skip_phase1]
                   [-skip_phase2] [-rendebug] [-maskdebug] [-searchtreedebug] [-hystdebug]

Movie advert removal tool.

options:
  -h, --help            show this help message and exit
  -i I                  Path to input .mpeg/.mp4 movie
  -fast                 Movie cutting mode
  -hevc                 Movie cutting mode
  -execute              Execute movie cutting
  -min_ok_time MIN_OK_TIME
                        Minimal movie time detection [s]
  -min_bad_time MIN_BAD_TIME
                        Minimal movie time detection [s]
  -delogo               Remove station logo from input file. Method#1 - rectangle
  -remlogo              Remove station logo from input file. Method#2 - accurate
  -stdhistbin STDHISTBIN
                        Logo detection average image percentile [0..19]
  -avghistbin AVGHISTBIN
                        Logo detection standard deviation image percentile [0..19]
  -t T                  Path to directory from previous execution to be reused
  -skip_phase1          Skipping phase 1 - calculation of average image and standard deviation image.
  -skip_phase2          Skipping phase 2 - calculation of logo/edge masks.
  -rendebug             Enable debug functionality of renaming thumbnails from thumb*.* to thumB*.* for advert
                        thumbnails
  -maskdebug            Enable debug mask generation
  -searchtreedebug      Enable debug of search tree
  -hystdebug            Enable debug for movie time detection
  
  
#Examples .bat files:

#Max comprssion, generate covertion script only - recommended:
for %%a in ("*.mpeg") do (python adve_cut.py -i %%a -hevc -rendebug) 

#Normal compression, generate conversion script and execute immediately
for %%a in ("*.mpeg") do (python adve_cut.py -i %%a -execute) 

#Fastest, no compression, generate conversion script only
for %%a in ("*.mpeg") do (python adve_cut.py -i %%a -fast -rendebug) 
