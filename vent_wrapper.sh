#!/bin/sh
# Полная демонизация без setsid
python3 -c "
import os, sys, subprocess;
if os.fork() == 0:
    os.setsid()
    if os.fork() == 0:
        os.chdir('/opt/config/mod_data/plugins/vent')
        os.umask(0)
        sys.stdout = open('vent_calc.log', 'a')
        sys.stderr = sys.stdout
        subprocess.Popen([
            '/usr/bin/python3',
            '/opt/config/mod_data/plugins/vent/vent_calc.py'
        ])
"