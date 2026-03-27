#!/bin/bash
pgrep -f signal_collect_v2.py || nohup python3.8 /home/rajmohan/amma-project/signal_collect_v2.py >> /home/rajmohan/amma-project/data/amma_data.txt 2>&1 & 
pgrep -f sentry.py || nohup python3.8 /home/rajmohan/amma-project/sentry.py >> /home/rajmohan/amma-project/data/alerts.txt 2>&1 &
