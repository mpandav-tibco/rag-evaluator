#!/bin/sh
cd /Users/milindpandav/git/rageval
nohup /Users/milindpandav/git/rageval/rageval >rageval-svc.log 2>&1 &
printf 'rageval started, PID=%s\n' "$!"
