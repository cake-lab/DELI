#!/usr/bin/env bash

tmux \
	new-session -d "bash -c './bench.sh $*; exec bash'" \; \
	new-window -d 'bash -c "./uwsgi.sh; exec bash"' \;
