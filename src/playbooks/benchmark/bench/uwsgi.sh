#!/usr/bin/env bash

set -e

while getopts "ds" arg; do
	case $arg in
		d)
			DAEMONIZE=1
		;;
		s)
			STOP=1
		;;
	esac
done


if [ $DAEMONIZE ]; then
	EXTRA_ARGS="--safe-pidfile /tmp/prefetch-server.pid --daemonize prefetch-server.log"
fi

cd $(dirname ${BASH_SOURCE[0]})/prefetch-server

# This was specific to our environment
# source /home/mqp/benchmarking/torch-reads/torch/bin/activate

if [ $STOP ]; then
	uwsgi --stop /tmp/prefetch-server.pid
	exit 0
fi

exec uwsgi --http-socket /tmp/prefetch-server.sock --wsgi-file prefetch-server.py --master --callable app $EXTRA_ARGS
