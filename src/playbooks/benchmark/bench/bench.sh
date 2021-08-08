#!/usr/bin/env bash

# This is a hack but because the python script has -- flags, we can't use getopt in a loop becuase it gets sad.
# Thankfully we have only one flag, so we suppress the error messages for non -n flags.
if getopts "n:" arg 2>/dev/null; then
	case $arg in
		n)
			FILE_SUFFIX=$(printf "%s%q" - "$OPTARG")
			;;
	esac
	shift "$(( OPTIND - 1 ))"
fi

DATE=$(date +%s)

# This was specific to our environment
# source /home/mqp/benchmarking/torch-reads/torch/bin/activate

set -o xtrace
exec python3 -u bench.py $@ |tee "bench-output-$DATE$FILE_SUFFIX.txt"
