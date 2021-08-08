#!/usr/bin/env bash

stop() {
	echo "Killing uwsgi..."
	ansible -i inventory.ini -a "~/ansible-run/uwsgi.sh -s" all
}

set -e
ansible-playbook -i inventory.ini copy.yml
ansible -i inventory.ini -a "~/ansible-run/uwsgi.sh -d" all
trap stop EXIT
while read run_args; do
	echo "Running with '$run_args'..."
	ansible-playbook -i inventory.ini copyrun.yml -e "just_bench=true $run_args test_name='${run_args// /:}'"
done < "${1:-/dev/stdin}"
