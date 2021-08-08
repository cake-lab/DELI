# Playbooks

For ease of benchmarking, we've made use of Ansible so that we can run commands on many nodes at once. Our playbooks will generally assume an inventory file that has the following hosts

```yml
[master]
master0 <ip>

[workers]
worker1 <ip>
worker2 <ip>
...
workern <ip>
```

Usage of these playbooks is not very exotic. Though more info can be found in the Ansible docs (which are quite detailed), the general usage of one of these playbooks is:

```
ansible-playbook -i inventory.ini playbook.yml
```

One may wish to set the `ANSIBLE_HOST_KEY_CHECKING=False` environment variable if 1) they trust the hosts they're connecting to 2) they're dealing with a GCP environment where IPs are floating and `known_hosts` ends up getting in the way.


## List of Playbooks
- `benchmark/copyrun.yml` This will copy the `bench` directory to the host, start up the necessary containers for a cache, and clear them. This does not do host provisioning and is specific to our hosts. Once the playbook finishes, the benchmarks will be running in a tmux pane on the target hosts.
- `killtmux.yml` Kills the tmux server on the given hosts. It will ask for confirmation before doing so, given that the author is quite error-prone himself :)
