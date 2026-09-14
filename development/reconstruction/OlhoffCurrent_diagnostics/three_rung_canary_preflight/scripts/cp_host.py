#!/usr/bin/env python3
"""cp_host.py -- Part G host record, and the MATLAB availability probe."""
import json, socket, subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).parents[1]


def sh(c, t=20):
    try:
        r = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t)
        return r.stdout.strip()
    except Exception as e:
        return f'<{type(e).__name__}>'


def port(host, p, t=6):
    try:
        with socket.create_connection((host, p), timeout=t):
            return 'OPEN'
    except Exception as e:
        return f'{type(e).__name__}'


h = {
    'when': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    'host': {
        'hostname': sh('hostname'),
        'cpu': sh('sysctl -n machdep.cpu.brand_string'),
        'ncpu': int(sh('sysctl -n hw.ncpu') or 0),
        'physicalcpu': int(sh('sysctl -n hw.physicalcpu') or 0),
        'ram_bytes': int(sh('sysctl -n hw.memsize') or 0),
        'os': sh('sw_vers -productVersion'),
        'build': sh('sw_vers -buildVersion'),
        'kernel': sh('uname -r'),
        'arch': sh('uname -m'),
    },
    'load': {
        'loadavg': sh('sysctl -n vm.loadavg'),
        'uptime': sh('uptime'),
        'swapusage': sh('sysctl -n vm.swapusage'),
        'vm_stat_head': sh('vm_stat | head -6'),
        'matlab_processes': sh('ps -eo pid,pcpu,comm | grep -ci "[M]ATLAB"'),
        'top_cpu': sh('ps -eo pcpu,comm -r | head -6'),
    },
    'thread_env': {k: sh(f'printenv {k}') for k in
                   ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                    'OPENBLAS_NUM_THREADS')},
    'matlab': {
        'probe_note': ('runtime_version is obtained by actually starting MATLAB; '
                       'a null value means MATLAB did not start.'),
        'app': '/Applications/MATLAB_R2025b.app',
        'installed': Path('/Applications/MATLAB_R2025b.app/bin/matlab').exists(),
        'on_PATH': sh('command -v matlab') or None,
        'runtime_version': sh('/Applications/MATLAB_R2025b.app/bin/matlab -batch '
                              '"fprintf(\'%s\', version)"', 400) or None,
        'license_files': sh('ls /Applications/MATLAB_R2025b.app/licenses/'),
        'network_lic': sh('cat /Applications/MATLAB_R2025b.app/licenses/network.lic'),
        'user_license_dir_exists': Path(
            '/Users/piotrek/Library/Application Support/MathWorks/MATLAB/R2025b_licenses').exists(),
    },
    'license_server_reachability': {
        'fqdn': 'zm8pc.ippt.pan.pl',
        'resolved': sh("python3 -c \"import socket;print(socket.gethostbyname('zm8pc.ippt.pan.pl'))\"", 20),
        'tcp_27000': port('148.81.54.174', 27000),
        'tcp_27001': port('148.81.54.174', 27001),
        'tcp_27002': port('148.81.54.174', 27002),
        'general_internet': 'OK' if '0.0% packet loss' in sh('ping -c 2 -W 2000 8.8.8.8') else 'DEGRADED',
        'dns_resolver': sh("scutil --dns | awk '/nameserver\\[0\\]/{print $3; exit}'"),
        'interpretation': ('name resolves and general internet works, but the licence '
                           'manager ports are unreachable: this host is not on the '
                           'institute network or its VPN'),
    },
}
(HERE / 'evidence/host_environment.json').write_text(json.dumps(h, indent=1) + '\n')
print(json.dumps({'matlab': h['matlab']['runtime_version'],
                  'ports': {k: v for k, v in h['license_server_reachability'].items()
                            if k.startswith('tcp')},
                  'cpu': h['host']['cpu'], 'ram_GB': h['host']['ram_bytes'] / 2**30,
                  'loadavg': h['load']['loadavg']}, indent=1))
