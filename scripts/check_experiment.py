"""One-shot, read-only SSH check. Do not loop; a thread heartbeat invokes this."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


def fetch(config, run=subprocess.run):
    ssh = ['ssh', '-T', '-i', config['identity_file'], '-p', str(config['port']),
           '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes',
           '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=20',
           '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=1',
           config['user'] + '@' + config['host']]
    result = run(ssh, capture_output=True, text=True, encoding='utf-8', timeout=60)
    if result.returncode:
        raise RuntimeError('SSH observation failed (not evidence of job failure): ' + result.stderr.strip())
    data = json.loads(result.stdout)
    if data.get('schema') != 1 or data.get('pipeline') != config['pipeline']:
        raise ValueError('Wrong server response or experiment')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--ack', type=int, help='Acknowledge only after handling events successfully')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    cache = Path(config['cache_dir'])
    cache.mkdir(parents=True, exist_ok=True)
    cursor_path = cache / 'acknowledged.json'
    cursor = json.loads(cursor_path.read_text(encoding='utf-8')) if cursor_path.exists() else {}
    old = int(cursor.get(config['pipeline'], 0))
    if args.ack is not None:
        last = json.loads((cache / 'last_snapshot.json').read_text(encoding='utf-8'))
        maximum = max([0] + [e['sequence'] for e in last['events']])
        if not old <= args.ack <= maximum:
            parser.error('Cannot acknowledge unseen events or move cursor backwards')
        cursor[config['pipeline']] = args.ack
        temp = cursor_path.with_suffix('.tmp')
        temp.write_text(json.dumps(cursor, indent=2), encoding='utf-8')
        temp.replace(cursor_path)
        print(json.dumps({'acknowledged': args.ack}))
        return
    try:
        data = fetch(config)
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        print(json.dumps({'observation': 'unavailable', 'observed_at': datetime.now(timezone.utc).isoformat(),
                          'error': str(error), 'action': 'retry_later_do_not_restart'}, ensure_ascii=False))
        sys.exit(2)
    (cache / 'last_snapshot.json').write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    data['new_events'] = [e for e in data['events'] if e['sequence'] > old]
    print(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()