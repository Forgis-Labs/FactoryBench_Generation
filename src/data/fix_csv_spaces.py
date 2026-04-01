"""Fix missing spaces in protocol CSV files (OCR artifact)."""
import re
import csv
from pathlib import Path
import wordninja

PRESERVE = {
    'rebooting', 'reboot', 'configuration', 'controller', 'communication',
    'connection', 'initializing', 'initialization', 'disconnected', 'connected',
    'unexpected', 'temperature', 'overheating', 'overloaded', 'overload',
    'corrupted', 'corruption', 'detected', 'protection', 'protecting',
    'shutdown', 'startup', 'checksum', 'mismatch', 'timeout', 'firmware',
    'calibration', 'encoder', 'software', 'hardware', 'network', 'ethernet',
    'frequency', 'voltage', 'current', 'package', 'packages', 'packet',
    'packets', 'motor', 'joint', 'robot', 'system', 'status', 'error',
    'safety', 'signal', 'cable', 'board', 'power', 'input', 'output',
    'position', 'velocity', 'trajectory', 'motion', 'control', 'program',
    'process', 'internal', 'external', 'module', 'register', 'memory',
    'address', 'command', 'request', 'response', 'master', 'received',
    'transmitted', 'detected', 'missing', 'failed', 'failure', 'invalid',
    'update', 'version', 'enabled', 'disabled', 'running', 'stopped',
    'parameter', 'parameters', 'value', 'values', 'limit', 'limits',
    'exceeded', 'threshold', 'within', 'without', 'during', 'before',
    'after', 'between', 'through', 'against', 'should', 'could', 'would',
    'restart', 'restarting', 'resetting', 'reset', 'retry', 'retrying',
    'interface', 'protocol', 'message', 'messages', 'number', 'counter',
    'index', 'offset', 'sequence', 'buffer', 'queue', 'stack', 'frame',
    'block', 'sector', 'segment', 'channel', 'mode', 'state', 'flag',
    'pending', 'complete', 'completed', 'active', 'inactive', 'enable',
    'disable', 'trigger', 'interrupt', 'exception', 'warning', 'critical',
    'attempt', 'attempts', 'timeout', 'watchdog', 'heartbeat', 'polling',
    'broadcasting', 'unicast', 'broadcast', 'multicast', 'address', 'port',
    'socket', 'payload', 'header', 'footer', 'data', 'information', 'report',
    'configure', 'setup', 'install', 'uninstall', 'upgrade', 'downgrade',
}

VALID_SINGLE = {'a', 'i', 'o'}


def valid_part(p):
    return len(p) > 1 or p.lower() in VALID_SINGLE or (len(p) == 1 and p.isupper())


def fix_word(word):
    if word.lower() in PRESERVE:
        return word
    parts = wordninja.split(word)
    if len(parts) > 1 and all(valid_part(p) for p in parts):
        result = ' '.join(parts)
        if word[0].isupper() and result[0].islower():
            result = result[0].upper() + result[1:]
        return result
    return word


def fix_spaces(text):
    if not text:
        return text
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)          # camelCase
    text = re.sub(r'[a-zA-Z]{2,}', lambda m: fix_word(m.group()), text)  # wordninja
    text = re.sub(r'\)(?=[^\s])', ') ', text)                   # space after )
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)           # digit→letter
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)           # letter→digit
    text = re.sub(r'\s*=\s*', ' = ', text)                     # spaces around =
    text = re.sub(r' {2,}', ' ', text).strip()
    return text


def fix_csv(path: Path):
    rows = list(csv.DictReader(path.open(encoding='utf-8')))
    fieldnames = list(rows[0].keys()) if rows else []
    text_fields = [f for f in fieldnames if f.lower() not in ('error_code', 'error code', 'code')]

    for row in rows:
        for field in text_fields:
            if row.get(field):
                row[field] = fix_spaces(row[field])

    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Fixed {len(rows)} rows in {path.name}")


if __name__ == '__main__':
    base = Path(__file__).resolve().parents[2] / 'data' / 'protocols'
    fix_csv(base / 'UR3_Runtime_Errors.csv')
    fix_csv(base / 'UR3_NonRuntime_Removed.csv')
    print("Done.")
