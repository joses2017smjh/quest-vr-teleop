"""Read-only Recoil inventory. Never sends mode, setpoint, heartbeat or flash writes.

Run from repository root with .venv/bin/python scripts/hardware/audit_motors.py.
The five comparison labels describe comparison to an EXAMPLE, not write approval.
No assertion of unique physical controllers can be made from CAN pings alone.
"""
import argparse
import csv
import datetime
import importlib.util
import json
import math
from pathlib import Path
import struct
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
LOW = ROOT / 'source/berkeley_humanoid_lite_lowlevel'
spec = importlib.util.spec_from_file_location('audit_recoil', LOW / 'berkeley_humanoid_lite_lowlevel/recoil/core.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


class ReadOnlyBus(r.Bus):
    def transmit(self, frame):
        if not ((frame.func_id == r.Function.RECEIVE_SDO and frame.size == 3 and frame.data[0] == 0x40)
                or (frame.func_id == r.Function.RECEIVE_PDO_1 and frame.data == b'\xca')):
            raise RuntimeError('Non-read operation prohibited')
        super().transmit(frame)

    def receive(self, filter_device_id=None, filter_function=None, timeout=0.15):
        # Unlike the stock receive, unrelated traffic cannot reset the deadline.
        deadline = time.monotonic() + (0.15 if timeout is None else timeout)
        while time.monotonic() < deadline:
            msg = self._Bus__bus.recv(timeout=max(0, deadline-time.monotonic()))
            if msg is None:
                return None
            if msg.is_error_frame:
                raise RuntimeError('CAN error frame: stop this bus')
            if msg.is_extended_id or msg.is_remote_frame:
                continue
            if msg.arbitration_id != ((filter_function << 7) | filter_device_id):
                continue
            expected_lengths = (4, 8) if filter_function == r.Function.TRANSMIT_SDO else (8,)
            if len(msg.data) not in expected_lengths:
                raise RuntimeError('Malformed response; stop bus to prevent misattribution')
            return r.CANFrame(filter_device_id, filter_function, len(msg.data), msg.data)
        return None


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def status(bus):
    return json.loads(subprocess.check_output(['ip', '-j', '-details', '-statistics', 'link', 'show', 'dev', bus], text=True))[0]


def healthy(s):
    info = s['linkinfo']['info_data']
    return info.get('state') == 'ERROR-ACTIVE' and info.get('bittiming', {}).get('bitrate') == 1000000


def errors(s):
    stats = s.get('stats64', s.get('stats', {}))
    result = {f'{direction}_{key}': stats.get(direction, {}).get(key, 0)
              for direction in ('rx', 'tx') for key in ('errors', 'dropped')}
    result.update(s['linkinfo'].get('info_xstats', {}))
    return {k: v for k, v in result.items() if isinstance(v, int)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--buses', nargs='+', choices=['can0','can1','can2','can3'], default=['can0','can1','can2','can3'])
    args = parser.parse_args()
    out = ROOT / 'logs/hardware_audit' / datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    out.mkdir(parents=True)
    backup = json.loads((LOW / 'robot_configuration.backup.json').read_text())
    topology = {}
    for joint, config in backup.items():
        side = joint.split('_')[0]
        arm = any(x in joint for x in ('shoulder','elbow','wrist'))
        bus = ('can0' if side == 'left' else 'can1') if arm else ('can3' if side == 'left' else 'can2')
        key = (bus, config['device_id'])
        assert key not in topology
        topology[key] = joint
    assert len(topology) == 22
    excluded = {'POWERSTAGE_HTIM','POWERSTAGE_HADC1','POWERSTAGE_HADC2','POWERSTAGE_ADC_READING_RAW',
                'POWERSTAGE_ADC_READING_OFFSET','ENCODER_HI2C','ENCODER_I2C_BUFFER','ENCODER_FLUX_OFFSET_TABLE'}
    unsigned = {'DEVICE_ID','FIRMWARE_VERSION','WATCHDOG_TIMEOUT','FAST_FRAME_FREQUENCY','MODE','ERROR',
                'POSITION_CONTROLLER_UPDATE_COUNTER','MOTOR_POLE_PAIRS','ENCODER_CPR','ENCODER_I2C_UPDATE_COUNTER'}
    signed = {'MOTOR_PHASE_ORDER','ENCODER_N_ROTATIONS','ENCODER_POSITION_RAW'}
    report = {'created_utc':out.name, 'controller_writes':[], 'buses':{}, 'motors':[],
              'excluded_parameters':sorted(excluded),
              'limitations':['Example comparisons are not validated repair values.',
              'Pointers and packed arrays/table excluded: scalar API does not define their serialization.',
              'Pings cannot exclude duplicate physical controllers with identical IDs.',
              'Joint names are expected topology, pending physical mapping verification.',
              'After any SDO timeout this bus is closed; late replies cannot be assigned to subsequent parameters.']}
    def save():
        (out/'inventory.json').write_text(json.dumps(clean(report), indent=2, allow_nan=False)+'\n')
    print('Output:',out,flush=True)
    try:
        for channel in args.buses:
            before = status(channel)
            record = report['buses'][channel] = {'before':before, 'scan':{}, 'health_checks':[]}
            save()
            if not healthy(before):
                record['blocked'] = 'Interface not ERROR-ACTIVE at 1 Mbps'
                print(channel, record['blocked'], flush=True)
                continue
            bus = ReadOnlyBus(channel)
            def check():
                current = status(channel)
                record['health_checks'].append(current)
                if not healthy(current) or any(v > errors(before).get(k, 0) for k,v in errors(current).items()):
                    raise RuntimeError('CAN health changed; no further traffic on this bus')
            try:
                for device in range(1,21):
                    check()
                    record['scan'][str(device)] = bus.ping(device, timeout=0.15)
                    print(channel, device, record['scan'][str(device)], flush=True)
                    save()
                for device in range(1,21):
                    if not record['scan'][str(device)]:
                        continue
                    joint = topology.get((channel,device), 'UNEXPECTED_ID')
                    motor = {'bus':channel, 'id':device, 'expected_joint':joint, 'parameters':{}}
                    report['motors'].append(motor)
                    for name,address in vars(r.Parameter).items():
                        if not name.isupper() or name in excluded:
                            continue
                        check()
                        raw = bus._read_parameter_bytes(device,address,timeout=0.15)
                        entry = motor['parameters'][name] = {'address':hex(address),'raw':None,'value':None,'classification':'UNREADABLE'}
                        if raw is None:
                            save()
                            raise RuntimeError(f'SDO timeout at {device} {name}; abort bus to avoid late-response mixup')
                        fmt = '<I' if name in unsigned else '<i' if name in signed else '<f'
                        value = struct.unpack(fmt,raw)[0]
                        entry.update(raw=raw.hex(), value=value, decode=fmt)
                        ref = backup.get(joint,{})
                        reference = None
                        for group in ('position_controller','current_controller','powerstage','motor','encoder'):
                            if name.startswith(group.upper()+'_'):
                                reference = ref.get(group,{}).get(name[len(group)+1:].lower())
                                break
                        else:
                            reference = ref.get(name.lower())
                        if name == 'FIRMWARE_VERSION' and reference is not None:
                            reference = int(reference,16)
                        entry['example_reference'] = reference
                        if 'OFFSET' in name or name == 'MOTOR_PHASE_ORDER':
                            label = 'MOTOR_SPECIFIC_DO_NOT_COMPARE_DIRECTLY'
                        elif reference is not None and (value == reference or math.isclose(value,reference,rel_tol=1e-5,abs_tol=1e-7)):
                            label = 'MATCH'
                        elif value == 0 or not math.isfinite(value):
                            label = 'MISSING/ZERO'
                        else:
                            label = 'DIFFERENT_BUT_PLAUSIBLE'
                        entry['classification'] = label
                        entry['note'] = 'Example equality only; not safety validation.' if label == 'MATCH' else 'Requires review; zero telemetry may be normal and plausible does not mean validated.'
                        save()
                        time.sleep(0.003)
                    print(channel,device,joint,'inventory complete',flush=True)
                record['repeat_pings'] = {str(d):[bus.ping(d,timeout=0.15) for _ in range(3)] for d in range(1,21) if record['scan'][str(d)]}
                check()
            except Exception as exc:
                record['blocked'] = str(exc)
                print(channel,'STOP:',exc,flush=True)
            finally:
                bus.stop()
                record['after'] = status(channel)
                save()
    finally:
        save()
        with (out/'parameters.csv').open('w') as f:
            writer = csv.writer(f)
            writer.writerow(['bus','id','expected_joint','parameter','value','example_reference','classification'])
            for m in report['motors']:
                for name,p in m['parameters'].items():
                    writer.writerow([m['bus'],m['id'],m['expected_joint'],name,p['value'],p.get('example_reference'),p['classification']])
        lines = ['# Read-only hardware inventory','', '| Bus | ID | Expected joint (not physically verified) | Ping |', '|---|---:|---|---|']
        for (b,d),j in sorted(topology.items()):
            scan = report['buses'].get(b,{}).get('scan',{})
            lines.append(f'| {b} | {d} | {j} | {scan.get(str(d), "NOT TESTED")} |')
        lines += ['', 'Controller writes: none. See inventory.json for errors and exclusions; parameters.csv for every scalar read.']
        (out/'summary.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
