"""Read-only supplement for firmware 0x20250226: LUT, packed ADC and boot state.

Layouts verified against Berkeley-linked Recoil firmware commit
3571ab60a05951561abbab2e27c816b67b315bb0 and local Downloads firmware.
The version word alone cannot establish the exact installed firmware build.
"""
import argparse
import datetime
import json
from pathlib import Path
import struct
import time

from audit_motors import ReadOnlyBus, clean, errors, healthy, r, status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inventory', type=Path)
    args = parser.parse_args()
    prior = json.loads(args.inventory.read_text())
    out = args.inventory.parent / ('extended_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S') + '.json')
    report = {'controller_writes': [], 'buses': {}, 'motors': []}
    def save():
        out.write_text(json.dumps(clean(report), indent=2, allow_nan=False)+'\n')
    try:
        for channel in prior['buses']:
            before = status(channel)
            rec = report['buses'][channel] = {'before':before}
            if not healthy(before):
                rec['blocked'] = 'Not ERROR-ACTIVE at 1 Mbps'
                continue
            bus = ReadOnlyBus(channel)
            def check():
                now = status(channel)
                if not healthy(now) or any(v > errors(before).get(k,0) for k,v in errors(now).items()):
                    raise RuntimeError('CAN health changed')
            def read(device, addr):
                raw = bus._read_parameter_bytes(device,addr,timeout=0.15)
                if raw is None:
                    raise RuntimeError(f'SDO timeout {device} {addr:#x}; stop bus')
                return bytes(raw)
            try:
                # Recheck missing expected IDs only. No mode changes or reset attempts.
                expected = {'can0':[1,3,5,7,9], 'can1':[2,4,6,8,10], 'can2':[2,4,6,8,12,14], 'can3':[1,3,5,7,11,13]}[channel]
                rec['expected_pings'] = {}
                # Inventory unexpected IDs too, without assigning them a joint.
                discovered = [m['id'] for m in prior['motors'] if m['bus'] == channel]
                for d in sorted(set(expected + discovered)):
                    check()
                    rec['expected_pings'][str(d)] = [bus.ping(d,timeout=0.15) for _ in range(3)]
                for old in prior['motors']:
                    if old['bus'] != channel:
                        continue
                    d = old['id']
                    check()
                    if not all(rec['expected_pings'].get(str(d),[False])):
                        raise RuntimeError(f'Previously online ID {d} not stable')
                    firmware = struct.unpack('<I',read(d,r.Parameter.FIRMWARE_VERSION))[0]
                    if firmware != 0x20250226:
                        raise RuntimeError('Extended layout not verified for this firmware version')
                    m = {'bus':channel,'id':d,'firmware':hex(firmware),'raw_words':{},'flux_offset_table':[],'boot_samples':[]}
                    report['motors'].append(m)
                    # Read handles as raw addresses only, never dereference.
                    for addr in [0xd8,0xdc,0xe0,0xe4,0xe8,0xec,0xf0,0x114,0x118]:
                        m['raw_words'][hex(addr)] = read(d,addr).hex()
                    for i in range(128):
                        if i % 16 == 0:
                            check()
                        raw = read(d,0x140+4*i)
                        m['flux_offset_table'].append({'raw':raw.hex(),'value':struct.unpack('<f',raw)[0],
                            'classification':'MOTOR_SPECIFIC_DO_NOT_COMPARE_DIRECTLY'})
                        time.sleep(0.003)
                    for _ in range(3):
                        sample = {}
                        for name,fmt in [('MODE','<I'),('ERROR','<I'),('POSITION_CONTROLLER_UPDATE_COUNTER','<I'),
                                         ('ENCODER_POSITION_RAW','<H'),('ENCODER_POSITION','<f'),
                                         ('POWERSTAGE_BUS_VOLTAGE_MEASURED','<f')]:
                            raw = read(d,getattr(r.Parameter,name))
                            sample[name] = struct.unpack(fmt,raw[:struct.calcsize(fmt)])[0]
                        m['boot_samples'].append(sample)
                        time.sleep(0.1)
                    print(channel,d,'extended read complete',flush=True)
                    save()
                check()
            except Exception as exc:
                rec['blocked'] = str(exc)
                print(channel,'STOP',exc,flush=True)
            finally:
                bus.stop()
                rec['after'] = status(channel)
                save()
    finally:
        save()
        print(out,flush=True)


if __name__ == '__main__':
    main()
