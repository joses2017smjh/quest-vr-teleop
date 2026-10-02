"""Generate a review report from a saved inventory; no hardware access or writes."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inventory', type=Path)
    args = parser.parse_args()
    audit = json.loads(args.inventory.read_text())
    root = Path(__file__).resolve().parents[2]
    example = json.loads((root / 'source/berkeley_humanoid_lite_lowlevel/robot_configuration.backup.json').read_text())
    lines = ['# Hardware audit — configuration and movement blocked', '',
             f"Fresh inventory: {audit['created_utc']}. Controller writes: **none**.", '',
             'Expected joint names below follow the user-supplied topology, not a verified physical mapping.', '',
             '| Bus | Expected limb | ID | Expected joint | Ping |', '|---|---|---:|---|---|']
    for joint, cfg in example.items():
        side = joint.split('_')[0]
        arm = any(x in joint for x in ('shoulder', 'elbow', 'wrist'))
        bus = ('can0' if side == 'left' else 'can1') if arm else ('can3' if side == 'left' else 'can2')
        scan = audit['buses'].get(bus, {}).get('scan', {})
        online = scan.get(str(cfg['device_id']), 'NOT TESTED')
        lines.append(f"| {bus} | {side} {'arm' if arm else 'leg'} | {cfg['device_id']} | {joint} | {online} |")
    lines += ['', '## CAN health', '']
    for bus, rec in audit['buses'].items():
        lines.append(f"- {bus}: responding IDs {[k for k,v in rec['scan'].items() if v]}; stop reason: {rec.get('blocked', 'none')}.")
        for stage in ('before', 'after'):
            s = rec.get(stage)
            if not s:
                continue
            info = s['linkinfo']
            stats = s.get('stats64', {})
            counters = dict(info.get('info_xstats', {}))
            counters.update({f'{direction}_{k}': stats.get(direction, {}).get(k) for direction in ('rx','tx') for k in ('errors','dropped')})
            lines.append(f"  - {stage}: {info['info_data']['state']}, bitrate {info['info_data'].get('bittiming',{}).get('bitrate')}; {counters}")
    lines += ['', '## Parameter overview', '',
              'All scalar reads, raw bytes, example values and required comparison labels are in parameters.csv and inventory.json. MATCH means example equality only. Zero integrators, IDLE telemetry and zero error codes are often normal; the MISSING/ZERO label alone is not a diagnosis. Actuator offsets and phase order must not be copied from the example.', '',
              '| Bus:ID | Mode | Error | Gear | Kp | Kd | Current limit | Pole pairs | Torque constant | Phase | Calibration A | Flux offset | Bus V |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    fields = ['MODE','ERROR','POSITION_CONTROLLER_GEAR_RATIO','POSITION_CONTROLLER_POSITION_KP',
              'POSITION_CONTROLLER_VELOCITY_KP','CURRENT_CONTROLLER_I_LIMIT','MOTOR_POLE_PAIRS',
              'MOTOR_TORQUE_CONSTANT','MOTOR_PHASE_ORDER','MOTOR_MAX_CALIBRATION_CURRENT',
              'ENCODER_FLUX_OFFSET','POWERSTAGE_BUS_VOLTAGE_MEASURED']
    for m in audit['motors']:
        values = [m['parameters'].get(k, {}).get('value', 'UNREADABLE') for k in fields]
        lines.append('| ' + ' | '.join([f"{m['bus']}:{m['id']}"] + [f'{v:.7g}' if isinstance(v,float) else str(v) for v in values]) + ' |')
    lines += ['', '## Complete example discrepancy list', '',
              'These are review candidates, not approved write values. Every row must pass physical identification and firmware validation before a repair can be proposed.', '',
              '| Bus:ID | Expected joint | Parameter | Current | Example candidate | Decision |', '|---|---|---|---|---|---|']
    for m in audit['motors']:
        for name, p in m['parameters'].items():
            ref = p.get('example_reference')
            if ref is None or p['classification'] == 'MATCH':
                continue
            decision = 'Preserve; actuator-specific' if p['classification'] == 'MOTOR_SPECIFIC_DO_NOT_COMPARE_DIRECTLY' else 'HOLD: physical identity/configuration not validated'
            lines.append(f"| {m['bus']}:{m['id']} | {m['expected_joint']} | {name} | {p['value']} | {ref} | {decision} |")
    lines += ['', '## Repair decisions and source findings', '',
              '- No controller settings, IDs, modes, setpoints or flash were written. No robot_configuration.json was generated.',
              '- Zero electrical configuration plus MODE=0 and zero voltage can indicate incomplete firmware initialization. Reference Encoder_init retries I2C indefinitely before motor/current initialization. Inspect encoder power, connector, I2C continuity and boot diagnostics before any SDO repair; this is a hypothesis, not a proven diagnosis.',
              '- Existing M6C12 values 0.08958 torque constant, current Kp about 0.190956 and Ki about 4538.506 match the archived firmware profile; the backup uses older/different characterization and gains. Preserve pending actuator identification. The 5010 profile has 0.1176 torque constant, Kp about 0.534071 and Ki about 7285.882. Do not halve gains merely to match the backup.',
              '- The expected right-leg proximal joints use the M6C12 example profile, but observed can2 profiles must be compared with physical motors before changing torque constants or gains. Pings cannot prove physical limb assignment or exclude duplicate same-bus IDs.',
              '- Gear ratio 1 versus example -15 is a joint scaling discrepancy. Position Kp 1 versus leg example 20/arm 50 and Kd 0.1 versus 2 are also discrepancies, not permission to raise gains. Example limits are infinite and do not provide safe mechanical limits.',
              '- Zero flux offsets need calibration investigation. A nonzero finite offset alone does not prove calibration. The reference firmware does not wrap the stored scalar to 2*pi; large values must be investigated without replacing them with another robot\'s value. Preserve the full actuator-specific LUT too.',
              '- The reference calibration resets scalar/LUT offsets, waits for bus voltage >=9 V, ramps open-loop voltage to calibration current, sweeps one rotor revolution each way, computes scalar/LUT, automatically stores configuration, then enters IDLE. It requires a working encoder, valid 1..32 pole pairs, correct phase wiring/order, valid power/current measurement and positive model-specific calibration current. It uses open-loop voltage; current PI gains are not the sweep regulator, but the entire configuration must be valid before its automatic flash save.',
              '- For a verified 5010-110KV actuator such as the example left ankle roll: candidate prerequisites are 14 pole pairs, CPR 4096, torque constant 0.1176 and calibration current 3 A. Phase order is wiring-specific. These remain conditional until the actual motor and boot state are verified. No flux value is proposed.',
              '- Official calibration Python merely sends CALIBRATION and sleeps 20 seconds; no completion/error check or finally cleanup. Do not execute unchanged on this assembled robot. Official docs require an unattached actuator free to spin.',
              '- Local Humanoid uses can0/can1 as legs, exposes only 12 joints, and stop() enters DAMPING rather than passive IDLE. It is unsuitable for this topology. read/write robot scripts inherit this mismatch; the writer copies offsets and flashes without sufficient gates.',
              '- Joint-zero calibration is separate: calibrate_joints.py samples signed manually reached limit positions, subtracts ideal limit angles and writes calibration.yaml after a gamepad trigger. Run after each power cycle only with verified mapping/scale; its Humanoid dependency and damping cleanup require correction before use.',
              '- Stock move_actuator.py commands a 1-radian, 1 Hz sine around zero and does not abort on absent PDO feedback. It was inspected, not run. No powered test utility is authorized to bypass failed gates.',
              '', '## Stage results and next safe actions', '',
              '- Electrical recalibrations: none. Flux before/after: unchanged; before values above. Flash writes: none, so no flash before/after pair exists.',
              '- Joint-zero calibration: not performed. Right-leg passive mapping and individual tests A–D: not performed. Coordinated test E: not performed.',
              '- Standing remains blocked by unresolved physical topology, missing expected nodes, incomplete initialization/calibration, joint settings/limits/zero verification and all movement gates.',
              '- Trace can1/can2 adapter cables to limbs before assigning unexpected IDs. Power off before inspecting connectors/continuity. For unreachable expected nodes, check local supply/ground, CAN-H/CAN-L continuity, connectors and ESC CAN solder joints; do not keep issuing mode/reset commands.',
              '- For zero-initialized nodes, inspect encoder supply and I2C wiring and obtain serial/debugger boot evidence with PWM disabled. No serial-by-id device was present during source review.',
              '- After physical corrections, repeat the read-only inventory. Then review one verified actuator\'s exact proposed diff before repairs, retaining snapshots and actuator-specific calibration.',
              '', '## Reproduction', '',
              'From the repository root, with each interface already ERROR-ACTIVE at 1 Mbps:', '',
              '```bash', '.venv/bin/python scripts/hardware/audit_motors.py',
              f'.venv/bin/python scripts/hardware/audit_extended.py {args.inventory}',
              f'python3 scripts/hardware/summarize_audit.py {args.inventory}', '```', '',
              'Use the newly printed inventory path for subsequent runs. SocketCAN/netlink requires host access outside the sandbox. No validated powered startup/test sequence exists yet.', '',
              'For a stopped interface only, host setup is `sudo ip link set canN up type can bitrate 1000000`. Inspect `ip -j -details -statistics link show type can` before and after; do not reset a faulted bus to conceal errors.', '',
              '## Sources and revision', '',
              '- Local root edc4738 is one README-only commit after v1.1.0 aa93e47. Current upstream root HEAD: 984741a. Local low-level 652777c equals official upstream HEAD; no pull/reset/checkout performed.',
              '- Saved upstream comparison changes README, dependency/lock metadata and training code; low-level submodule remains the same. Evidence and hashes: ../20260912T085310.014708Z/source_evidence/provenance.json.',
              '- [Official flashing/calibration workflow](https://berkeley-humanoid-lite.gitbook.io/docs/getting-started-with-hardware/flashing-the-motor-controllers)',
              '- [Official low-level revision](https://github.com/HybridRobotics/Berkeley-Humanoid-Lite-Lowlevel/tree/652777cc7c49884e7cd7ddfada758dc1979bf627)',
              '- [Berkeley-linked firmware reference](https://github.com/T-K-233/Recoil-Motor-Controller-BESC/tree/3571ab60a05951561abbab2e27c816b67b315bb0)',
              '- Firmware version 0x20250226 alone does not prove that exact firmware build is installed.',
              '', 'Files created/changed during this continuation: fresh timestamped audit artifacts; scripts/hardware/summarize_audit.py added; scripts/hardware/audit_extended.py expanded to include unexpected responding IDs. Existing working-tree edits were preserved.']
    (args.inventory.parent / 'review.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
