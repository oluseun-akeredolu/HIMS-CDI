import pathlib

p = pathlib.Path(r"phase-1-cic-iomt/src/csra/streaming_engine.py")
c = p.read_text()
lines = c.splitlines()

new_lines = []
in_process_event = False
for i, line in enumerate(lines):
    if 'def process_event(' in line:
        in_process_event = True
    if in_process_event and 'return None' in line and 'if missing' not in line:
        stripped = line.lstrip()
        if stripped == 'return None':
            new_lines.append('        return None')
            continue
    if in_process_event and 'x = np.array' in line and 'event_dict[col]' in line:
        stripped = line.lstrip()
        if stripped.startswith('x = np.array'):
            new_lines.append('    ' + stripped)
            continue
    new_lines.append(line)

p.write_text('\n'.join(new_lines) + '\n')
print('Fixed return None indentation')
