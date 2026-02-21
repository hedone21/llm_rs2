import re

with open('results/README.md', 'r') as f:
    content = f.read()

# The table starts with | Date | Model | Backend ...
# Let's find the header row and separator row
lines = content.split('\n')
new_lines = []
in_table = False

for line in lines:
    if re.search(r'^\| Date\s*\| Model', line):
        in_table = True
        # Insert Device after Backend (which is at index 3, if split by | then 1=Date, 2=Model, 3=Backend)
        parts = line.split('|')
        if 'Device' not in line:
            parts.insert(4, ' Device ')
        new_lines.append('|'.join(parts))
    elif in_table and re.search(r'^\| :---', line):
        parts = line.split('|')
        # Only add if it hasn't been added
        if len(parts) < len(new_lines[-2].split('|')):
            parts.insert(4, ' :--- ')
        new_lines.append('|'.join(parts))
    elif in_table and line.startswith('|'):
        parts = line.split('|')
        if len(parts) > 4: # It's a data row
            # If the current line length is shorter than the header length, we need to insert
            if len(parts) < len(new_lines[-2].split('|')):
                parts.insert(4, ' Galaxy S25 ')
            new_lines.append('|'.join(parts))
        else:
            new_lines.append(line)
    else:
        if not line.startswith('|'):
            in_table = False
        new_lines.append(line)

with open('results/README.md', 'w') as f:
    f.write('\n'.join(new_lines))
print('Done updating results/README.md')
