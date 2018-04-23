import argparse
import sys
import os.path
import re
import shutil
from datetime import datetime

GREP_KEYWORD = r'@\s*nearest_rock:'


parser = argparse.ArgumentParser(description='log converter')
parser.add_argument(
    'input_filename',
    type=str,
    nargs='?',
    default='',
    help='Path to log file.'
)
parser.add_argument(
    'output_filename',
    type=str,
    nargs='?',
    default='',
    help='Path to output json.'
)
args = parser.parse_args()

if args.input_filename == '' or args.output_filename == '':
    print('Error: Not specified filenames.', file=sys.stderr)
    sys.exit(1)

if not os.path.exists(args.input_filename):
    print('Error: Input file is not exist.', file=sys.stderr)
    sys.exit(1)


# Backup input logs in same directory
backup_filename = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
path, ext = os.path.splitext(args.input_filename)
shutil.copy(args.input_filename, path + backup_filename + ext)


# Grep inputfilename
matched = None
with open(args.input_filename, 'r') as f:
    lines = f.readlines()
    matched = [ re.sub(GREP_KEYWORD, '', s) for s in lines if re.search(GREP_KEYWORD, s) ]


# To JSON like format?
json_like = [
    s.replace('(','[').replace(')',']').replace("'", '"') \
     .replace(' None', ' null').replace(' nan', ' null') \
    for s in matched ]

# print(len(json_like))
# print(''.join(json_like[:2]))
# print(''.join(json_like[-2:]))

# Write
with open(args.output_filename, 'w') as f:
    f.write('[\n' + ','.join(json_like) + ']')