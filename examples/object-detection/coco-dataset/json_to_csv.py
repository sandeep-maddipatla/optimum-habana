import pandas as pd
import sys
import json
from pathlib import Path
import os

p = Path(sys.argv[1])
with p.open('r', encoding='utf-8') as inputfile:
    data = json.loads(inputfile.read())
    df = pd.json_normalize(data)
    outfile_name = sys.argv[1] + '.csv'
    df.to_csv(outfile_name, encoding='utf-8', index=False)
    cmd = "sed -i 's/},/\\n/g'" + " {filename}".format(filename=outfile_name)
    print(cmd)
    os.system(cmd)
