import json, sys
import os
here = os.path.dirname(os.path.abspath(__file__))
tpl = open(f'{here}/template.html').read()
eng = open(f'{here}/engine.js').read()
# A run path inlines the game; "-" leaves it out so the page fetches run.json beside it.
run = 'null' if sys.argv[1] == '-' else json.dumps(json.load(open(sys.argv[1])), separators=(',', ':'))
out = tpl.replace('/*__ENGINE__*/', eng).replace('/*__RUN__*/', run)
open(sys.argv[2], 'w').write(out)
print(sys.argv[2], len(out) // 1024, 'KB')
