import json, sys
here = sys.argv[0].rsplit('/', 1)[0]
tpl = open(f'{here}/template.html').read()
eng = open(f'{here}/engine.js').read()
run = json.load(open(sys.argv[1]))
out = tpl.replace('/*__ENGINE__*/', eng).replace('/*__RUN__*/', json.dumps(run, separators=(',', ':')))
open(sys.argv[2], 'w').write(out)
print(sys.argv[2], len(out) // 1024, 'KB')
