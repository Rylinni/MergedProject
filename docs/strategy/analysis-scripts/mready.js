const L=require('./lib.js'); const {potAll}=require('./bigchains.js'); const E=L.E;
let n=0,fired=0; const what={};
for(const f of process.argv.slice(2)){ const R=L.replay(f); for(const m of R.recs){ if(!(m.vals.length===1&&m.vals[0]===7)) continue; const p=potAll(m.before,m.un); if(p.best<6) continue; n++; const k=m.events.some(e=>e.type==='boom')?'exploded':'held'; what[k]=(what[k]||0)+1; } }
console.log('M single arriving while 6-ready:',n,JSON.stringify(what));
