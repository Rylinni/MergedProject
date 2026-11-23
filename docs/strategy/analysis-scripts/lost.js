const L=require('./lib.js'); const {potAll}=require('./bigchains.js');
const files=process.argv.slice(2); const end=[]; const holdMoves=[];
for(const f of files){ const R=L.replay(f); let on=false;
  for(let t=0;t<R.recs.length;t++){ const m=R.recs[t]; const p=potAll(m.before,m.un); const six=p.best>=6;
    if(six){ const after=t+1<R.recs.length?potAll(m.after,m.unAfter).best:0; holdMoves.push({chain:m.chain,pts:m.pts,emp:L.empties(m.before),keep:after>=6,fired:m.chain>=5,piece:m.vals.length});
      if(after<6 && m.chain<5) end.push({chain:m.chain,pts:m.pts,emp:L.empties(m.before),piece:m.vals.map(x=>L.CH[x]).join('+'),afterBest:after}); }
  }}
const cnt=(arr,fn)=>{const o={};for(const x of arr){const k=fn(x);o[k]=(o[k]||0)+1;}return JSON.stringify(o);};
console.log('lost-ending moves',end.length,'chain',cnt(end,e=>e.chain),'emp',cnt(end,e=>e.emp),'afterBest',cnt(end,e=>e.afterBest));
const h=holdMoves.filter(x=>!x.fired); console.log('moves while 6-ready (not firing)',h.length,'merge rate',(h.filter(x=>x.chain>0).length/h.length).toFixed(2),'kept',(h.filter(x=>x.keep).length/h.length).toFixed(2),'mean emp',(h.reduce((s,x)=>s+x.emp,0)/h.length).toFixed(1));
