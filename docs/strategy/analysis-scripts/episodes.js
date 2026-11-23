const L=require('./lib.js'); const {potAll}=require('./bigchains.js');
const files=process.argv.slice(2); const eps=[]; let boards=0, pot5=0,pot6=0;
for(const f of files){ const R=L.replay(f); let cur=null;
  for(let t=0;t<R.recs.length;t++){ const m=R.recs[t]; const p=potAll(m.before,m.un); boards++; if(p.best>=5)pot5++; if(p.best>=6)pot6++;
    if(p.best>=6){ if(!cur) cur={f,start:t,len:0,outcome:'lost',outs:Object.keys(p.outs).filter(k=>p.outs[k][0]==6)}; cur.len++;
      if(m.chain>=5){cur.outcome='fired'+m.chain; eps.push(cur); cur=null;} }
    else if(cur){ cur.outcome = m.chain>=5?'fired'+m.chain:'lost'; eps.push(cur); cur=null; }
  }
  if(cur){cur.outcome='gameover';eps.push(cur);}
}
const cnt=(arr,fn)=>{const o={};for(const x of arr){const k=fn(x);o[k]=(o[k]||0)+1;}return JSON.stringify(o);};
console.log('boards',boards,'pot>=5 %',(100*pot5/boards).toFixed(1),'pot>=6 %',(100*pot6/boards).toFixed(1));
console.log('episodes',eps.length,cnt(eps,e=>e.outcome));
console.log('len by outcome'); for(const o of ['fired6','fired5','lost','gameover']){const e=eps.filter(x=>x.outcome==o); if(e.length) console.log(o,'mean len',(e.reduce((s,x)=>s+x.len,0)/e.length).toFixed(1),cnt(e,x=>x.len>=20?'20+':x.len>=10?'10-19':x.len>=5?'5-9':x.len));}
console.log('outs at episode start',cnt(eps,e=>e.outs.join('|')));
