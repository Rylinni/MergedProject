const L=require('./lib.js'); const E=L.E;
const R=L.replay(process.argv[2]); const pr=(m,tag)=>{console.log(`## ${tag} move ${m.i} piece ${m.vals.map(x=>L.CH[x]).join('+')} -> ${L.rc(m.a)} ${L.rc(m.b)} pts ${m.pts} chain ${m.chain} empt ${L.empties(m.before)}`);console.log(L.show(m.before).join('\n'));console.log('after');console.log(L.show(m.after).join('\n'));console.log(m.events.map(e=>e.type+'@'+L.rc(e.cell)+':'+L.CH[e.value||0]+'x'+(e.cells||[]).length).join(' '));};
let c={dec:0,boom:0,low:0,ho:0,big:0};
for(const m of R.recs){ if(m.i<3000) continue;
  let best=0,bm=null; for(const [a,bb] of E.legalMoves(m.before,m.vals)){const nb=Uint8Array.from(m.before);const r=E.apply(nb,m.vals,a,bb,m.un); if(r.points>best){best=r.points;bm=[a,bb];}}
  const ne=L.empties(m.before);
  if(c.dec<3 && best>=30 && m.pts===0 && ne>=14){c.dec++; pr(m,'DECLINE best='+best+' at '+bm.map(L.rc));}
  if(c.boom<2 && m.events.some(e=>e.type==='boom') && m.vals.length===1 && m.vals[0]===7 && m.i>5000){c.boom++; pr(m,'BOOM');}
  if(c.low<3 && ne<=3 && m.i>4000){c.low++; pr(m,'LOW');}
  if(c.ho<2 && m.vals.length===2 && m.chain>=3 && m.events.some(e=>e.cells&&e.cells.includes(m.a)&&e.cells.includes(m.b)) && m.i>6000){c.ho++; pr(m,'HANDOFF');}
  if(m.chain>=4 && c.big<4){c.big++; pr(R.recs[m.i-1],'PRE'); pr(m,'BIG');}
}
