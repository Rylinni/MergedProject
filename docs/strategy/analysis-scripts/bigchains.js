const L=require('./lib.js'); const E=L.E; const fs=require('fs');
function pieces(un){const v=[];for(let x=1;x<=7;x++)if(un>>x&1)v.push(x);const p=v.map(x=>[x]);for(let i=0;i<v.length;i++)for(let j=i+1;j<v.length;j++)p.push([v[i],v[j]]);return p;}
function potAll(b,un){ let best=0,bestPts=0; const outs={}; 
  for(const pc of pieces(un)){ let pb=0,pp=0; for(const [a,bb] of E.legalMoves(b,pc)){ const nb=Uint8Array.from(b); const r=E.apply(nb,pc,a,bb,un); if(r.chain>pb||(r.chain===pb&&r.points>pp)){pb=r.chain;pp=r.points;} }
    if(pb>=5) outs[pc.map(x=>L.CH[x]).join('+')]=[pb,pp]; if(pb>best){best=pb;} if(pp>bestPts)bestPts=pp; }
  return {best,bestPts,outs}; }
module.exports={potAll,pieces};
if(require.main===module){
const [,,mode,...files]=process.argv;
const events=[];
for(const f of files){ const R=L.replay(f); R.recs.forEach((m,t)=>{ if(m.chain>=5){
   const boom=m.events.find(e=>e.type==='boom'); const debris=boom?boom.cells.reduce((s,x)=>s+(m.before[x]||0)+0,0):0;
   // debris: values in box at time of boom are not before-values necessarily; recompute by replaying to boom
   events.push({f,t,vals:m.vals.map(x=>L.CH[x]).join('+'),cells:[L.rc(m.a),L.rc(m.b)],pts:m.pts,chain:m.chain,sum:m.pts/[1,1,2,6,24,120,720][m.chain],boom:!!boom,boomCell:boom?L.rc(boom.cell):null,nDebris:boom?boom.cells.length:0,gameMoves:R.recs.length,
     ev:m.events.map(e=>e.type==='merge'?`${L.rc(e.cell)}:${L.CH[e.value]}x${e.cells.length}`:`boom@${L.rc(e.cell)}`).join(' '),
     before:L.show(m.before), recs:R.recs});
 }});}
console.error('events',events.length);
const summary=[];
for(const ev of events){ const R=ev.recs; let age=0; const hist=[];
  for(let k=1;k<=80&&ev.t-k>=0;k++){ const m=R[ev.t-k+1]; /* board before move t-k+1 = after move t-k */ const p=potAll(m.before,m.un); hist.push(p.best); if(p.best>=5)age=k; else break; }
  const pm=R[ev.t]; const p1=potAll(pm.before,pm.un);
  const s={f:ev.f.split('/').slice(-2).join('/'),t:ev.t,vals:ev.vals,cells:ev.cells,pts:ev.pts,chain:ev.chain,sum:ev.sum,boom:ev.boom,boomCell:ev.boomCell,nDebris:ev.nDebris,ev:ev.ev,loadedAge:age,outs:p1.outs,emp:L.empties(pm.before),gameMoves:ev.gameMoves,moveFrac:(ev.t/ev.gameMoves).toFixed(2),before:ev.before};
  if(mode==='show'){ s.prev=[]; for(let k=5;k>=1;k--) if(ev.t-k>=0){const q=R[ev.t-k]; s.prev.push({t:ev.t-k,before:L.show(q.before).join('/'),piece:q.vals.map(x=>L.CH[x]).join('+'),move:[L.rc(q.a),L.rc(q.b)],pts:q.pts});} }
  summary.push(s);
}
console.log(JSON.stringify(summary));
}
