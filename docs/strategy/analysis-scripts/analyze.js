const L=require('./lib.js'); const E=L.E; const fs=require('fs');
const [,,label,...files]=process.argv;
const enumAll = !process.env.NOENUM;
const A={label,games:0,moves:0,scores:[],lens:[],chainHist:[0,0,0,0,0,0,0],maxMove:[],
 phase:{},empHist:new Array(26).fill(0),
 cellVal:Array.from({length:8},()=>new Array(25).fill(0)),
 pairsPerBoard:0,livePairs:0,pairsByVal:new Array(8).fill(0),triplesPlus:0,
 singletons:0,buried:0,
 declineAvail:0,declineCount:0,declineByEmp:{},takeLess:0,matchGreedy:0,enumMoves:0,
 bestAvailHist:{},
 potHist:new Array(8).fill(0),
 lowPlace:{n:0,touchEq:0,cls:{}},
 dom:{n:0,merge:0,handoff:0,both:0,nonMergeBothTouch:0,nonMerge:0},
 boom:{n:0,cls:{},destroyed:0,destroyedSum:0,chain:{},Mtouch:0},
 death:[],
 lowSpace:{n:0,merge:0,declined:0},
 Mpair:0,MboardCount:0,Mtiles:0,MinCornerEdge:0,
 cornerHigh:0,
};
function ph(un){const m=L.maxUn(un);return m<=3?'opening':m<=6?'mid':'late';}
function pot(b,un){ // max single-tile chain over empties/values
  let best=0; for(let c=0;c<25;c++){ if(b[c])continue; for(let v=1;v<=7;v++){ if(!(un>>v&1))continue;
    // quick check: need 2 equal neighbours group
    const nb=Uint8Array.from(b); const r=E.apply(nb,[v],c,-1,un); if(r.chain>best)best=r.chain; } } return best; }
for(const f of files){
  let R; try{R=L.replay(f);}catch(e){console.error(e.message);continue;}
  A.games++; A.scores.push(R.score); A.lens.push(R.recs.length);
  let mm=0; let minE=25;
  for(const m of R.recs){
    A.moves++; mm=Math.max(mm,m.pts); A.chainHist[m.chain]++;
    const p=ph(m.un); const P=A.phase[p]||(A.phase[p]={n:0,merge:0,emp:0,pts:0,chain:[0,0,0,0,0,0,0]}); P.n++; if(m.chain)P.merge++; P.pts+=m.pts; P.chain[m.chain]++;
    const b=m.after; const ne=L.empties(b); P.emp+=ne; A.empHist[ne]++; minE=Math.min(minE,ne);
    for(let i=0;i<25;i++) A.cellVal[b[i]][i]++;
    const cs=L.comps(b);
    for(const c of cs){ if(c.cells.length===1)A.singletons++; if(c.cells.length===2){A.pairsPerBoard++;A.pairsByVal[c.v]++; if(c.cells.some(x=>E.NB[x].some(y=>!b[y])))A.livePairs++;} if(c.cells.length>=3)A.triplesPlus++; }
    for(let i=0;i<25;i++) if(b[i]&&E.NB[i].every(y=>b[y])) A.buried++;
    let Ms=0; for(let i=0;i<25;i++) if(b[i]===7){Ms++; if(L.cls(i)!=='inner'&&L.cls(i)!=='center')A.MinCornerEdge++;}
    A.Mtiles+=Ms; if(Ms)A.MboardCount++; if(cs.some(c=>c.v===7&&c.cells.length===2))A.Mpair++;
    // low tile placement
    if(m.vals.length===1 && m.pts===0 && m.vals[0]<=2){ A.lowPlace.n++; const c=L.cls(m.a); A.lowPlace.cls[c]=(A.lowPlace.cls[c]||0)+1; if(E.NB[m.a].some(y=>m.before[y]===m.vals[0]))A.lowPlace.touchEq++; }
    if(m.vals.length===2){ A.dom.n++; if(m.chain)A.dom.merge++;
      const merges=m.events.filter(e=>e.type==='merge');
      const ho=merges.some(e=>e.cells.includes(m.a)&&e.cells.includes(m.b));
      if(ho)A.dom.handoff++; else { const cellsM=new Set(merges.map(e=>e.cell)); if(cellsM.has(m.a)&&cellsM.has(m.b))A.dom.both++; }
      if(!m.chain){A.dom.nonMerge++; if(E.NB[m.a].some(y=>y!==m.b&&m.before[y]===m.vals[0])&&E.NB[m.b].some(y=>y!==m.a&&m.before[y]===m.vals[1]))A.dom.nonMergeBothTouch++;}
    }
    for(const e of m.events) if(e.type==='boom'){A.boom.n++; const c=L.cls(e.cell); A.boom.cls[c]=(A.boom.cls[c]||0)+1; A.boom.destroyed+=e.cells.length; for(const x of e.cells)A.boom.destroyedSum+=m.before[x]||0; A.boom.chain[m.chain]=(A.boom.chain[m.chain]||0)+1;}
    const neB=L.empties(m.before);
    if(enumAll){
      let best=0; for(const [a,bb] of E.legalMoves(m.before,m.vals)){ const nb=Uint8Array.from(m.before); const r=E.apply(nb,m.vals,a,bb,m.un); if(r.points>best)best=r.points; }
      A.enumMoves++;
      if(best>0){ A.declineAvail++; if(m.pts===0){A.declineCount++; const k=Math.min(neB,10); A.declineByEmp[k]=(A.declineByEmp[k]||[0,0]); A.declineByEmp[k][0]++;} else if(m.pts<best)A.takeLess++; const k=Math.min(neB,10); A.declineByEmp[k]=(A.declineByEmp[k]||[0,0]); A.declineByEmp[k][1]++; }
      const gm=E.bestMove(m.before,m.vals,m.un); if(gm && gm[0]===m.a && gm[1]===m.b)A.matchGreedy++;
      if(neB<=4){A.lowSpace.n++; if(m.chain)A.lowSpace.merge++; if(best>0&&m.pts===0)A.lowSpace.declined++;}
      if(m.i%5===0) A.potHist[pot(b,m.unAfter)]++;
    }
  }
  const last=R.recs[R.recs.length-1];
  const lb=last.after; const cs=L.comps(lb);
  A.death.push({emp:L.empties(lb),groups:cs.length,pairs:cs.filter(c=>c.cells.length==2).length,distinct:new Set(cs.map(c=>c.v)).size,minE,
     emp20:L.empties(R.recs[Math.max(0,R.recs.length-21)].after), lastPiece: last.vals.length});
  A.maxMove.push(mm);
}
const med=a=>{const s=[...a].sort((x,y)=>x-y);return s[Math.floor(s.length/2)];};
const mean=a=>a.reduce((x,y)=>x+y,0)/a.length;
const o={label,games:A.games,moves:A.moves,meanScore:mean(A.scores).toFixed(0),medScore:med(A.scores),meanLen:mean(A.lens).toFixed(0),
 maxMoveMean:mean(A.maxMove).toFixed(0),maxMoveMax:Math.max(...A.maxMove),
 chainPct:A.chainHist.map(x=>(100*x/A.moves).toFixed(3)),chainCount:A.chainHist,
 meanEmpty:(A.empHist.reduce((s,x,i)=>s+x*i,0)/A.moves).toFixed(2),
 empPct:A.empHist.map(x=>(100*x/A.moves).toFixed(1)).slice(0,14),
 pairsPerBoard:(A.pairsPerBoard/A.moves).toFixed(2),livePairFrac:(A.livePairs/A.pairsPerBoard).toFixed(3),pairsByVal:A.pairsByVal.map(x=>(x/A.moves).toFixed(2)),
 singletonsPerBoard:(A.singletons/A.moves).toFixed(2),buriedPerBoard:(A.buried/A.moves).toFixed(2),triplesPlus:A.triplesPlus,
 phase:Object.fromEntries(Object.entries(A.phase).map(([k,P])=>[k,{n:P.n,mergeRate:(P.merge/P.n).toFixed(3),emp:(P.emp/P.n).toFixed(2),ptsPerMove:(P.pts/P.n).toFixed(1),chainPct:P.chain.map(x=>(100*x/P.n).toFixed(2))}])),
 decline: enumAll?{movesWithMergeAvail:A.declineAvail,declined:A.declineCount,declinePct:(100*A.declineCount/A.declineAvail).toFixed(1),tookSmallerPct:(100*A.takeLess/A.declineAvail).toFixed(1),byEmptyBefore:Object.fromEntries(Object.entries(A.declineByEmp).map(([k,v])=>[k,`${v[0]}/${v[1]}=${(100*v[0]/v[1]).toFixed(1)}%`]))}:null,
 matchGreedyPct:enumAll?(100*A.matchGreedy/A.enumMoves).toFixed(1):null,
 potHistPct:A.potHist.map(x=>(100*x/A.potHist.reduce((a,b)=>a+b,0)).toFixed(1)),
 lowPlace:{n:A.lowPlace.n,touchEqPct:(100*A.lowPlace.touchEq/A.lowPlace.n).toFixed(1),cls:Object.fromEntries(Object.entries(A.lowPlace.cls).map(([k,v])=>[k,(100*v/A.lowPlace.n).toFixed(1)]))},
 dom:{...A.dom,mergePct:(100*A.dom.merge/A.dom.n).toFixed(1),handoffPctOfMerging:(100*A.dom.handoff/A.dom.merge).toFixed(1),nonMergeBothTouchPct:(100*A.dom.nonMergeBothTouch/A.dom.nonMerge).toFixed(1)},
 boom:{...A.boom,avgDestroyed:(A.boom.destroyed/A.boom.n).toFixed(2),avgDestroyedSum:(A.boom.destroyedSum/A.boom.n).toFixed(1)},
 lowSpace:A.lowSpace,
 M:{boardsWithM:A.MboardCount,MpairBoardsPct:(100*A.Mpair/A.MboardCount).toFixed(1),MtilesPerMBoard:(A.Mtiles/A.MboardCount).toFixed(2),MonRimPct:(100*A.MinCornerEdge/A.Mtiles).toFixed(1)},
 death:{meanEmp:mean(A.death.map(d=>d.emp)).toFixed(2),meanGroups:mean(A.death.map(d=>d.groups)).toFixed(1),meanPairs:mean(A.death.map(d=>d.pairs)).toFixed(1),meanDistinct:mean(A.death.map(d=>d.distinct)).toFixed(1),meanEmp20Before:mean(A.death.map(d=>d.emp20)).toFixed(1),meanMinE:mean(A.death.map(d=>d.minE)).toFixed(2), lastDominoPct:(100*A.death.filter(d=>d.lastPiece==2).length/A.death.length).toFixed(0)},
};
// heat: fraction of value-v tiles by class, and density grid
const clsCount={corner:4,edge:12,inner:8,center:1};
o.heat={};
for(let v=1;v<=7;v++){const tot=A.cellVal[v].reduce((a,b)=>a+b,0); if(!tot)continue; const byc={}; A.cellVal[v].forEach((x,i)=>{const c=L.cls(i);byc[c]=(byc[c]||0)+x;});
 o.heat[L.CH[v]]={perBoard:(tot/A.moves).toFixed(2),relDensity:Object.fromEntries(Object.entries(byc).map(([k,x])=>[k,((x/tot)/(clsCount[k]/25)).toFixed(2)])),
  grid:[0,1,2,3,4].map(r=>[0,1,2,3,4].map(c=>(100*A.cellVal[v][r*5+c]/tot).toFixed(0).padStart(3)).join(''))};}
o.emptyGrid=[0,1,2,3,4].map(r=>[0,1,2,3,4].map(c=>(100*A.cellVal[0][r*5+c]/A.moves).toFixed(0).padStart(3)).join(''));
console.log(JSON.stringify(o,null,1));
