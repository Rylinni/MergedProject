const L=require('./lib.js'); const E=L.E;
const files=process.argv.slice(2);
const dec={}, decVal={}, decPair={n:0,makesPair:0,decBestPts:[]}; const destroyed=new Array(8).fill(0);
const traj=new Array(61).fill(0), trajN=new Array(61).fill(0);
const Mfire={avail:0,fired:0}; const Mplace={}; let low5=0, low5surv=0; const recov=[];
const takenPts=[];
for(const f of files){ const R=L.replay(f); const n=R.recs.length;
  let lowSeen=false;
  R.recs.forEach((m,t)=>{
    const neB=L.empties(m.before);
    let best=0,bestMove=null; for(const [a,bb] of E.legalMoves(m.before,m.vals)){const nb=Uint8Array.from(m.before);const r=E.apply(nb,m.vals,a,bb,m.un); if(r.points>best){best=r.points;bestMove=r;}}
    if(best>0){ const k=neB>=20?'20+':neB>=16?'16-19':neB>=13?'13-15':neB>=10?'10-12':neB>=7?'7-9':'<=6'; dec[k]=dec[k]||[0,0]; dec[k][1]++; 
      const mv=bestMove.events[0].value; decVal[mv]=decVal[mv]||[0,0]; decVal[mv][1]++;
      if(m.pts===0){dec[k][0]++; decVal[mv][0]++; decPair.n++; decPair.decBestPts.push(best);
        // does placement create a pair (touch equal) for some half
        const cells=m.b>=0?[[m.a,m.vals[0]],[m.b,m.vals[1]]]:[[m.a,m.vals[0]]]; if(cells.some(([c,v])=>E.NB[c].some(y=>y!==m.b&&y!==m.a&&m.before[y]===v)))decPair.makesPair++; } }
    for(const e of m.events) if(e.type==='boom') for(const x of e.cells){ /* value before boom: approximate by before board */ destroyed[m.before[x]]++; }
    const back=n-1-t; if(back<=60){traj[back]+=L.empties(m.after);trajN[back]++;}
    // M single with an M pair available
    if(m.vals.length===1&&m.vals[0]===7){ const cs=L.comps(m.before); const canFire=best>0; if(canFire){Mfire.avail++; if(m.pts>0)Mfire.fired++;} else {const c=L.cls(m.a); Mplace[c]=(Mplace[c]||0)+1;} }
    if(neB<=5 && !lowSeen){lowSeen=true; low5++; recov.push(n-t);}
  });
}
const cnt=o=>Object.entries(o).map(([k,v])=>`${k}:${v[0]}/${v[1]}=${(100*v[0]/v[1]).toFixed(1)}%`).join('  ');
console.log('decline by empties-before:',cnt(dec)); console.log('decline by merge value:',cnt(decVal));
const s=decPair.decBestPts.sort((a,b)=>a-b); console.log('declined: makes pair',(100*decPair.makesPair/decPair.n).toFixed(1)+'%','median forgone pts',s[s.length>>1],'p90',s[Math.floor(s.length*0.9)]);
console.log('destroyed by explosions (value: count)',destroyed.map((x,i)=>L.CH[i]+':'+x).join(' '));
console.log('empties after move, k moves before game end:',[60,40,30,20,15,10,7,5,3,2,1,0].map(k=>`T-${k}:${(traj[k]/trajN[k]).toFixed(1)}`).join(' '));
console.log('M single with explosion available: fired',Mfire.fired,'/',Mfire.avail); console.log('M single, no explosion avail, placed in',JSON.stringify(Mplace));
const r=recov.sort((a,b)=>a-b); console.log('games that ever hit <=5 empties',low5,'of',files.length,'moves remaining after first time: median',r[r.length>>1],'p10',r[Math.floor(r.length*.1)],'p90',r[Math.floor(r.length*.9)]);
