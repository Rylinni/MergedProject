const E=require('/home/user/MergedProject/replay/engine.js');
const fs=require('fs');
const CH='.123456M';
function show(b){let s=[];for(let r=0;r<5;r++){let l='';for(let c=0;c<5;c++)l+=CH[b[r*5+c]];s.push(l);}return s;}
function rc(i){return i<0?'-':`${Math.floor(i/5)},${i%5}`;}
// replay game; yields records
function replay(file){
  const g=JSON.parse(fs.readFileSync(file));
  const b=new Uint8Array(25); let un=E.START_UNLOCKED; const out=[]; let score=0;
  g.moves.forEach((m,i)=>{
    const [v0,v1,a,bb,pts]=m; const vals=v1?[v0,v1]:[v0];
    const before=Uint8Array.from(b); const unBefore=un;
    const r=E.apply(b,vals,a,bb,un); un=r.unlocked;
    if(r.points!==pts) throw new Error(`mismatch ${file} move ${i} ${r.points} vs ${pts}`);
    score+=pts;
    out.push({i,before,after:Uint8Array.from(b),vals,a,b:bb,pts,chain:r.chain,events:r.events,un:unBefore,unAfter:un});
  });
  return {g,recs:out,score};
}
function comps(b){ // connected groups
  const seen=new Uint8Array(25), res=[];
  for(let i=0;i<25;i++){ if(!b[i]||seen[i]) continue; const v=b[i], st=[i], cells=[i]; seen[i]=1;
    while(st.length){const x=st.pop(); for(const j of E.NB[x]) if(!seen[j]&&b[j]===v){seen[j]=1;cells.push(j);st.push(j);}}
    res.push({v,cells}); }
  return res;
}
function empties(b){let n=0;for(let i=0;i<25;i++)if(!b[i])n++;return n;}
function maxUn(un){let m=0;for(let v=1;v<=7;v++)if(un>>v&1)m=v;return m;}
function cls(i){const r=Math.floor(i/5),c=i%5;const e=(r==0||r==4)+(c==0||c==4);if(e==2)return 'corner';if(e==1)return 'edge';if(i==12)return 'center';return 'inner';}
module.exports={E,show,rc,replay,comps,empties,maxUn,cls,CH};
