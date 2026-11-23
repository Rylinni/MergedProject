const L=require('./lib.js');
const files=process.argv.slice(2); const ages={}; let n=0; const lastAdded={};
for(const f of files){ const R=L.replay(f); R.recs.forEach((m,t)=>{ if(m.chain<6) return; n++;
  const merges=m.events.filter(e=>e.type==='merge'); let newest=null,newestAge=1e9;
  merges.forEach((e,k)=>{ const tiles=e.cells.filter(c=>c!==m.a&&c!==m.b); let rungAge=1e9;
    for(const c of tiles){ const v=m.before[c]; if(!v) continue; let a=0; while(t-a-1>=0 && R.recs[t-a-1].after[c]===v && R.recs[t-a-1].before[c]===v) a++; rungAge=Math.min(rungAge,a+1);} 
    const key=`step${k+1}(${L.CH[e.value]})`; (ages[key]=ages[key]||[]).push(rungAge); if(rungAge<newestAge){newestAge=rungAge;newest=key;} });
  lastAdded[newest]=(lastAdded[newest]||0)+1;
 });}
const med=a=>{const s=[...a].sort((x,y)=>x-y);return s[s.length>>1];};
console.log('6-step events',n); for(const [k,a] of Object.entries(ages)) console.log(k,'n',a.length,'median age (moves present before firing)',med(a),'p25',[...a].sort((x,y)=>x-y)[Math.floor(a.length*.25)]);
console.log('rung completed last:',JSON.stringify(lastAdded));
