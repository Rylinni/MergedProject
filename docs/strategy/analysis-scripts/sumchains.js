const L=require('./lib.js');
const a=[...require('./out/big2_chains.json').map(x=>({...x,src:'d2'})),...require('./out/big_chains.json').map(x=>({...x,src:'d1'}))];
const cnt=(arr,fn)=>{const o={};for(const x of arr){const k=fn(x);o[k]=(o[k]||0)+1;}return Object.entries(o).sort((x,y)=>y[1]-x[1]).map(([k,v])=>k+':'+v).join(' ');};
const mean=arr=>(arr.reduce((s,x)=>s+x,0)/arr.length).toFixed(2);
const med=arr=>{const s=[...arr].sort((x,y)=>x-y);return s[Math.floor(s.length/2)];};
for(const ch of [5,6]){ const e=a.filter(x=>x.chain===ch);
 console.log(`== ${ch}-step n=${e.length}`);
 console.log(' piece',cnt(e,x=>x.vals.split('+').map(c=>c).sort().join('+')));
 console.log(' boom',cnt(e,x=>x.boom));
 console.log(' sum mean',mean(e.map(x=>x.sum)),'max',Math.max(...e.map(x=>x.sum)),'pts max',Math.max(...e.map(x=>x.pts)));
 const bm=e.filter(x=>x.boom); console.log(' debris tiles mean',mean(bm.map(x=>x.nDebris)),'debris dist',cnt(bm,x=>x.nDebris));
 // debris points = sum - merged values. merged values: from ev string groups 'cell:Vxn'
 const deb=bm.map(x=>{let m=0; for(const g of x.ev.split(' ')){const mm=g.match(/:(\w)x(\d+)/); if(mm){const v='.123456M'.indexOf(mm[1]); m+=v*+mm[2];}} return x.sum-m;});
 console.log(' debris points mean',mean(deb),'max',Math.max(...deb),'dist',cnt(deb,x=>x));
 console.log(' boomCell class',cnt(bm,x=>{const [r,c]=x.boomCell.split(',').map(Number);return L.cls(r*5+c);}));
 console.log(' loadedAge (moves a >=5 setup existed before firing) mean',mean(e.map(x=>x.loadedAge)),'median',med(e.map(x=>x.loadedAge)),'dist',cnt(e,x=>x.loadedAge>=10?'10+':x.loadedAge));
 console.log(' #outs (pieces giving >=5) at fire time mean',mean(e.map(x=>Object.keys(x.outs).length)));
 console.log(' #outs giving 6',cnt(e,x=>Object.values(x.outs).filter(v=>v[0]==6).length));
 console.log(' empties before',mean(e.map(x=>x.emp)),'dist',cnt(e,x=>x.emp));
 console.log(' moveFrac',mean(e.map(x=>+x.moveFrac)),' gameMoves med',med(e.map(x=>x.gameMoves)));
 console.log(' first merge',cnt(e,x=>x.ev.split(' ')[0].split(':')[1]));
 console.log(' partner merge size',cnt(e,x=>{const vs=x.vals.split('+');const g=x.ev.split(' ').map(s=>s.split(':')[1]).filter(Boolean);return g.join(',');}));
 console.log(' by src',cnt(e,x=>x.src));
}
