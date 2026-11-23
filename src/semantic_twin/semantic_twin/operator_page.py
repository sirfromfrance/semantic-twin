"""Operator page of the semantic twin (one self-contained HTML document).

Live mode: served by operator_node, polls /twin.json and sends queries to /query.
Static mode: tools/render_twin.py replaces __TWIN__ with a saved twin export.
"""

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Semantic twin &ndash; operator</title>
<style>
:root{--bg:#f6f5f2;--panel:#fff;--ink:#222;--mute:#6b6b6b;--line:#d9d6cf;--ruin:#b9b2a4;--debris:#d8cdb4;
--fire:#d9480f;--smoke:#9aa0a6;--veh:#4c6ef5;--tree:#5c940d;--free:#d3e3c4;--route:#1c7ed6;--tall:#8d8678;--up:#2b8a3e;--ly:#c92a2a;--unk:#868e96;--chg:#e67700}
@media (prefers-color-scheme:dark){:root{--bg:#161616;--panel:#1f1f1f;--ink:#eee;--mute:#a0a0a0;--line:#333;
--ruin:#4a463f;--debris:#3a362d;--fire:#ff6b35;--smoke:#6c7278;--veh:#748ffc;--tree:#8ce99a;--free:#2c3a26;--route:#4dabf7;--tall:#6b655a;--up:#51cf66;--ly:#ff6b6b;--chg:#ffa94d}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.4 system-ui,sans-serif}
main{display:grid;grid-template-columns:minmax(0,1fr) 360px;gap:16px;padding:16px;max-width:1280px;margin:auto}
@media (max-width:860px){main{grid-template-columns:1fr}}
h1{font-size:18px;margin:0 0 4px}.sub{color:var(--mute);margin:0 0 8px;font-size:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px}
svg{width:100%;height:auto;display:block;background:var(--bg);border-radius:6px}
.legend{display:flex;flex-wrap:wrap;gap:10px;margin-top:8px;font-size:12px;color:var(--mute)}
.legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;vertical-align:-1px}
.row{padding:8px 4px;border-top:1px solid var(--line);cursor:pointer}.row:first-child{border-top:0}
.row b{font-weight:600}.tag{font-size:11px;padding:1px 6px;border-radius:9px;border:1px solid var(--line);margin-left:6px}
.high{color:var(--ly);border-color:var(--ly)}.medium{color:#e67700;border-color:#e67700}.low{color:var(--mute)}
.why{color:var(--mute);font-size:12px}.sel{background:rgba(120,120,120,.14)}
.sum{display:flex;gap:14px;margin:0 0 8px;flex-wrap:wrap}.sum div{font-size:12px;color:var(--mute)}.sum b{display:block;font-size:20px;color:var(--ink)}
.tabs{display:flex;gap:4px;margin:6px 0}.tabs button,.bar button{font:inherit;font-size:13px;padding:4px 10px;border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:6px;cursor:pointer}
.tabs button.on,.bar button.on{background:var(--ink);color:var(--bg)}
.filters{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0 8px;font-size:12px}
.filters select,.filters label{font:inherit;font-size:12px;color:var(--ink)}.filters select{background:var(--panel);border:1px solid var(--line);border-radius:5px;padding:2px 4px}
.bar{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin-bottom:8px;font-size:12px;color:var(--mute)}
#status{font-size:12px}.ok{color:var(--up)}.bad{color:var(--ly)}
#hint{font-size:12px;color:var(--route);min-height:16px;margin-top:6px}
.list{max-height:70vh;overflow:auto}
</style></head><body><main>
<section class="card"><h1>Semantic twin &ndash; operator view</h1>
<p class="sub"><span id="meta"></span> &middot; <span id="status"></span></p>
<div class="bar"><button id="bChg" class="on">change regions</button><button id="bTrail" class="on">trails</button>
<button id="bGrid" class="on">sensed map</button><span id="mode"></span></div>
<svg id="map" role="img" aria-label="Top-down map of the scene with detected people"></svg>
<div id="hint"></div>
<div class="legend">
<span><i style="background:var(--up)"></i>upright / survivor</span><span><i style="background:var(--ly)"></i>lying / casualty</span>
<span><i style="background:var(--unk)"></i>unknown</span><span><i style="background:var(--fire)"></i>fire (seen) / dashed (description only)</span>
<span><i style="background:var(--smoke)"></i>smoke</span><span><i style="background:var(--free)"></i>free ground (seen)</span><span><i style="background:var(--ruin)"></i>ruin / obstacle</span><span><i style="background:var(--tall)"></i>tall structure</span>
<span><i style="background:var(--chg)"></i>map change</span><span><i style="background:var(--route)"></i>route (select a person)</span></div></section>
<aside class="card"><div class="sum" id="sum"></div>
<div class="tabs"><button data-t="people" class="on">People</button><button data-t="changes">Changes</button><button data-t="structs">Structures</button></div>
<div class="filters" id="filters">
<select id="fLevel"><option value="all">any priority</option><option value="high">high</option><option value="medium">medium+</option></select>
<select id="fStatus"><option value="all">any status</option><option value="present">present</option><option value="not_found_in_latest">not found</option></select>
<select id="fPosture"><option value="all">any posture</option><option value="lying">lying</option><option value="upright">upright</option></select>
<select id="fEvac"><option value="all">any route</option><option value="unreachable">no route</option><option value="ok">has route</option></select>
<label><input type="checkbox" id="fChanged"> changed only</label></div>
<div class="list" id="list"></div></aside>
</main>
<script>
const T0 = __TWIN__;
const NS='http://www.w3.org/2000/svg', svg=document.getElementById('map');
let T=null, sel=null, tab='people', focus=null, lastRaw='', pickGoal=false;
const show={chg:true,trail:true,grid:true};
const $=id=>document.getElementById(id);
function el(n,a,p=svg){const e=document.createElementNS(NS,n);for(const k in a)e.setAttribute(k,a[k]);p.appendChild(e);return e}
const KIND={building_ruin:'building ruin',vehicle_or_large_object:'vehicle / large object',small_obstacle:'small obstacle',tall_object:'tall object',vegetation:'vegetation',spike_artifact:'noise?'};
const col=p=>p.posture==='upright'?'var(--up)':p.posture==='lying'?'var(--ly)':'var(--unk)';
const changed=p=>(p.status&&p.status!=='present')||(p.changes&&p.changes.length>0);
const LV={low:0,medium:1,high:2};
function passes(p){
 const l=$('fLevel').value,s=$('fStatus').value,po=$('fPosture').value,e=$('fEvac').value;
 if(l!=='all'&&LV[p.triage.level]<LV[l])return false;
 if(s!=='all'&&p.status!==s)return false;
 if(po!=='all'&&p.posture!==po)return false;
 if(e==='unreachable'&&!(p.evacuation&&p.evacuation.status!=='ok'))return false;
 if(e==='ok'&&!(p.evacuation&&p.evacuation.status==='ok'))return false;
 if($('fChanged').checked&&!changed(p))return false;
 return true}
let X,Y,K=20,bounds;
function computeBounds(){
 const S=T.static,G=S.grid,pts=[];
 S.debris_fields.forEach(d=>pts.push([d.x-d.radius,d.y-d.radius],[d.x+d.radius,d.y+d.radius]));
 Object.values(S.building_cells).forEach(c=>c.forEach(k=>pts.push(k,[k[0]+1,k[1]+1])));
 if(G)G.rows.forEach((r,j)=>{for(let i=0;i<r.length;i++)if(r[i]!=='.'&&r[i]!=='_'){const x=G.x0+i*G.resolution,y=G.y0+j*G.resolution;pts.push([x,y],[x+G.resolution,y+G.resolution])}});
 [...S.hazards,...S.vehicles,...S.trees,...T.people].forEach(o=>pts.push([o.x,o.y]));
 if(T.grid_changes&&T.grid_changes.regions)T.grid_changes.regions.forEach(r=>pts.push([r.x-2,r.y-2],[r.x+2,r.y+2]));
 (T.people.flatMap(p=>(p.history||[])).concat(T.people.flatMap(p=>(p.evacuation&&p.evacuation.path)||[]).map(q=>({x:q[0],y:q[1]})))).forEach(o=>pts.push([o.x,o.y]));
 if(!pts.length)pts.push([-10,-10],[10,10]);
 const x0=Math.min(...pts.map(p=>p[0]))-2,x1=Math.max(...pts.map(p=>p[0]))+2;
 const y0=Math.min(...pts.map(p=>p[1]))-2,y1=Math.max(...pts.map(p=>p[1]))+2;
 return {x0,x1,y0,y1}}
function drawMap(){
 svg.innerHTML='';const S=T.static,G=S.grid;
 bounds=computeBounds();const {x0,x1,y0,y1}=bounds;
 svg.setAttribute('viewBox',`0 0 ${(x1-x0)*K} ${(y1-y0)*K}`);
 X=x=>(x-x0)*K;Y=y=>(y1-y)*K;
 for(let g=Math.ceil(x0/5)*5;g<=x1;g+=5){el('line',{x1:X(g),x2:X(g),y1:0,y2:(y1-y0)*K,stroke:'var(--line)','stroke-width':.5});el('text',{x:X(g)+2,y:10,fill:'var(--mute)','font-size':9}).textContent=g}
 for(let g=Math.ceil(y0/5)*5;g<=y1;g+=5){el('line',{y1:Y(g),y2:Y(g),x1:0,x2:(x1-x0)*K,stroke:'var(--line)','stroke-width':.5});el('text',{x:2,y:Y(g)-2,fill:'var(--mute)','font-size':9}).textContent=g}
 if(G&&show.grid){const gg=el('g',{});const c2={'_':['var(--free)',.5],'o':['var(--debris)',.9],'#':['var(--ruin)',.95],'T':['var(--tall)',1]};
  G.rows.forEach((r,j)=>{for(let i=0;i<r.length;i++){const c=c2[r[i]];if(!c)continue;
   el('rect',{x:X(G.x0+i*G.resolution),y:Y(G.y0+(j+1)*G.resolution),width:G.resolution*K+.4,height:G.resolution*K+.4,fill:c[0],opacity:c[1]},gg)}});
  S.structures.forEach(s=>{const t=el('text',{x:X(s.x),y:Y(s.y),fill:'var(--ink)','font-size':10,'text-anchor':'middle',style:'cursor:pointer'});
   t.textContent=(KIND[s.kind]||s.name)+' '+s.max_height+' m';t.onclick=()=>{focus={x:s.x,y:s.y,r:Math.sqrt(s.area_m2/3.14)+1};tab='structs';render()}})}
 S.debris_fields.forEach(d=>el('circle',{cx:X(d.x),cy:Y(d.y),r:d.radius*K,fill:'var(--debris)',opacity:.7}));
 S.hazards.forEach(h=>{const f=h.type==='fire',seen=h.source==='sensed'||h.source==='prior+sensed';
  const c=el('circle',{cx:X(h.x),cy:Y(h.y),r:f?7:11,fill:f?(seen?'var(--fire)':'none'):'var(--smoke)',stroke:f?'var(--fire)':'none','stroke-width':2,'stroke-dasharray':f&&!seen?'3 2':'',opacity:f?.95:.45});
  el('title',{},c).textContent=f?`fire (${h.source==='prior'?'from the scene description, not seen by the camera':h.source==='sensed'?'seen by the camera only':'scene description, confirmed by the camera'})`:'smoke (scene description)'});
 const gc=T.grid_changes;
 if(show.chg&&gc&&gc.regions)gc.regions.forEach(r=>{const rad=Math.max(Math.sqrt(r.area_m2/3.14)+.4,.9);
  el('circle',{cx:X(r.x),cy:Y(r.y),r:rad*K,fill:'none',stroke:'var(--chg)','stroke-width':2,'stroke-dasharray':'5 3'}).appendChild(document.createElementNS(NS,'title')).textContent=`${r.kind}: ${r.area_m2} m2, ${r.height_before_m} -> ${r.height_after_m} m`});
 if(show.trail)T.people.forEach(p=>{if(p.history&&p.history.length>1&&passes(p)){
  el('polyline',{points:p.history.map(h=>X(h.x)+','+Y(h.y)).join(' '),fill:'none',stroke:col(p),'stroke-width':1.5,opacity:.6});
  p.history.slice(0,-1).forEach(h=>el('circle',{cx:X(h.x),cy:Y(h.y),r:3,fill:'none',stroke:col(p),'stroke-width':1.5,opacity:.7}))}});
 if(focus)el('circle',{cx:X(focus.x),cy:Y(focus.y),r:(focus.r||1.5)*K,fill:'none',stroke:'var(--route)','stroke-width':3,opacity:.9});
 T.people.forEach(p=>{const ok=passes(p),g=el('g',{style:'cursor:pointer',opacity:ok?1:.22});
  el('circle',{cx:X(p.x),cy:Y(p.y),r:p.id===sel?11:8,fill:col(p),stroke:p.status==='not_found_in_latest'?'var(--mute)':'#fff','stroke-width':2,'stroke-dasharray':p.status==='not_found_in_latest'?'3 2':''},g);
  if(p.triage.level==='high')el('circle',{cx:X(p.x),cy:Y(p.y),r:14,fill:'none',stroke:'var(--ly)','stroke-width':2},g);
  el('text',{x:X(p.x)+12,y:Y(p.y)+4,fill:'var(--ink)','font-size':11,'font-weight':600},g).textContent='#'+p.id;
  el('title',{},g).textContent=`#${p.id} ${p.posture}/${p.role} at (${p.x}, ${p.y}), ${p.status||''}`;
  g.onclick=ev=>{ev.stopPropagation();sel=p.id;pickGoal=false;render()}});
 if(sel!==null)drawRoute();
}
let customRoute=null;
function drawRoute(){
 const p=T.people.find(q=>q.id===sel);if(!p)return;
 const e=customRoute&&customRoute.person===sel?customRoute.route:p.evacuation;
 if(e&&e.status==='ok'&&e.path&&e.path.length>1){const g=el('g',{});
  el('polyline',{points:e.path.map(q=>X(q[0])+','+Y(q[1])).join(' '),fill:'none',stroke:'var(--route)','stroke-width':3,'stroke-dasharray':'7 4','stroke-linecap':'round'},g);
  el('circle',{cx:X(e.goal[0]),cy:Y(e.goal[1]),r:8,fill:'none',stroke:'var(--route)','stroke-width':3},g);
  el('text',{x:X(e.goal[0])+11,y:Y(e.goal[1])+4,fill:'var(--ink)','font-size':11,'font-weight':600},g).textContent=(customRoute&&customRoute.person===sel)?'chosen goal':'safe area'}}
function evText(p){const e=p.evacuation;if(!e)return'';
 if(e.status==='ok')return `<div class="why">evacuation: ${e.length_m} m, about ${e.walk_time_s} s &middot; closest fire ${e.min_fire_distance_m} m${e.rubble_cells?` &middot; ${e.rubble_cells} rubble cells`:''}</div>`+(e.verified===false?`<div class="why" style="color:var(--ly)">&#9888; starts inside a ruin: route unverified, check before use</div>`:'');
 return `<div class="why" style="color:var(--ly)">evacuation: ${e.status==='unreachable'?'no passable route on the sensed map':e.status.replace(/_/g,' ')}</div>`}
function whereText(p){const r=p.relations,ins=r.inside_structure,ns=r.nearest_structure||{};
 return ins?`in ${KIND[r.inside_kind]||ins}`:(ns.name?`${ns.distance_m} m from ${KIND[ns.kind]||ns.name}`:'no structure nearby')}
function peopleList(){const box=document.createElement('div');
 const ps=T.people.filter(passes).sort((a,b)=>(b.triage.score-a.triage.score)||(a.id-b.id));
 if(!ps.length)box.innerHTML='<div class="why">No people match the filters.</div>';
 ps.forEach(p=>{const d=document.createElement('div');d.className='row'+(p.id===sel?' sel':'');
  const h=(p.history||[]).map(h=>`${h.flight.replace('flight_','F')}: (${h.x}, ${h.y}) ${h.posture}`).join(' &rarr; ');
  d.innerHTML=`<b>#${p.id}</b> ${p.posture} / ${p.role}<span class="tag ${p.triage.level}">${p.triage.level}</span>${p.status==='not_found_in_latest'?'<span class="tag">not found</span>':''}
  <div class="why">(${p.x}, ${p.y}) &middot; ${whereText(p)} &middot; fire ${p.relations.nearest_fire_m} m &middot; open ground ${p.relations.open_ground_m} m</div>
  <div class="why">${p.triage.reasons.join('; ')||'no risk factors'}</div>${evText(p)}
  ${changed(p)?`<div class="why">${[p.status&&p.status!=='present'?p.status.replace(/_/g,' '):'',...(p.changes||[])].filter(Boolean).join('; ')}</div>`:''}
  ${p.id===sel?`<div class="why">history: ${h}</div><div class="why"><button id="goalBtn">route to a point on the map</button>${customRoute&&customRoute.person===sel?' <button id="goalClr">clear</button>':''}</div>`:''}`;
  d.onclick=()=>{sel=p.id;customRoute=customRoute&&customRoute.person===p.id?customRoute:null;pickGoal=false;render()};box.appendChild(d)});
 return box}
function changesList(){const box=document.createElement('div');const gc=T.grid_changes;
 const ch=T.people.filter(changed);
 box.innerHTML=`<div class="why">People (${ch.length})</div>`;
 ch.forEach(p=>{const d=document.createElement('div');d.className='row';
  d.innerHTML=`<b>#${p.id}</b> ${p.posture}<div class="why">${[p.status&&p.status!=='present'?p.status.replace(/_/g,' '):'',...(p.changes||[])].filter(Boolean).join('; ')}</div>`;
  d.onclick=()=>{sel=p.id;focus={x:p.x,y:p.y,r:1.5};render()};box.appendChild(d)});
 if(!ch.length)box.innerHTML+='<div class="why">none</div>';
 const h=document.createElement('div');
 if(!gc||!gc.regions){h.innerHTML='<div class="why" style="margin-top:10px">Map: needs two flights to compare.</div>';box.appendChild(h);return box}
 h.innerHTML=`<div class="why" style="margin-top:10px">Map, ${gc.flight} vs earlier: ${gc.changed_cells} of ${gc.compared_cells} compared cells changed (${gc.added} added, ${gc.removed} removed), ${gc.regions.length} regions. Not verified: check a region before acting on it.</div>`;
 box.appendChild(h);
 gc.regions.forEach((r,i)=>{const d=document.createElement('div');d.className='row';
  d.innerHTML=`<b>region ${i+1}</b> ${r.kind}<div class="why">(${r.x}, ${r.y}) &middot; ${r.area_m2} m&sup2; &middot; height ${r.height_before_m} &rarr; ${r.height_after_m} m${r.was_open_ground!=null?` &middot; ${Math.round(r.was_open_ground*100)}% was open ground`:''}</div>`;
  d.onclick=()=>{focus={x:r.x,y:r.y,r:Math.sqrt(r.area_m2/3.14)+1};render()};box.appendChild(d)});
 return box}
function hazardLine(){const c=T.static.hazard_check;if(!c)return '';
 if(!c.fire_colour_available)return 'Fire: scene description only (no colour data in this twin).';
 return `Fire: ${c.prior_fires} in the scene description, ${c.confirmed} confirmed by the camera, ${c.not_seen_in_observed_area} not seen where the drone looked, ${c.area_not_observed} in unobserved areas; ${c.sensed_only} seen by the camera only.`}
function structsList(){const box=document.createElement('div');
 const hl=document.createElement('div');hl.className='why';hl.style.marginBottom='6px';hl.textContent=hazardLine();box.appendChild(hl);
 T.static.structures.forEach(s=>{const d=document.createElement('div');d.className='row';
  const inside=T.people.filter(p=>p.relations.inside_structure===s.name);
  d.innerHTML=`<b>${KIND[s.kind]||s.name}</b> <span class="why">${s.name}</span><div class="why">${s.area_m2} m&sup2; &middot; up to ${s.max_height} m &middot; centre (${s.x}, ${s.y})</div>
  <div class="why">${s.kind_reason||''}${inside.length?` &middot; people inside: ${inside.map(p=>'#'+p.id).join(', ')}`:''}</div>`;
  d.onclick=()=>{focus={x:s.x,y:s.y,r:Math.sqrt(s.area_m2/3.14)+1};render()};box.appendChild(d)});
 if(!T.static.structures.length)box.innerHTML='<div class="why">No structures yet.</div>';
 return box}
function render(){
 if(!T)return;
 const m=T.meta,sm=T.summary;
 $('meta').textContent=`flight ${m.current_flight} (${m.flights.length} in the twin) | updated ${m.updated} | map: ${m.world}`;
 $('sum').innerHTML=`<div><b>${sm.people}</b>people</div><div><b>${sm.casualties}</b>lying</div><div><b>${sm.high_priority}</b>high priority</div><div><b>${sm.present_in_latest}</b>seen this flight</div><div><b>${sm.observed_cells}</b>map cells</div>`;
 document.querySelectorAll('.tabs button').forEach(b=>b.classList.toggle('on',b.dataset.t===tab));
 $('filters').style.display=tab==='people'?'flex':'none';
 drawMap();
 const list=$('list');list.innerHTML='';
 list.appendChild(tab==='people'?peopleList():tab==='changes'?changesList():structsList());
 const gb=$('goalBtn');if(gb)gb.onclick=ev=>{ev.stopPropagation();pickGoal=true;$('hint').textContent=T0?'Live mode is needed for custom routes.':'Click a point on the map.'};
 const gcl=$('goalClr');if(gcl)gcl.onclick=ev=>{ev.stopPropagation();customRoute=null;render()};
 if(!pickGoal)$('hint').textContent='';
}
svg.addEventListener('click',ev=>{
 if(!pickGoal||sel===null||T0)return;
 const pt=svg.createSVGPoint();pt.x=ev.clientX;pt.y=ev.clientY;const q=pt.matrixTransform(svg.getScreenCTM().inverse());
 const gx=q.x/K+bounds.x0,gy=bounds.y1-q.y/K;pickGoal=false;$('hint').textContent=`Planning route to (${gx.toFixed(1)}, ${gy.toFixed(1)}) ...`;
 fetch('/query?q='+encodeURIComponent(JSON.stringify({op:'route',person:sel,to:[gx,gy]}))).then(r=>r.json()).then(r=>{
  if(!r.ok){$('hint').textContent='Query failed: '+(r.error||'');return}
  customRoute={person:sel,route:r.route};const rt=r.route;
  $('hint').textContent=rt.status==='ok'?`Route to (${gx.toFixed(1)}, ${gy.toFixed(1)}): ${rt.length_m} m, about ${rt.walk_time_s} s, closest fire ${rt.min_fire_distance_m} m.`
   :`No passable route to (${gx.toFixed(1)}, ${gy.toFixed(1)}): ${(rt.notes||[rt.status]).join('; ')}`;
  drawMap()}).catch(e=>{$('hint').textContent='Query failed: '+e})});
document.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>{tab=b.dataset.t;focus=null;render()});
['fLevel','fStatus','fPosture','fEvac','fChanged'].forEach(i=>$(i).onchange=render);
[['bChg','chg'],['bTrail','trail'],['bGrid','grid']].forEach(([id,k])=>$(id).onclick=()=>{show[k]=!show[k];$(id).classList.toggle('on',show[k]);render()});
function poll(){
 fetch('/twin.json',{cache:'no-store'}).then(r=>{if(!r.ok)throw new Error(r.status);return r.text()}).then(txt=>{
  $('status').textContent='live';$('status').className='ok';
  if(txt!==lastRaw){lastRaw=txt;T=JSON.parse(txt);if(sel!==null&&!T.people.some(p=>p.id===sel))sel=null;render()}
 }).catch(()=>{$('status').textContent='no connection to the twin';$('status').className='bad'})}
if(T0){T=T0;$('status').textContent='saved twin (static)';$('mode').textContent='';render()}
else{$('mode').textContent='updates every 2 s';poll();setInterval(poll,2000)}
</script></body></html>
"""
