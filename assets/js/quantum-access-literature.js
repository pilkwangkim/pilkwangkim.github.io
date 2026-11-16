/* Static-data literature explorer. Citation and similarity links remain distinct. */
(function (factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (typeof document !== 'undefined') {
    const start = () => document.querySelectorAll('[data-qal-root]').forEach(api.initialize);
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
    else start();
  }
})(function () {
  'use strict';
  const NS = 'http://www.w3.org/2000/svg';
  const VIEWS = ['citations', 'coupling', 'timeline'];
  const BRANCHES = ['all','foundations','mode-coupling','scattering','device-design'];
  const normalizeDoi = (doi) => String(doi || '').trim().replace(/^https?:\/\/(?:dx\.)?doi\.org\//i, '').toLowerCase();
  const pairKey = (a, b) => [a, b].sort().join('~');
  const referenceLabel = (key) => { const match = String(key || '').match(/(\d+[a-z]?)$/i); return match ? '[' + match[1] + ']' : ''; };

  function computeCouplings(nodes) {
    const seeds = nodes.filter((node) => node.seed);
    const result = [];
    seeds.forEach((a, i) => seeds.slice(i + 1).forEach((b) => {
      const refsA = new Set((a.reference_dois || []).map(normalizeDoi).filter(Boolean));
      const refsB = new Set((b.reference_dois || []).map(normalizeDoi).filter(Boolean));
      const shared = Array.from(refsA).filter((doi) => refsB.has(doi)).sort();
      result.push({ source:a.id, target:b.id, key:pairKey(a.id,b.id), shared_dois:shared,
        count:shared.length, cosine:refsA.size && refsB.size ? shared.length / Math.sqrt(refsA.size * refsB.size) : 0 });
    }));
    return result;
  }

  function preferredPair(paper, couplings) {
    const candidates = couplings.filter((pair) => pair.source === paper || pair.target === paper);
    candidates.sort((a,b) => b.count - a.count || b.cosine - a.cosine || a.key.localeCompare(b.key));
    const fallback = couplings.slice().sort((a,b) => b.count-a.count || b.cosine-a.cosine);
    return candidates[0] ? candidates[0].key : (fallback[0] && fallback[0].key);
  }

  function parseState(input, nodes, couplings) {
    const url = new URL(input, 'https://example.invalid/');
    const byId = new Map(nodes.map((node) => [node.id,node]));
    const requestedPaper = url.searchParams.get('paper');
    const defaultPaper = byId.has('venugopal2004') ? 'venugopal2004' : (nodes.find((node) => node.seed) || nodes[0]).id;
    let paper = byId.has(requestedPaper) ? requestedPaper : defaultPaper;
    const requestedView = url.searchParams.get('view');
    const requestedPair = url.searchParams.get('pair');
    const view = VIEWS.includes(requestedView) ? requestedView : 'citations';
    const context = url.searchParams.get('context') !== '0';
    if (view === 'citations' && !context && !byId.get(paper).seed) paper = defaultPaper;
    const branch = BRANCHES.includes(url.searchParams.get('branch')) ? url.searchParams.get('branch') : 'all';
    if(view==='citations' && branch!=='all' && !(byId.get(paper).branches||[]).includes(branch)) {
      const first=nodes.find((node)=>(context||node.seed)&&(node.branches||[]).includes(branch));
      if(first)paper=first.id;
    }
    return { paper, view, context, branch, scope:url.searchParams.get('scope')==='neighbors'?'neighbors':'overview',
      q:(url.searchParams.get('q')||'').slice(0,160),
      pair:couplings.some((pair) => pair.key === requestedPair) ? requestedPair : preferredPair(paper,couplings) };
  }

  function stateUrl(input, state) {
    const url = new URL(input, 'https://example.invalid/');
    url.searchParams.set('paper',state.paper);
    url.searchParams.set('view',state.view);
    url.searchParams.set('context',state.context ? '1' : '0');
    url.searchParams.set('scope',state.scope||'overview');
    url.searchParams.set('branch',state.branch||'all');
    if(state.q)url.searchParams.set('q',state.q);else url.searchParams.delete('q');
    if (state.view === 'coupling' && state.pair) url.searchParams.set('pair',state.pair);
    else url.searchParams.delete('pair');
    return url;
  }

  function selectCitationGraph(nodes,edges,state) {
    let visible=nodes.filter((node)=>(state.context||node.seed)&&(state.branch==='all'||(node.branches||[]).includes(state.branch)));
    if(!visible.some((node)=>node.id===state.paper))visible.push(nodes.find((node)=>node.id===state.paper));
    let ids=new Set(visible.filter(Boolean).map((node)=>node.id));
    if(state.scope!=='overview') {
      const neighbors=new Set([state.paper]);
      edges.forEach((edge)=>{if(edge.source===state.paper)neighbors.add(edge.target);if(edge.target===state.paper)neighbors.add(edge.source);});
      visible=visible.filter((node)=>neighbors.has(node.id));ids=new Set(visible.map((node)=>node.id));
    }
    const links=edges.filter((edge)=>ids.has(edge.source)&&ids.has(edge.target)&&(state.scope==='overview'||edge.source===state.paper||edge.target===state.paper));
    return {nodes:visible.filter(Boolean),edges:links};
  }

  function graphLayout(nodes,edges,state,width,height) {
    const positions=new Map();
    if(state.scope!=='overview') {
      const center={x:width/2,y:height/2-20},columns=width>=650?4:width>=440?3:2;
      positions.set(state.paper,center);
      const incoming=nodes.filter((node)=>node.id!==state.paper&&edges.some((edge)=>edge.source===node.id&&edge.target===state.paper)).sort((a,b)=>b.year-a.year||a.id.localeCompare(b.id));
      const outgoing=nodes.filter((node)=>node.id!==state.paper&&!incoming.includes(node)).sort((a,b)=>b.year-a.year||a.id.localeCompare(b.id));
      [incoming,outgoing].forEach((group,side)=>group.forEach((node,i)=>{
        const row=Math.floor(i/columns),rowCount=Math.min(columns,group.length-row*columns),column=i%columns;
        positions.set(node.id,{x:48+(column+.5)*(width-96)/rowCount,y:center.y+(side?1:-1)*(130+row*90)});
      }));
      return positions;
    }
    const ordered=nodes.slice().sort((a,b)=>a.id.localeCompare(b.id));
    const points=ordered.map((node,i)=>({id:node.id,x:width/2+Math.cos(i*2*Math.PI/ordered.length)*(width/2-50),y:height/2+Math.sin(i*2*Math.PI/ordered.length)*(height/2-65)}));
    const index=new Map(points.map((point,i)=>[point.id,i]));
    for(let iteration=0;iteration<180;iteration++) {
      const force=points.map(()=>({x:0,y:0}));
      for(let a=0;a<points.length;a++)for(let b=a+1;b<points.length;b++) {
        const dx=points[a].x-points[b].x,dy=points[a].y-points[b].y,d2=Math.max(25,dx*dx+dy*dy),distance=Math.sqrt(d2),strength=1600/d2;
        force[a].x+=dx/distance*strength;force[a].y+=dy/distance*strength;force[b].x-=dx/distance*strength;force[b].y-=dy/distance*strength;
      }
      edges.forEach((edge)=>{const ai=index.get(edge.source),bi=index.get(edge.target);if(ai===undefined||bi===undefined)return;const a=points[ai],b=points[bi],dx=b.x-a.x,dy=b.y-a.y,d=Math.hypot(dx,dy)||1,k=(d-100)*.006;force[ai].x+=dx/d*k;force[ai].y+=dy/d*k;force[bi].x-=dx/d*k;force[bi].y-=dy/d*k;});
      points.forEach((point,i)=>{point.x=Math.max(30,Math.min(width-30,point.x+force[i].x+(width/2-point.x)*.001));point.y=Math.max(30,Math.min(height-60,point.y+force[i].y+(height/2-point.y)*.001));});
    }
    points.forEach((point)=>positions.set(point.id,point));return positions;
  }

  function fitGraphTransform(positions,width,height) {
    const points=Array.from(positions.values());
    if(!points.length)return{scale:1,x:0,y:0};
    const xs=points.map((point)=>point.x),ys=points.map((point)=>point.y);
    const minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys);
    const scale=Math.max(.12,Math.min(1.2,(width-110)/Math.max(1,maxX-minX),(height-120)/Math.max(1,maxY-minY)));
    return{scale,x:width/2-(minX+maxX)/2*scale,y:(height-30)/2-(minY+maxY)/2*scale};
  }

  function initialize(root) {
    if (root.dataset.qalReady) return;
    root.dataset.qalReady = 'initializing';
    const find = (name) => root.querySelector('[data-qal-' + name + ']');
    let data;
    try { data = JSON.parse(find('data').textContent); } catch (error) {
      find('status').textContent = root.dataset.lang === 'ko' ? '논문 데이터를 읽지 못했습니다.' : 'The paper data could not be read.';
      root.dataset.qalReady = 'error'; return;
    }
    const ko = root.dataset.lang === 'ko';
    const t = (en,kr) => ko ? kr : en;
    const nodes = data.nodes;
    const byId = new Map(nodes.map((node) => [node.id,node]));
    const couplings = computeCouplings(nodes);
    let state = parseState(window.location.href,nodes,couplings);
    let drawnWidth = 0;
    let graphTransform={scale:1,x:0,y:0};
    let currentGraph=null,currentPositions=null,panMode=false,suppressGraphClick=false,searchTimer;
    const manualLayouts = new Map();
    const layoutKey = () => state.scope === 'overview' ? 'overview' : 'neighbors:' + state.paper;
    const graphWidth = () => Math.max(240,Math.round(root.getBoundingClientRect().width));

    function saveManualViewport(width,height) {
      const manual = manualLayouts.get(layoutKey());
      if(manual) manual.viewport = {scale:graphTransform.scale,x:graphTransform.x/width,y:graphTransform.y/height};
    }

    function manualLayoutForDrag(width,height) {
      const key = layoutKey();
      if(!manualLayouts.has(key)) manualLayouts.set(key,{
        positions:new Map(Array.from(currentPositions,([id,point]) => [id,{x:point.x/width,y:point.y/height}])),
        dragged:new Set(),viewport:{scale:graphTransform.scale,x:graphTransform.x/width,y:graphTransform.y/height}
      });
      return manualLayouts.get(key);
    }
    const paperSelect = find('paper');
    const pairSelect = find('pair');
    const inspector = find('inspector');
    const tabs = Array.from(root.querySelectorAll('[data-qal-view]'));
    const svgElement = (tag,attributes,text) => {
      const el = document.createElementNS(NS,tag);
      Object.entries(attributes || {}).forEach(([key,value]) => el.setAttribute(key,value));
      if (text !== undefined) el.textContent = text;
      return el;
    };
    const htmlElement = (tag,className,text) => {
      const el = document.createElement(tag);
      if (className) el.className = className;
      if (text !== undefined) el.textContent = text;
      if (/^h[1-6]$/.test(tag)) el.setAttribute('data-toc-skip','');
      return el;
    };
    const externalLink = (label,url) => {
      const a = htmlElement('a','',label); a.href = url; a.target = '_blank'; a.rel = 'noopener noreferrer'; return a;
    };
    const name = (node) => node.label.replace(/\s*·\s*\d{4}$/, '');
    const paperLabel = (node) => (node.seed ? 'P' + node.seed_index + ' · ' : '') + node.label;
    const localized = (node,field) => node.i18n && node.i18n[ko ? 'ko' : 'en'] && node.i18n[ko ? 'ko' : 'en'][field] || node[field] || '';
    const sourceByDoi = new Map(nodes.map((node) => [normalizeDoi(node.doi),node]));
    const pairByKey = new Map(couplings.map((pair) => [pair.key,pair]));

    // Paper previews are transient: hovering or focusing never changes the graph selection.
    const tooltip = htmlElement('div','qal-tooltip');
    tooltip.id = 'qal-paper-tooltip'; tooltip.dataset.qalTooltip = ''; tooltip.setAttribute('role','tooltip'); tooltip.hidden = true; root.append(tooltip);
    let tooltipTrigger = null, tooltipFromFocus = false;
    const firstParagraph = (value) => String(Array.isArray(value) ? value[0] || '' : value || '').split(/\n\s*\n/)[0].trim();

    function hideTooltip() {
      tooltip.hidden = true; tooltipFromFocus = false;
      if(tooltipTrigger) {
        const described = (tooltipTrigger.getAttribute('aria-describedby') || '').split(/\s+/).filter((id) => id && id !== tooltip.id);
        if(described.length) tooltipTrigger.setAttribute('aria-describedby',described.join(' ')); else tooltipTrigger.removeAttribute('aria-describedby');
      }
      tooltipTrigger = null;
    }

    function positionTooltip(trigger,pointer) {
      const margin = 12, gap = 24, rect = (trigger.querySelector('.qal-hit') || trigger).getBoundingClientRect();
      const anchorX = pointer ? pointer.x : rect.right;
      const exclusionTop = Math.min(rect.top,pointer ? pointer.y : rect.top);
      const exclusionBottom = Math.max(rect.bottom,pointer ? pointer.y : rect.bottom);
      tooltip.style.maxHeight = '';
      let box = tooltip.getBoundingClientRect();
      const below = Math.max(0,window.innerHeight - margin - exclusionBottom - gap);
      const above = Math.max(0,exclusionTop - gap - margin);
      const putBelow = below >= box.height || (above < box.height && below >= above);
      const available = putBelow ? below : above;
      tooltip.style.maxHeight = Math.floor(available) + 'px';
      box = tooltip.getBoundingClientRect();
      const left = Math.max(margin,Math.min(anchorX + gap,window.innerWidth - box.width - margin));
      const top = putBelow ? exclusionBottom + gap : exclusionTop - gap - box.height;
      tooltip.style.left = left + 'px'; tooltip.style.top = Math.max(margin,top) + 'px';
    }

    function showTooltip(node,trigger,event) {
      if(panMode || suppressGraphClick || drag && drag.moved) return;
      hideTooltip(); tooltipTrigger = trigger; tooltipFromFocus = !event;
      tooltip.replaceChildren(htmlElement('div','qal-tooltip-meta',paperLabel(node) + ' · ' + (node.journal || '')),htmlElement('h4','qal-tooltip-title',node.title));
      const summary = firstParagraph(localized(node,'summary'));
      if(summary) tooltip.append(htmlElement('p','qal-tooltip-summary',summary));
      const scope = firstParagraph(localized(node,'limitation'));
      if(scope) tooltip.append(htmlElement('p','qal-tooltip-scope',scope));
      const described = (trigger.getAttribute('aria-describedby') || '').split(/\s+/).filter(Boolean);
      if(!described.includes(tooltip.id)) described.push(tooltip.id); trigger.setAttribute('aria-describedby',described.join(' '));
      tooltip.style.left = '12px'; tooltip.style.top = '12px'; tooltip.hidden = false;
      positionTooltip(trigger,event && event.pointerType !== 'touch' ? {x:event.clientX,y:event.clientY} : null);
    }

    window.addEventListener('keydown',(event) => { if(event.key === 'Escape' && !tooltip.hidden) hideTooltip(); });
    window.addEventListener('resize',hideTooltip);
    window.addEventListener('wheel',(event) => { if(!(event.target instanceof Element && tooltip.contains(event.target))) hideTooltip(); },{capture:true,passive:true});
    window.addEventListener('scroll',(event) => {
      if(event.target instanceof Element && tooltip.contains(event.target)) return;
      // Native keyboard focus may scroll a node into view after its focus event.
      if(!tooltip.hidden && tooltipFromFocus && tooltipTrigger && tooltipTrigger.matches(':focus')) positionTooltip(tooltipTrigger);
      else hideTooltip();
    },{capture:true,passive:true});

    function commit(patch,history) {
      state = Object.assign({},state,patch);
      if (patch.paper && !patch.pair) state.pair = preferredPair(state.paper,couplings);
      if (!state.context && !byId.get(state.paper).seed && state.view === 'citations') state.paper = 'venugopal2004';
      if(state.view==='citations' && state.branch!=='all' && !(byId.get(state.paper).branches||[]).includes(state.branch)) {
        const first=nodes.find((node)=>(state.context||node.seed)&&(node.branches||[]).includes(state.branch));if(first)state.paper=first.id;
      }
      render();
      if (history !== false) {
        const url = stateUrl(window.location.href,state);
        if (url.href !== window.location.href) window.history.pushState(null,'',url);
      }
    }

    function choosePaper(id) {
      const patch = {paper:id};
      if(state.view==='coupling')patch.pair=state.pair;
      if (!byId.get(id).seed) patch.context = true;
      if(state.branch!=='all'&&!(byId.get(id).branches||[]).includes(state.branch))patch.branch='all';
      commit(patch);
    }

    function updateUrlLinks() {
      const shared = stateUrl(window.location.href,state);
      if (!shared.hash) shared.hash = 'quantum-access-literature';
      find('share').href = shared.href;
    }

    function openDetails() {
      const heading=inspector.querySelector('h3');if(!heading)return;heading.tabIndex=-1;
      const reduced=window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      heading.focus({preventScroll:true});inspector.scrollIntoView({block:'start',behavior:reduced?'auto':'smooth'});
    }

    function paperButton(node,className,showDetails) {
      const button = htmlElement('button',className,paperLabel(node));
      button.type = 'button'; button.dataset.qalChoose = node.id;
      button.setAttribute('aria-pressed',String(state.paper === node.id));
      button.addEventListener('click',(event) => {
        const timelineOrigin = Boolean(button.closest('[data-qal-panel="timeline"]'));
        choosePaper(node.id);
        if(showDetails||timelineOrigin){openDetails();return;}
        if (event.detail === 0) {
          const target = timelineOrigin ? root.querySelector('[data-qal-panel="timeline"] [data-qal-choose="' + node.id + '"]') : paperSelect;
          if (target) target.focus();
        }
      });
      return button;
    }

    function makeNode(node,point,selected,options) {
      options=options||{};const scale=options.scale||1,labelY=options.labelY||30;
      const g = svgElement('g',{transform:'translate(' + point.x + ' ' + point.y + ')',class:'qal-node' + (selected ? ' selected' : ''),
        role:'button',tabindex:options.focusable===false?'-1':'0','aria-label':paperLabel(node) + ': ' + node.title,'aria-pressed':String(selected),'data-paper':node.id});
      g.addEventListener('pointerenter',(event) => { if(event.pointerType !== 'touch') showTooltip(node,g,event); });
      g.addEventListener('pointermove',(event) => { if(tooltipTrigger === g && !tooltip.hidden && event.pointerType !== 'touch') positionTooltip(g,{x:event.clientX,y:event.clientY}); });
      g.addEventListener('pointerleave',() => { if(!g.matches(':focus')) hideTooltip(); });
      g.addEventListener('focus',() => showTooltip(node,g));
      g.addEventListener('blur',() => { if(tooltipTrigger === g) hideTooltip(); });
      g.append(node.seed ? svgElement('circle',{r:11,class:'qal-node-shape'}) : svgElement('path',{d:'M0 -12L12 0L0 12L-12 0Z',class:'qal-node-shape context'}));
      if(options.showLabel!==false){g.append(svgElement('text',{x:(options.labelX||0)/scale,y:labelY/scale,'text-anchor':'middle',class:'qal-node-label',style:'font-size:'+13.5/scale+'px;stroke-width:'+5/scale+'px'},name(node)));
      g.append(svgElement('text',{x:(options.labelX||0)/scale,y:(labelY+17)/scale,'text-anchor':'middle',class:'qal-node-year',style:'font-size:'+12/scale+'px;stroke-width:'+4/scale+'px'},(node.seed ? 'P' + node.seed_index + ' · ' : '') + node.year));}
      g.append(svgElement('circle',{r:22/scale,class:'qal-hit'}));
      g.addEventListener('click',() => {if(suppressGraphClick){suppressGraphClick=false;return;}choosePaper(node.id);});
      g.addEventListener('keydown',(event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); choosePaper(node.id); const activeNode=root.querySelector('[data-qal-panel="'+state.view+'"] [data-paper="' + node.id + '"]'); if(activeNode)activeNode.focus(); } });
      return g;
    }

    let positionCacheKey='',positionCache;
    function drawCitations(width) {
      hideTooltip();
      const svg = find('citation-graph'),height=560;
      currentGraph=selectCitationGraph(nodes,data.edges,state);
      const signature=width+'|'+state.scope+'|'+state.paper+'|'+currentGraph.nodes.map((node)=>node.id).join(',');
      const manual = manualLayouts.get(layoutKey());
      if(signature!==positionCacheKey){
        positionCache=graphLayout(currentGraph.nodes,currentGraph.edges,state,width,height);positionCacheKey=signature;
        if(manual) {
          positionCache.forEach((point,id)=>{
            const saved=manual.positions.get(id);
            if(saved) positionCache.set(id,{x:saved.x*width,y:saved.y*height});
            else manual.positions.set(id,{x:point.x/width,y:point.y/height});
          });
          graphTransform={scale:manual.viewport.scale,x:manual.viewport.x*width,y:manual.viewport.y*height};
        } else graphTransform=fitGraphTransform(positionCache,width,height);
      }
      currentPositions=positionCache;
      svg.setAttribute('viewBox','0 0 ' + width + ' ' + height); svg.setAttribute('width',width); svg.setAttribute('height',height); svg.replaceChildren();
      svg.dataset.panMode=String(panMode);svg.dataset.manualLayout=String(Boolean(manual));svg.dataset.manualCount=String(manual?manual.dragged.size:0);
      svg.append(svgElement('title',{},t('Verified direct citations','확인된 직접 인용 관계')));
      const visible=currentGraph.nodes,edges=currentGraph.edges.slice();
      const defs = svgElement('defs');
      ['muted','related'].forEach((kind) => { const marker = svgElement('marker',{id:'qal-arrow-' + kind,viewBox:'0 0 8 8',refX:7,refY:4,markerWidth:6,markerHeight:6,orient:'auto',markerUnits:'userSpaceOnUse'}); marker.append(svgElement('path',{d:'M0 0L8 4L0 8Z',class:'qal-arrow-' + kind})); defs.append(marker); });
      svg.append(defs);
      const plot=svgElement('g',{transform:'translate('+graphTransform.x+' '+graphTransform.y+') scale('+graphTransform.scale+')','data-qal-plot':''});svg.append(plot);
      edges.sort((a,b) => Number(a.source === state.paper || a.target === state.paper) - Number(b.source === state.paper || b.target === state.paper));
      edges.forEach((edge) => {
        const a = currentPositions.get(edge.source), b = currentPositions.get(edge.target), dx = b.x-a.x, dy = b.y-a.y, len = Math.hypot(dx,dy) || 1;
        const sx = a.x + dx/len*14, sy = a.y + dy/len*14, tx = b.x - dx/len*16, ty = b.y - dy/len*16;
        const bend = Math.min(44,Math.abs(dy)*.11) * (dx > 0 ? 1 : -1), related = edge.source === state.paper || edge.target === state.paper;
        plot.append(svgElement('path',{d:'M' + sx + ' ' + sy + 'Q' + ((sx+tx)/2+bend) + ' ' + ((sy+ty)/2) + ' ' + tx + ' ' + ty,class:'qal-edge' + (related ? ' related' : ''),'marker-end':'url(#qal-arrow-' + (related ? 'related':'muted') + ')','data-source':edge.source,'data-target':edge.target,'vector-effect':'non-scaling-stroke'}));
      });
      const neighborIds=new Set([state.paper]);data.edges.forEach((edge)=>{if(edge.source===state.paper)neighborIds.add(edge.target);if(edge.target===state.paper)neighborIds.add(edge.source);});
      const labels=new Map(),occupied=[];
      const projected=new Map(visible.map((node)=>{const p=currentPositions.get(node.id);return[node.id,{x:p.x*graphTransform.scale+graphTransform.x,y:p.y*graphTransform.scale+graphTransform.y}];}));
      visible.slice().sort((a,b)=>Number(b.id===state.paper)-Number(a.id===state.paper)||Number(b.seed)-Number(a.seed)||a.id.localeCompare(b.id)).forEach((node)=>{
        const point=projected.get(node.id);if(point.x<0||point.x>width||point.y<0||point.y>height)return;
        if(state.scope==='overview'&&!neighborIds.has(node.id))return;
        const labelWidth=Math.max(72,Math.min(140,name(node).length*7.5));
        const centerX=Math.max(labelWidth/2+5,Math.min(width-labelWidth/2-5,point.x));
        for(const offset of [30,-46]){
          const rect={x:centerX-labelWidth/2,y:point.y+offset-14,width:labelWidth,height:35};
          const overlaps=occupied.some((r)=>rect.x<r.x+r.width+5&&rect.x+rect.width+5>r.x&&rect.y<r.y+r.height+3&&rect.y+rect.height+3>r.y);
          const hitsNode=Array.from(projected.entries()).some(([id,p])=>id!==node.id&&p.x>=rect.x-8&&p.x<=rect.x+rect.width+8&&p.y>=rect.y-8&&p.y<=rect.y+rect.height+8);
          if(rect.y>=3&&rect.y+rect.height<=height-3&&!overlaps&&(!hitsNode||node.id===state.paper)){labels.set(node.id,{y:offset,x:centerX-point.x});occupied.push(rect);break;}
        }
      });
      visible.forEach((node)=>{const p=projected.get(node.id),label=labels.get(node.id),el=makeNode(node,currentPositions.get(node.id),state.paper===node.id,{scale:graphTransform.scale,showLabel:Boolean(label),labelY:label&&label.y,labelX:label&&label.x,focusable:p.x>=0&&p.x<=width&&p.y>=0&&p.y<=height});
        if(manual&&manual.dragged.has(node.id))el.dataset.manualPosition='true';plot.append(el);
      });
      saveManualViewport(width,height);
      find('citation-caption').textContent=t('Network contains ','표시된 논문 ')+visible.length+t(' papers and ','편 · 확인된 인용 ')+edges.length+t(' verified links. Labels are decluttered; use the paper list for every title. Layout distances are not calendar intervals.','개. 저자·연도 표시가 겹치면 일부를 생략합니다. 전체 제목은 논문 목록에서 확인할 수 있습니다. 논문 사이의 거리는 실제 시간 간격을 나타내지 않습니다.');
      if(visible.length===1)find('citation-caption').textContent+=' '+t('No direct neighbor was found within this collection and filter. Explore another branch or shared foundations.','현재 필터 조건에서는 직접 연결된 논문이 없습니다. 연구 주제를 바꾸거나 공통 선행연구를 확인해 보세요.');
    }

    function drawCoupling(width) {
      const box=find('coupling-graph'),pair=pairByKey.get(state.pair);box.replaceChildren();if(!pair)return;
      const a=byId.get(pair.source),b=byId.get(pair.target);
      const sourceCard=(node,letter)=>{const section=htmlElement('section','qal-foundation-paper');section.append(htmlElement('span','qal-meta',letter),paperButton(node,'qal-paper-chip',true),htmlElement('p','',node.title),htmlElement('span','qal-meta',t('Available reference DOIs: ','수집한 참고문헌 DOI: ')+(new Set((node.reference_dois||[]).map(normalizeDoi))).size));return section;};
      box.className='qal-foundation-bridge';box.append(sourceCard(a,'A'),htmlElement('span','qal-foundation-arrow','→'));
      const middle=htmlElement('section','qal-foundation-shared');middle.append(htmlElement('h3','',t('References cited by both','두 논문이 함께 인용한 선행연구')+' ('+pair.count+')'));
      const list=htmlElement('ol');
      pair.shared_dois.forEach((doi)=>{const known=sourceByDoi.get(doi),metadata=data.reference_metadata&&data.reference_metadata[doi],item=htmlElement('li');item.dataset.sharedDoi=doi;if(known){item.append(paperButton(known,'qal-paper-chip',true),htmlElement('p','',known.title));}else item.append(externalLink(metadata&&metadata.title?metadata.title:doi,metadata&&metadata.url?metadata.url:'https://doi.org/'+doi));list.append(item);});
      if(pair.count)middle.append(list);else middle.append(htmlElement('p','qal-note',t('No exact shared DOI in the available reference records.','수집한 참고문헌에서 공통 DOI를 찾지 못했습니다.')));
      box.append(middle,htmlElement('span','qal-foundation-arrow','←'),sourceCard(b,'B'));
      drawPairInspector();
    }

    function drawPairInspector() {
      const box=find('pair-inspector'),pair=pairByKey.get(state.pair); box.replaceChildren(); if(!pair)return;
      box.append(htmlElement('strong','',paperLabel(byId.get(pair.source))+' ↔ '+paperLabel(byId.get(pair.target))));
      box.append(htmlElement('div','qal-meta',t('Both papers draw on these references. Shared references do not imply identical conclusions or research quality.','두 논문이 함께 인용한 선행연구입니다. 같은 문헌을 인용해도 결론이 같다는 뜻은 아니며, 공통 문헌 수가 연구의 우열을 나타내지도 않습니다.')));
      const direct=data.edges.filter((edge)=>(edge.source===pair.source&&edge.target===pair.target)||(edge.source===pair.target&&edge.target===pair.source));
      if(direct.length){const p=htmlElement('p','qal-note',t('Verified direct citation: ','두 논문 사이에 확인된 직접 인용: '));direct.forEach((edge)=>p.append(externalLink(paperLabel(byId.get(edge.source))+' → '+paperLabel(byId.get(edge.target))+' '+referenceLabel(edge.reference_key),edge.evidence_url)));box.append(p);}
      else box.append(htmlElement('p','qal-note',t('No direct citation between this pair is verified in the collected records.','수집한 참고문헌에서는 두 논문이 서로를 인용한 기록을 찾지 못했습니다.')));
    }

    function drawStages() {
      const list=find('stages'); list.replaceChildren();
      data.stages.forEach((stage)=>{const selected=stage.ids.includes(state.paper),item=htmlElement('li','qal-stage'+(selected?' is-selected':'')),body=htmlElement('div'),translation=stage.i18n && stage.i18n[ko?'ko':'en'] || stage;
        item.append(htmlElement('span','qal-stage-period',stage.period));body.append(htmlElement('h3','',translation.label),htmlElement('p','',translation.text));
        if(selected)body.append(htmlElement('span','qal-stage-selected',t('Selected paper: ','선택한 논문: ')+paperLabel(byId.get(state.paper))));
        const chips=htmlElement('div','qal-paper-chips');stage.ids.forEach((id)=>{if(byId.has(id))chips.append(paperButton(byId.get(id),'qal-paper-chip',true));});body.append(chips);item.append(body);list.append(item);});
    }

    function drawInspector() {
      const node=byId.get(state.paper),rich=node.rich&&(node.rich[ko?'ko':'en']||node.rich);inspector.replaceChildren();
      inspector.append(htmlElement('div','qal-meta',(node.seed ? 'P'+node.seed_index+' · ':t('Context paper · ','관련 논문 · '))+node.year+' · '+(node.journal||'')));
      const title=htmlElement('h3','',node.title);title.id='qal-selected-paper-title';title.tabIndex=-1;inspector.append(title);
      inspector.append(htmlElement('div','qal-meta',(node.authors||[]).join(', ')));
      if(!rich){const dl=htmlElement('dl'); [[t('Contribution','요약'),localized(node,'summary')],[t('Scope','적용 범위'),localized(node,'limitation')]].forEach(([label,value])=>{if(value){const row=htmlElement('div');row.append(htmlElement('dt','',label),htmlElement('dd','',value));dl.append(row);}});inspector.append(dl);}
      const links=htmlElement('div','qal-links');if(node.url||node.doi)links.append(externalLink('DOI / '+t('article','원문'),node.url||'https://doi.org/'+node.doi));if(node.crossref_url)links.append(externalLink(t('Reference record','참고문헌 기록'),node.crossref_url));if(node.openalex_id)links.append(externalLink('OpenAlex',node.openalex_id));(node.sources||[]).forEach((source)=>{if(source.url)links.append(externalLink(source.label||t('Primary source','원문 근거'),source.url));});inspector.append(links);
      const localCounts=htmlElement('div','qal-local-counts');localCounts.append(htmlElement('span','',t('References within this collection: ','이 지도에 포함된 참고문헌: ')+data.edges.filter((edge)=>edge.source===node.id).length),htmlElement('span','',t('Citing papers within this collection: ','이 지도에 포함된 후속 인용 논문: ')+data.edges.filter((edge)=>edge.target===node.id).length));inspector.append(localCounts);
      const richContainer=htmlElement('div','qal-rich');
      if(rich)[['question',t('Research question','연구 질문')],['method',t('Method and comparison conditions','방법과 비교 조건')],['findings',t('Main findings','주요 결과')],['constriction',t('Implications for channel constriction','Channel constriction에 대한 시사점')],['limits',t('Limits and interpretation','적용 범위와 한계')],['reading_scope',t('Evidence checked','검토한 자료')]].forEach(([field,label])=>{
        if(!rich[field]||Array.isArray(rich[field])&&!rich[field].length)return;const section=htmlElement('section',field==='constriction'?'qal-constriction':'');section.dataset.richSection=field;section.append(htmlElement('h4','',label));if(Array.isArray(rich[field])){const list=htmlElement('ul');rich[field].forEach((paragraph)=>list.append(htmlElement('li','',paragraph)));section.append(list);}else String(rich[field]).split(/\n\s*\n/).forEach((paragraph)=>section.append(htmlElement('p','',paragraph)));richContainer.append(section);
      });if(richContainer.childElementCount)inspector.append(richContainer);
      const groups=htmlElement('div','qal-connections');
      [[true,t('Cites within this collection','이 논문이 인용한 선행연구')],[false,t('Cited by within this collection','이 논문을 인용한 후속연구')]].forEach(([outgoing,title])=>{
        const edges=data.edges.filter((edge)=>outgoing?edge.source===node.id:edge.target===node.id),group=htmlElement('div');group.append(htmlElement('h4','',title+' ('+edges.length+')'));
        if(!edges.length)group.append(htmlElement('span','qal-meta',t('No verified link within this collection.','이 지도에 포함된 논문 중에서는 확인된 인용 관계가 없습니다.')));
        else{const list=htmlElement('ul');edges.forEach((edge)=>{const other=byId.get(outgoing?edge.target:edge.source);if(!other)return;const item=htmlElement('li');item.append(paperButton(other,'qal-related-paper',true));if(edge.evidence_url)item.append(document.createTextNode(' · '),externalLink(referenceLabel(edge.reference_key)||t('record','기록'),edge.evidence_url));list.append(item);});group.append(list);}groups.append(group);
      });inspector.append(groups);
      inspector.append(htmlElement('div','qal-meta',t('OpenAlex cumulative citations: ','OpenAlex 누적 피인용 수: ')+(node.cited_by_count==null?t('unavailable','미확인'):node.cited_by_count)+' · '+data.snapshot_date+' · '+t('Counts reflect age and coverage, not a quality ranking.','논문이 발표된 시기와 OpenAlex의 수록 범위에 따라 달라지며, 연구의 우열을 나타내는 순위는 아닙니다.')));
    }

    function updatePaperOptions() {
      const query=state.q.trim().toLocaleLowerCase();let matches=0;
      Array.from(paperSelect.options).forEach((option)=>{
        const node=byId.get(option.value);if(!node)return;
        const allowed=(state.view!=='citations'||state.context||node.seed)&&(state.view!=='citations'||state.branch==='all'||(node.branches||[]).includes(state.branch));
        const text=[node.title,node.label,node.doi,(node.authors||[]).join(' ')].join(' ').toLocaleLowerCase();
        const match=allowed&&(!query||text.includes(query));if(match)matches++;
        option.hidden=!match&&option.value!==state.paper;option.disabled=option.hidden;
      });
      find('search-status').textContent=state.view==='citations'?t('Paper list matches: ','검색 결과: ')+matches:t('Paper details: ','선택 논문: ')+paperLabel(byId.get(state.paper));
      paperSelect.value=state.paper;
    }

    function render() {
      hideTooltip();
      paperSelect.value=state.paper;find('context').checked=state.context;
      find('search').value=state.q;find('branch').value=state.branch;find('scope').value=state.scope;
      root.querySelector('.qal-discovery').hidden=state.view!=='citations';root.querySelector('.qal-controls').hidden=state.view!=='citations';
      find('context-label').hidden=state.view!=='citations';find('context-key').hidden=!state.context;
      tabs.forEach((tab)=>{const selected=tab.dataset.qalView===state.view;tab.setAttribute('aria-selected',String(selected));tab.tabIndex=selected?0:-1;});
      root.querySelectorAll('[data-qal-panel]').forEach((panel)=>{panel.hidden=panel.dataset.qalPanel!==state.view;});
      updatePaperOptions();
      pairSelect.value=state.pair;
      const width=Math.max(240,Math.round(root.getBoundingClientRect().width));drawnWidth=width;
      if(state.view==='citations')drawCitations(width);else if(state.view==='coupling')drawCoupling(width);else drawStages();
      drawInspector();updateUrlLinks();
      const seeds=nodes.filter((node)=>node.seed).length,context=nodes.length-seeds;
      find('context-count').textContent=t('Include ','관련 논문 ')+context+t(' context papers','편 포함');
      find('status').textContent=t('Evidence snapshot: ','데이터 수집일: ')+data.snapshot_date+' · '+seeds+t(' seed + ','편의 기준 논문 + ')+context+t(' context papers','편의 관련 논문');
      root.dataset.qalView=state.view;root.dataset.qalPaper=state.paper;root.dataset.qalScope=state.scope;root.dataset.qalBranch=state.branch;
    }

    function redrawGraph() { if(state.view==='citations')drawCitations(Math.max(240,Math.round(root.getBoundingClientRect().width))); }
    function zoom(multiplier) {
      const width=Math.max(240,Math.round(root.getBoundingClientRect().width)),height=560,old=graphTransform.scale,next=Math.max(.12,Math.min(4,old*multiplier));
      graphTransform={scale:next,x:width/2-(width/2-graphTransform.x)*next/old,y:height/2-(height/2-graphTransform.y)*next/old};saveManualViewport(width,height);redrawGraph();
    }
    find('scope').addEventListener('change',()=>commit({scope:find('scope').value}));
    find('branch').addEventListener('change',()=>commit({branch:find('branch').value}));
    find('search').addEventListener('input',()=>{
      state.q=find('search').value.slice(0,160);updatePaperOptions();updateUrlLinks();window.clearTimeout(searchTimer);searchTimer=window.setTimeout(()=>window.history.replaceState(null,'',stateUrl(window.location.href,state)),250);
    });
    find('zoom-in').addEventListener('click',()=>zoom(1.3));find('zoom-out').addEventListener('click',()=>zoom(1/1.3));
    find('zoom-reset').addEventListener('click',()=>{manualLayouts.delete(layoutKey());positionCacheKey='';redrawGraph();});
    find('zoom-fit').addEventListener('click',()=>{
      if(!currentPositions||!currentPositions.size)return;graphTransform=fitGraphTransform(currentPositions,Math.max(240,Math.round(root.getBoundingClientRect().width)),560);saveManualViewport(graphWidth(),560);redrawGraph();
    });
    const citationSvg=find('citation-graph');
    find('pan').addEventListener('click',()=>{hideTooltip();panMode=!panMode;find('pan').setAttribute('aria-pressed',String(panMode));citationSvg.dataset.panMode=String(panMode);});
    let drag=null;
    function pointInPlot(clientX,clientY) {
      const plot=citationSvg.querySelector('[data-qal-plot]'),matrix=plot&&plot.getScreenCTM();
      if(!matrix)return null;
      const point=citationSvg.createSVGPoint();point.x=clientX;point.y=clientY;
      return point.matrixTransform(matrix.inverse());
    }
    citationSvg.addEventListener('pointerdown',(event)=>{
      if(event.button!==0 || drag)return;
      hideTooltip();suppressGraphClick=false;
      const element=event.target.closest('[data-paper]'),paper=element&&element.dataset.paper;
      if(paper&&!panMode) {
        const point=pointInPlot(event.clientX,event.clientY),position=currentPositions&&currentPositions.get(paper);if(!point||!position)return;
        drag={kind:'node',paper,id:event.pointerId,x:event.clientX,y:event.clientY,start:point,base:{x:position.x,y:position.y},moved:false};
      } else {
        if(event.pointerType==='touch'&&!panMode)return;
        drag={kind:'pan',id:event.pointerId,x:event.clientX,y:event.clientY,baseX:graphTransform.x,baseY:graphTransform.y,moved:false};
      }
    });
    citationSvg.addEventListener('pointermove',(event)=>{
      if(!drag||drag.id!==event.pointerId)return;
      const dx=event.clientX-drag.x,dy=event.clientY-drag.y;
      if(!drag.moved&&Math.hypot(dx,dy)<=5)return;
      if(!drag.moved){drag.moved=true;citationSvg.setPointerCapture(event.pointerId);}
      event.preventDefault();hideTooltip();citationSvg.dataset.dragging='true';
      if(drag.kind==='node') {
        const point=pointInPlot(event.clientX,event.clientY);if(!point)return;
        const width=graphWidth(),height=560,manual=manualLayoutForDrag(width,height);
        const position={x:drag.base.x+point.x-drag.start.x,y:drag.base.y+point.y-drag.start.y};
        currentPositions.set(drag.paper,position);manual.positions.set(drag.paper,{x:position.x/width,y:position.y/height});manual.dragged.add(drag.paper);
        citationSvg.dataset.dragNode=drag.paper;
      } else {
        const matrix=citationSvg.getScreenCTM();
        const ratioX=matrix?Math.hypot(matrix.a,matrix.b):1,ratioY=matrix?Math.hypot(matrix.c,matrix.d):1;
        graphTransform.x=drag.baseX+dx/(ratioX||1);graphTransform.y=drag.baseY+dy/(ratioY||1);
      }
      saveManualViewport(graphWidth(),560);redrawGraph();
    });
    const stopDrag=(event)=>{
      if(!drag||event.pointerId!==drag.id)return;
      const completed=drag;drag=null;citationSvg.dataset.dragging='false';delete citationSvg.dataset.dragNode;
      suppressGraphClick=event.type==='pointerup'&&completed.moved;
      if(citationSvg.hasPointerCapture(completed.id))citationSvg.releasePointerCapture(completed.id);
      if(completed.moved)hideTooltip();
    };
    citationSvg.addEventListener('pointerup',stopDrag);citationSvg.addEventListener('pointercancel',stopDrag);
    // Captured drags can retarget the final click to the SVG. Consume that click,
    // then clear on every new pointerdown so an ordinary next click is never lost.
    citationSvg.addEventListener('click',(event)=>{
      if(!suppressGraphClick||event.detail===0)return;
      suppressGraphClick=false;event.preventDefault();event.stopImmediatePropagation();
    },true);
    window.addEventListener('pointerup',stopDrag);window.addEventListener('pointercancel',stopDrag);
    citationSvg.addEventListener('keydown',(event)=>{if(event.target!==citationSvg)return;const steps={ArrowLeft:[35,0],ArrowRight:[-35,0],ArrowUp:[0,35],ArrowDown:[0,-35]};if(!steps[event.key])return;event.preventDefault();graphTransform.x+=steps[event.key][0];graphTransform.y+=steps[event.key][1];saveManualViewport(graphWidth(),560);redrawGraph();});

    couplings.forEach((pair)=>{const option=htmlElement('option','',paperLabel(byId.get(pair.source))+' ↔ '+paperLabel(byId.get(pair.target))+' · '+pair.count);option.value=pair.key;pairSelect.append(option);});
    paperSelect.addEventListener('change',()=>choosePaper(paperSelect.value));
    pairSelect.addEventListener('change',()=>{const pair=pairByKey.get(pairSelect.value);commit({pair:pair.key,paper:pair.source});});
    find('context').addEventListener('change',()=>commit({context:find('context').checked}));
    tabs.forEach((tab,index)=>{tab.addEventListener('click',()=>commit({view:tab.dataset.qalView}));tab.addEventListener('keydown',(event)=>{let next;if(event.key==='ArrowRight')next=(index+1)%tabs.length;else if(event.key==='ArrowLeft')next=(index+tabs.length-1)%tabs.length;else if(event.key==='Home')next=0;else if(event.key==='End')next=tabs.length-1;else return;event.preventDefault();commit({view:tabs[next].dataset.qalView});tabs[next].focus();});});
    window.addEventListener('popstate',()=>{window.clearTimeout(searchTimer);state=parseState(window.location.href,nodes,couplings);positionCacheKey='';render();});
    const resize=new ResizeObserver(()=>{if(Math.round(root.getBoundingClientRect().width)!==drawnWidth)render();});resize.observe(root);
    render();root.dataset.qalReady='ready';
  }
  return {normalizeDoi,pairKey,referenceLabel,computeCouplings,preferredPair,parseState,stateUrl,selectCitationGraph,graphLayout,fitGraphTransform,initialize};
});
