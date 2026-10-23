// Offline equation typesetting only. Does not open a browser or fetch the preview.
const fs = require('fs');
const path = require('path');
const MathJax = require(process.env.RSNA_MATHJAX_PATH || '/tmp/rsna-blog-math/node_modules/mathjax');
const base = path.resolve(__dirname,'../_draft/_checks/rsna-series-20261022');
const inputs = JSON.parse(fs.readFileSync(path.join(base,'math-inputs.json'),'utf8'));
const out = path.join(base,'math-render');
fs.mkdirSync(out,{recursive:true});
(async()=>{
  const m=await MathJax.init({loader:{load:['input/tex','output/svg']},startup:{typeset:false},svg:{fontCache:'none',linebreaks:{inline:false}}});
  const result=[];
  for(let i=0;i<inputs.length;i++){
    const item=inputs[i];
    const node=await m.tex2svgPromise(item.tex,{display:item.display});
    const html=m.startup.adaptor.outerHTML(node);
    const fragments=html.match(/<svg\b[\s\S]*?<\/svg>/g) || [];
    const error=/data-mjx-error|data-mml-node="merror"/.test(html) || fragments.length!==1;
    const file=`equation-${String(i).padStart(3,'0')}.svg`;
    fs.writeFileSync(path.join(out,file),fragments[0] || '');
    result.push({...item, file, error});
  }
  const report={engine:'MathJax 4.1.3 offline SVG',scope:'Individual math fragments; no browser/page layout or CDN check',
                formulas:result.length, errors:result.filter(x=>x.error),result};
  fs.writeFileSync(path.join(base,'math-check.json'),JSON.stringify(report,null,2)+'\n');
  console.log(JSON.stringify({formulas:report.formulas,errors:report.errors.length}));
  if(report.errors.length)process.exitCode=1;
})().catch(e=>{console.error(e);process.exitCode=1});
