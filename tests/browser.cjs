const { chromium } = require('playwright');
(async()=>{
const browser=await chromium.launch({headless:true,channel:'chrome',args:['--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream']});
const page=await browser.newPage();
const errors=[]; page.on('pageerror',e=>errors.push(e.message));
await page.goto('http://127.0.0.1:8765');
await page.locator('#consent').check();
await page.locator('#resume').setInputFiles({name:'mock.txt',mimeType:'text/plain',buffer:Buffer.from('Synthetic Python SQL FastAPI profile')});
await page.waitForFunction(()=>!document.querySelector('#connect').disabled);
for (const width of [1440,768,375]) {
 await page.setViewportSize({width,height:1000});
 if(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth)) throw Error('Horizontal overflow at '+width);

}
await page.route('**/assets/vendor/**',r=>r.abort());
await page.evaluate(()=>{
 let resolveMedia;
 window.pendingMedia=new Promise(r=>resolveMedia=r);
 window.resolveMedia=()=>{const track={stop(){window.trackStopped=true},addEventListener(){}};resolveMedia({getTracks:()=>[track]});};
 navigator.mediaDevices.getUserMedia=()=>window.pendingMedia;
});
await page.locator('#connect').click();
await page.locator('#disconnect').click();
await page.evaluate(()=>window.resolveMedia());
await page.waitForFunction(()=>window.trackStopped===true);
if(await page.locator('#device-state').textContent()!=='OFF')throw Error('Device cancellation failed');
await page.unroute('**/assets/vendor/**');
await page.reload();
await page.locator('#consent').check();
await page.locator('#resume').setInputFiles({name:'mock.txt',mimeType:'text/plain',buffer:Buffer.from('Python SQL')});
await page.waitForFunction(()=>!document.querySelector('#connect').disabled);
await page.evaluate(()=>{
 let turns=0;
 window.WebSocket=class {
  static OPEN=1;
  readyState=1; bufferedAmount=0;
  constructor(){this.number=turns++;setTimeout(()=>this.onopen?.(),5);}
  send(data){
   if(typeof data!=='string')return;
   if(data==='finish')setTimeout(()=>this.onmessage?.({data:JSON.stringify({final:true,confidence:.9,text:this.number%2===0?'Explain Python generators':'In Python a generator uses yield to return one value at a time through an iterator without allocating the entire sequence in memory.'})}),5);
   else setTimeout(()=>this.onmessage?.({data:JSON.stringify({ready:true})}),5);
  }
  close(){this.readyState=3;this.onclose?.();}
 };
});
await page.locator('#connect').click();
await page.waitForFunction(()=>!document.querySelector('#question-start').disabled,{},{timeout:60000});
for(let i=0;i<11;i++){
 await page.locator('#question-start').click();
 await page.locator('#finish').click();
 await page.locator('#answer-start').click();
 await page.locator('#finish').click();
 await page.waitForFunction(()=>!document.querySelector('#question-start').disabled);
}
if(await page.locator('#history details').count()!==10)throw Error('History retention failed');
await page.locator('#end-interview').click();
await page.locator('#summary-panel').waitFor({state:'visible'});
if(await page.locator('#summary-points').textContent()!=='10')throw Error('Summary retention failed');
if(await page.locator('#summary-skills').textContent().then(t=>!t.includes('Not asked')))throw Error('Unasked skill was evaluated');
await page.locator('#disconnect').click();
await page.goto('http://127.0.0.1:8765/assets/demo.html');
await page.locator('#synthetic').check();
await page.locator('#evaluate').click();
await page.locator('#result').waitFor({state:'visible'});
if(await page.locator('#tier').textContent()!=='LOW')throw Error('Simulator result failed');
if(errors.length)throw Error(errors.join('\n'));
console.log('Passed: resume upload, 3 responsive widths, pending-device cancellation, 11 synthetic Q&A turns, summary retention, simulator; no browser errors.');
await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
