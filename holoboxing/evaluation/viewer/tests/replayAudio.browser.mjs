import { build } from 'esbuild'
import { chromium } from 'playwright-core'
import { createServer } from 'node:http'
import { mkdtemp, readFile, mkdir } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const directory = await mkdtemp(path.join(os.tmpdir(), 'holoboxing-audio-'))
await build({absWorkingDir:root,entryPoints:['tests/replayAudio.fixture.tsx'],outfile:path.join(directory,'test.js'),bundle:true,format:'esm',target:'es2020',loader:{'.wasm':'file'},define:{'process.env.NODE_ENV':'"production"'},plugins:[{name:'browser-node-stubs',setup(builder){
 builder.onResolve({filter:/^(node:)?(module|fs|path|url|crypto)$/},args=>({path:args.path,namespace:'node-only'}))
 builder.onLoad({filter:/.*/,namespace:'node-only'},()=>({contents:'export default {}; export const createRequire = () => {}',loader:'js'}))
}}]})
const bundle=await readFile(path.join(directory,'test.js'))
const server=createServer(async(request,response)=>{
 try{
 if(['/hit.wav','/ko.wav'].includes(request.url) && process.env.HOLOBOXING_SOUND_PREVIEWS){
 response.setHeader('Content-Type','audio/wav');response.end(await readFile(path.join(process.env.HOLOBOXING_SOUND_PREVIEWS,request.url.slice(1))));return}
 response.setHeader('Content-Type',request.url==='/test.js'?'text/javascript':'text/html')
 response.end(request.url==='/test.js'?bundle:'<div id="root"></div><script type="module" src="/test.js"></script>')
 }catch(error){response.statusCode=500;response.end(String(error))}
})
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve))
const browser=await chromium.launch({executablePath:process.env.CHROME_BIN||'/usr/bin/google-chrome',headless:true,args:['--no-sandbox','--disable-dev-shm-usage']})
const assert=(condition,message)=>{if(!condition)throw Error(message)}
try{
 const page=await browser.newPage();const errors=[]
 page.on('pageerror',error=>errors.push(String(error)))
 await page.goto(`http://127.0.0.1:${server.address().port}`)
 await page.getByRole('button',{name:'播放',exact:true}).waitFor()
 const parity=process.env.HOLOBOXING_SOUND_PREVIEWS?await page.evaluate(()=>window.checkPreviewParity()):'not requested'
 const calls=()=>page.evaluate(()=>window.audioCalls)
 // Silent seek must not construct or play an effect.
 await page.getByRole('slider',{name:'回放进度'}).fill('60')
 assert((await calls()).length===0,'Seeking triggered audio')
 await page.getByRole('button',{name:'回放音效',exact:true}).click()
 await page.getByRole('button',{name:'播放',exact:true}).click()
 await page.waitForTimeout(120)
 assert((await calls()).length===0,'Muted playback scheduled audio')
 await page.getByRole('button',{name:'暂停',exact:true}).click()
 await page.getByRole('slider',{name:'回放进度'}).fill('0')
 await page.getByRole('button',{name:'回放音效',exact:true}).click()
 await page.getByRole('button',{name:'播放',exact:true}).click()
 await page.waitForTimeout(120)
 const first=await calls();assert(first.filter(x=>x.kind==='start').length===2,'Recorded hit and KO not scheduled')
 await page.getByRole('button',{name:'暂停',exact:true}).click()
 await page.waitForFunction(()=>window.audioCalls.filter(x=>x.kind==='stop').length===2)
 const paused=await calls();assert(paused.filter(x=>x.kind==='stop').length===2,'Pause left scheduled recordings active')
 await page.getByRole('slider',{name:'回放进度'}).fill('110')
 const afterSeek=await calls();assert(afterSeek.length===paused.length,'Seek replayed an effect')
 await page.getByRole('button',{name:'播放',exact:true}).click()
 await page.waitForTimeout(100)
 const resumed=await calls();assert(resumed.filter(x=>x.kind==='start').length===3,'Resume replayed an earlier hit')
 await page.getByRole('combobox',{name:'播放速度'}).selectOption('2')
 await page.getByRole('button',{name:'播放',exact:true}).waitFor()
 await page.waitForFunction(()=>window.audioCalls.filter(x=>x.kind==='stop').length===3)
 assert((await calls()).filter(x=>x.kind==='stop').length===3,'Speed change left audio scheduled')
 // A terminal KO must finish its recording; the player holds the final pose meanwhile.
 await page.getByRole('slider',{name:'回放进度'}).fill('145')
 await page.getByRole('button',{name:'播放',exact:true}).click()
 await page.waitForTimeout(200)
 await page.getByRole('button',{name:'暂停',exact:true}).waitFor()
 await page.getByRole('button',{name:'播放',exact:true}).waitFor({timeout:4000})
 assert(errors.length===0,JSON.stringify(errors))
 console.log(JSON.stringify({status:'PASS',previewMaximumErrors:parity,checked:['silent seek','mute','play','pause cancels cues','resume skips old hit','speed change cancels cues','terminal KO tail']},null,2))
}finally{await browser.close();await new Promise(resolve=>server.close(resolve))}
