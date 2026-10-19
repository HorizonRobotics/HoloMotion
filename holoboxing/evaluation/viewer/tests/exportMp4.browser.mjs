import { build } from 'esbuild'
import { chromium } from 'playwright-core'
import { createServer } from 'node:http'
import { mkdtemp, writeFile, readFile, mkdir } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const directory = process.env.HOLOBOXING_VIDEO_TEST_OUTPUT || await mkdtemp(path.join(os.tmpdir(), 'holoboxing-video-'))
await mkdir(directory, { recursive: true })
await build({absWorkingDir:root, entryPoints:['tests/exportMp4.fixture.ts'],outfile:path.join(directory,'test.js'),bundle:true,platform:'browser',format:'esm',target:'es2020'})
const bundle = await readFile(path.join(directory,'test.js'))
const server = createServer((request,response)=> {
  response.setHeader('Content-Type',request.url==='/test.js'?'text/javascript':'text/html')
  response.end(request.url==='/test.js'?bundle:'<script type="module" src="/test.js"></script>')
})
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve))
const browser = await chromium.launch({executablePath:process.env.CHROME_BIN || '/usr/bin/google-chrome',headless:true,args:['--no-sandbox','--disable-dev-shm-usage','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader']})
try {
  const page=await browser.newPage();const errors=[]
  page.on('pageerror',error=>errors.push(String(error)))
  page.on('console',message=> {if(message.type()==='error')errors.push(message.text())})
  await page.goto(`http://127.0.0.1:${server.address().port}`)
  await page.waitForFunction(()=>typeof window.runExportTests==='function')
  page.setDefaultTimeout(120000)
  const result=await page.evaluate(()=>window.runExportTests())
  for(const [name,bytes] of Object.entries(result.files)) await writeFile(path.join(directory,name),Buffer.from(bytes))
  delete result.files
  if(errors.length)throw Error(JSON.stringify(errors))
  console.log(JSON.stringify({...result,directory},null,2))
} finally {await browser.close();await new Promise(resolve=>server.close(resolve))}
