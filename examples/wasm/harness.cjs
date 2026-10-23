// Drive the *static* build exactly as index.html does: instantiate via the
// createLlama factory, write the model into MEMFS, callMain.
const fs = require('fs');
const createLlama = require('./llama-wasm.js');
const [, , modelPath, prompt = 'Once upon a time', n = '16'] = process.argv;

(async () => {
  const lines = [];
  const mod = await createLlama({
    noInitialRun: true,
    print: (t) => lines.push(t),
    printErr: (t) => { if (process.env.VERBOSE) console.error('[llama]', t); },
  });
  const bytes = new Uint8Array(fs.readFileSync(modelPath));
  console.log(`MEMFS: writing ${bytes.length} bytes to /model.gguf`);
  mod.FS.writeFile('/model.gguf', bytes);
  mod.callMain(['/model.gguf', prompt, n]);
  console.log('--- wasm stdout ---');
  console.log(lines.join('\n'));
})().catch((e) => { console.error('FAILED:', e); process.exit(1); });
