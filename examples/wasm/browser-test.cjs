// Drive the demo in a real Chrome over the DevTools Protocol, and assert it
// actually generates text.
//
// The node harness exercises the wasm module but not the page: it never touches
// the DOM, the file input, or the factory callbacks index.html relies on. This
// does, so a broken button or a mis-wired callback fails here rather than in
// front of someone.
//
// No dependencies -- node 18+ has a global WebSocket, so CDP needs nothing
// installed. Start Chrome yourself first:
//
//   chrome --headless=new --remote-debugging-port=9222 \
//          --user-data-dir=$(mktemp -d) --no-first-run about:blank
//
//   node browser-test.cjs file:///abs/path/index.html /abs/path/model.gguf picker
//   node browser-test.cjs http://127.0.0.1:8000/index.html "" sample
//
// `picker` sets the <input type=file> (the file:// route, no server needed);
// `sample` clicks the bundled-model button (the hosted route, needs a fetch).
const http = require('http');

const PORT = Number(process.env.CDP_PORT || 9222);
const URL_ = process.argv[2];
const MODEL = process.argv[3];
const USE_SAMPLE = process.argv[4] === 'sample';

const getJSON = (path) => new Promise((res, rej) => {
  http.get({ host: '127.0.0.1', port: PORT, path }, (r) => {
    let b = ''; r.on('data', (d) => b += d); r.on('end', () => res(JSON.parse(b)));
  }).on('error', rej);
});

(async () => {
  // Find the about:blank page target Chrome opened.
  let targets = [];
  for (let i = 0; i < 40; i++) {
    try { targets = (await getJSON('/json/list')).filter((t) => t.type === 'page'); } catch {}
    if (targets.length) break;
    await new Promise((r) => setTimeout(r, 500));
  }
  if (!targets.length) throw new Error('no page target; is Chrome up?');

  const ws = new WebSocket(targets[0].webSocketDebuggerUrl);
  let id = 0;
  const pending = new Map();
  const console_ = [];
  const pageErrors = [];

  ws.addEventListener('message', (ev) => {
    const m = JSON.parse(ev.data);
    if (m.id && pending.has(m.id)) {
      const { resolve, reject } = pending.get(m.id);
      pending.delete(m.id);
      m.error ? reject(new Error(JSON.stringify(m.error))) : resolve(m.result);
    } else if (m.method === 'Runtime.consoleAPICalled') {
      console_.push(m.params.args.map((a) => a.value ?? a.description ?? '').join(' '));
    } else if (m.method === 'Runtime.exceptionThrown') {
      pageErrors.push(m.params.exceptionDetails.text + ' ' +
        (m.params.exceptionDetails.exception?.description || ''));
    }
  });
  const send = (method, params = {}) => new Promise((resolve, reject) => {
    const myId = ++id;
    pending.set(myId, { resolve, reject });
    ws.send(JSON.stringify({ id: myId, method, params }));
  });
  const evaluate = async (expr, awaitPromise = false) => {
    const r = await send('Runtime.evaluate', {
      expression: expr, returnByValue: true, awaitPromise,
    });
    if (r.exceptionDetails) throw new Error('eval threw: ' + r.exceptionDetails.text);
    return r.result.value;
  };

  await new Promise((r) => ws.addEventListener('open', r, { once: true }));
  await send('Page.enable');
  await send('Runtime.enable');
  await send('DOM.enable');

  console.log(`navigating to ${URL_}`);
  await send('Page.navigate', { url: URL_ });
  await new Promise((r) => {
    const h = (ev) => {
      if (JSON.parse(ev.data).method === 'Page.loadEventFired') {
        ws.removeEventListener('message', h); r();
      }
    };
    ws.addEventListener('message', h);
  });

  // Wait for the wasm runtime to come up (the factory promise resolving flips
  // #out to "ready").
  const waitFor = async (pred, label, ms = 180000) => {
    const t0 = Date.now();
    while (Date.now() - t0 < ms) {
      const text = await evaluate("document.getElementById('out').textContent");
      if (pred(text)) return text;
      await new Promise((r) => setTimeout(r, 400));
    }
    throw new Error(`timeout waiting for ${label}; #out = ` +
      JSON.stringify(await evaluate("document.getElementById('out').textContent")));
  };

  const ready = await waitFor((t) => /ready/i.test(t) || /failed/i.test(t), 'runtime init');
  console.log(`runtime: ${ready}`);
  if (/failed/i.test(ready)) throw new Error('runtime failed to start');

  if (USE_SAMPLE) {
    console.log('clicking "load the bundled sample"…');
    await evaluate("document.getElementById('sample').click()");
  } else {
    console.log(`setting the file input to ${MODEL}…`);
    const { root } = await send('DOM.getDocument');
    const { nodeId } = await send('DOM.querySelector', { nodeId: root.nodeId, selector: '#model' });
    await send('DOM.setFileInputFiles', { nodeId, files: [MODEL] });
    // setFileInputFiles does not fire `change`, so dispatch it.
    await evaluate("document.getElementById('model').dispatchEvent(new Event('change'))");
  }

  const mounted = await waitFor((t) => /mounted at|could not/i.test(t), 'model mount');
  console.log(`mount: ${mounted}`);
  if (/could not/i.test(mounted)) throw new Error('model did not mount');

  console.log('clicking Generate…');
  await evaluate("document.getElementById('run').click()");
  const result = await waitFor(
    (t) => !/^generating/.test(t) && t.trim() !== '' && !/mounted at/.test(t),
    'generation');

  console.log('\n=== page output ===');
  console.log(result);
  console.log('=== end ===\n');
  if (pageErrors.length) {
    console.log('page errors:'); pageErrors.forEach((e) => console.log('  ' + e));
  }
  const bad = console_.filter((l) => /error|abort|RuntimeError/i.test(l));
  if (bad.length) { console.log('suspicious console lines:'); bad.slice(0, 5).forEach((l) => console.log('  ' + l)); }

  if (!/Once upon a time/.test(result)) { console.error('FAIL: expected text not generated'); process.exit(1); }
  console.log('PASS');
  process.exit(0);
})().catch((e) => { console.error('ERROR:', e.message); process.exit(1); });
