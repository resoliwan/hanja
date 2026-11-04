const fs = require('fs');
const vm = require('vm');
const path = require('path');
const base = path.resolve(__dirname, '../vendor/hanzi_lookup/web_demo');
global.self = global;
vm.runInThisContext(fs.readFileSync(path.join(base, 'hanzi_lookup.js'), 'utf8'));
(async () => {
  await wasm_bindgen(fs.readFileSync(path.join(base, 'hanzi_lookup_bg.wasm')));
  const samples = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
  const results = [];
  for (const s of samples) {
    const start = performance.now();
    try {
      const candidates = JSON.parse(wasm_bindgen.lookup(s.strokes, 5));
      results.push({id:s.id, candidates:candidates.map(c=>c.hanzi), ms:performance.now()-start, error:null});
    } catch (e) {
      results.push({id:s.id, candidates:[], ms:performance.now()-start, error:String(e)});
      // A Rust panic poisons thread-local state; recreate the instance for the next sample.
      await wasm_bindgen(fs.readFileSync(path.join(base, 'hanzi_lookup_bg.wasm')));
    }
  }
  console.log(JSON.stringify(results));
})();
