'use strict';
const assert = require('node:assert/strict');
const lab = require('../assets/js/article-labs-search-dijkstra.js');
let checks = 0;
const eq = (a, b) => { assert.deepEqual(a, b); checks++; };

// Independent all-pairs oracle: no queue, parent or relaxation helper from the bundle.
function floydWarshall(graph) {
  const ids = graph.nodes.map(node => node.id), d = Object.fromEntries(ids.map(a => [a, Object.fromEntries(ids.map(b => [b, a === b ? 0 : Infinity]))]));
  for (const edge of graph.edges) if (!edge.closed) d[edge.a][edge.b] = d[edge.b][edge.a] = edge.minutes;
  for (const k of ids) for (const i of ids) for (const j of ids) d[i][j] = Math.min(d[i][j], d[i][k] + d[k][j]);
  return d;
}
for (const preset of lab.PRESETS) {
  const graph = lab.makeNetwork(preset), oracle = floydWarshall(graph), ids = graph.nodes.map(node => node.id);
  eq(lab.validateNetwork(graph), true);
  for (const source of ids) {
    const full = new lab.Dijkstra(graph, [source]).run();
    for (const goal of ids) {
      const search = new lab.Dijkstra(graph, [source], goal).run();
      eq(search.distance[goal], oracle[source][goal]); eq(full.distance[goal], oracle[source][goal]);
      eq(search.found, Number.isFinite(oracle[source][goal]));
      const path = search.path(goal);
      if (Number.isFinite(oracle[source][goal])) { eq(path[0], source); eq(path[path.length - 1], goal); eq(lab.manualRoute(graph, path).cost, oracle[source][goal]); }
      else eq(path, []);
    }
  }
  for (let a = 0; a < ids.length; a++) for (let b = a + 1; b < ids.length; b++) {
    const sources = [ids[b], ids[a]], multi = new lab.Dijkstra(graph, sources).run();
    for (const goal of ids) {
      const best = Math.min(...sources.map(source => oracle[source][goal]));
      eq(multi.distance[goal], best);
      if (Number.isFinite(best)) {
        const owner = sources.filter(source => oracle[source][goal] === best).sort()[0];
        eq(multi.owner[goal], owner); eq(multi.path(goal)[0], owner); eq(lab.manualRoute(graph, multi.path(goal)).cost, best);
      } else eq(multi.path(goal), []);
    }
  }
}
const normal = lab.makeNetwork();
eq(new lab.Dijkstra(normal, ['A'], 'H').run().distance.H, 10);
eq(new lab.Dijkstra(normal, ['A'], 'H').run().path('H'), ['A', 'D', 'E', 'F', 'G', 'H']);
eq(lab.manualRoute(normal, ['A', 'H']).cost, 18);
eq(lab.manualRoute(normal, ['A', 'B', 'C', 'H']).cost, 12);
eq(new lab.Dijkstra(lab.makeNetwork('traffic'), ['A'], 'H').run().distance.H, 12);
eq(new lab.Dijkstra(lab.makeNetwork('bridge'), ['A'], 'H').run().distance.H, 12);
eq(new lab.Dijkstra(lab.makeNetwork('isolated'), ['A'], 'H').run().found, false);
eq(new lab.Dijkstra(normal, ['A', 'G']).run().owner.E, 'A');
eq(new lab.Dijkstra(normal, ['A', 'G']).run().distance.E, 4);
eq(lab.manualRoute(normal, ['A', 'B', 'A']).cost, 8);
eq(lab.manualRoute(normal, ['A', 'C']).legal, false);
eq(lab.manualRoute(lab.makeNetwork('bridge'), ['E', 'F']).legal, false);
eq(lab.manualRoute(normal, ['A', 'Z']).legal, false);
eq(lab.manualRoute(normal, []).legal, false);
assert.throws(() => lab.makeNetwork('unknown')); assert.throws(() => new lab.Dijkstra(normal, [])); assert.throws(() => new lab.Dijkstra(normal, ['Z']));
assert.throws(() => lab.validateNetwork({ nodes: [{ id: 'A' }, { id: 'B' }], edges: [{ a: 'A', b: 'B', minutes: -1 }] }));

// Discovery is not acceptance: H=18 is generated on the first expansion.
const stepped = new lab.Dijkstra(normal, ['A'], 'H');
const first = stepped.step();
eq(first.entry.id, 'A'); eq(first.updates.find(update => update.to === 'H').candidate, 18); eq(stepped.found, false); eq(stepped.done, false);
eq(stepped.pops, 1); eq(stepped.settled.size, 1);
stepped.run(); eq(stepped.event.kind, 'goal'); eq(stepped.event.entry.distance, 10);
eq(stepped.entries().filter(entry => !entry.valid).map(entry => entry.distance), [12, 18]);
const exhausted = new lab.Dijkstra(normal, ['A']).run(); eq(exhausted.stale, 2); eq(exhausted.pops, exhausted.settled.size + exhausted.stale);
const trace = () => { const search = new lab.Dijkstra(normal, ['A', 'G']), events = []; while (!search.done) events.push(search.step()); return events; };
eq(trace(), trace());

// Exercise actual bilingual factories without a browser or timer dependency.
const factories = {};
global.window = { ArticleLabs: { register(id, factory) { factories[id] = factory; } } };
delete require.cache[require.resolve('../assets/js/article-labs-search-dijkstra.js')];
require('../assets/js/article-labs-search-dijkstra.js');
function node(tag, attrs = {}, children = []) {
  const element = { tag, attrs, children: Array.isArray(children) ? children : [children], textContent: '', listeners: {}, classList: { add() {} },
    append(...items) { this.children.push(...items); }, replaceChildren(...items) { this.children = items; },
    addEventListener(event, listener) { this.listeners[event] = listener; }, setAttribute(name, value) { this.attrs[name] = value; }
  };
  return element;
}
let renders = 0, actions = 0;
for (const factory of Object.values(factories)) for (const ko of [false, true]) {
  const controls = [], buttons = [], ctx = {
    root: { classList: { add() {} }, isConnected: true }, ko, text: (en, korean) => ko ? korean : en,
    el: node, svg: node, view: node('view'), ui: node('ui'), status: node('status'), metric: () => node('metric'), table: (headers, rows) => node('table', {}, [headers, rows]),
    control(en, korean, options) { const control = { value: String(options.value), options }; controls.push(control); return control; },
    button(en, korean, handler) { const button = node('button', {}, en); button.handler = handler; buttons.push(button); return button; },
    onChange(handler) { ctx.render = handler; handler(); }, onReset(handler) { ctx.reset = handler; }, checkpointState(config) { ctx.checkpoint = config; }
  };
  factory(ctx); renders++;
  for (const control of controls) {
    const value = control.value;
    for (const choice of control.options.options) { control.value = choice.value; ctx.render(); renders++; }
    control.value = value; ctx.render();
  }
  for (const button of buttons) { button.handler(); actions++; }
  ctx.reset(); ctx.render(); renders++;
  const board = ctx.view.children[0], network = board.children[0];
  const stops = network.children.filter(element => element.attrs.role === 'button'); eq(stops.length, 8);
  stops.find(element => element.attrs['aria-label'].startsWith('D,')).listeners.click(); actions++;
}
delete global.window;
console.log(`Dijkstra: ${checks} exact oracle/path/queue checks; ${renders} bilingual renders and ${actions} button/node actions passed (2 factories).`);
