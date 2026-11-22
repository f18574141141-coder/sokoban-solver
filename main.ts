// Sokoban solver: room decomposition + push-optimal A* per room.
//
// The levels are a hallway with 1-4 rooms off it; a box pushed out of a
// room's door can never come back, so each room is an independent problem
// and the level optimum is the sum of the rooms' optima. We detect the
// hallway (the player's row), split the board into room components, solve
// each room push-optimally with A* (Hungarian assignment heuristic over
// box-to-goal walking distances, static dead squares, 2x2 blocks and
// frozen boxes), then walk the player between rooms through the hallway
// and concatenate the move strings. A whole-level A* is the fallback when
// decomposition fails. The search budget is counted in expansions, never
// in wall-clock time, so two runs write the same bytes.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Search budgets, counted in state expansions (deterministic across runs).
const ROOM_EXPANSIONS = 600000;
const LEVEL_EXPANSIONS = 1500000;
const TOTAL_EXPANSIONS = 6000000;

const DIRS: [number, number, string][] = [
  [0, -1, 'u'],
  [0, 1, 'd'],
  [-1, 0, 'l'],
  [1, 0, 'r'],
];

let spent = 0; // expansions spent over the whole run, shared by every search

interface Board {
  w: number;
  h: number;
  wall: Uint8Array; // blocks the player and boxes (1 = wall)
  boxWall: Uint8Array; // blocks boxes only (1 = a box may never sit here)
  goals: number[];
  goalSet: Uint8Array;
  dead: Uint8Array; // static dead squares for boxes
  distGoal: number[][]; // distGoal[g][cell]: walking distance goal g -> cell, -1 = unreachable
  floor: number[]; // cells a box may sit on
}

function parseLevel(text: string): { w: number; h: number; wall: Uint8Array; goals: number[]; boxes: number[]; player: number } {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = 0;
  rows.forEach((row, y) => {
    for (let x = 0; x < w; x++) {
      const ch = row[x] ?? '#';
      const i = y * w + x;
      if (ch === '#') wall[i] = 1;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

// Breadth-first walking distances from `from` over cells where `open(i)` holds.
// Returns an Int32Array of distances (-1 = unreachable).
function bfsDist(w: number, h: number, open: (i: number) => boolean, from: number): Int32Array {
  const dist = new Int32Array(w * h).fill(-1);
  if (!open(from)) return dist;
  const q: number[] = [from];
  dist[from] = 0;
  for (let qi = 0; qi < q.length; qi++) {
    const c = q[qi]!;
    const cx = c % w;
    const cy = (c / w) | 0;
    if (cx > 0 && open(c - 1) && dist[c - 1] < 0) { dist[c - 1] = dist[c]! + 1; q.push(c - 1); }
    if (cx < w - 1 && open(c + 1) && dist[c + 1] < 0) { dist[c + 1] = dist[c]! + 1; q.push(c + 1); }
    if (cy > 0 && open(c - w) && dist[c - w] < 0) { dist[c - w] = dist[c]! + 1; q.push(c - w); }
    if (cy < h - 1 && open(c + w) && dist[c + w] < 0) { dist[c + w] = dist[c]! + 1; q.push(c + w); }
  }
  return dist;
}

// A walking path (letters) from `from` to `to` avoiding walls and `blocked`
// cells, or null. Deterministic: neighbours in u, d, l, r order.
function bfsPath(w: number, h: number, wall: Uint8Array, blocked: Uint8Array, from: number, to: number): string | null {
  if (from === to) return '';
  const prev = new Int32Array(w * h).fill(-2);
  const prevCh: string[] = new Array(w * h).fill('');
  prev[from] = -1;
  const q: number[] = [from];
  for (let qi = 0; qi < q.length; qi++) {
    const c = q[qi]!;
    for (const [dx, dy, ch] of DIRS) {
      const nx = (c % w) + dx;
      const ny = ((c / w) | 0) + dy;
      if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
      const n = ny * w + nx;
      if (wall[n] || blocked[n] || prev[n] !== -2) continue;
      prev[n] = c;
      prevCh[n] = ch;
      if (n === to) {
        let s = '';
        for (let t = n; t !== from; t = prev[t]!) s = prevCh[t] + s;
        return s;
      }
      q.push(n);
    }
  }
  return null;
}

// Static dead squares for boxes: wall runs with no goal, then a fixpoint
// (a square is dead when every push from it lands on a dead square or is
// impossible). Sound: it only ever marks squares a box can never leave
// towards a goal. Note the two kinds of blocked cell: `solid` blocks a box
// (a wall, or a square a box may never sit on), while only a wall blocks
// the player - the player may stand on a box-only square, e.g. the cell
// outside a room's door, and push a box from there.
function computeDead(b: Board): Uint8Array {
  const { w, h, wall, boxWall, goalSet } = b;
  const solid = (i: number): boolean => i < 0 || i >= w * h || wall[i] === 1 || boxWall[i] === 1;
  const pWall = (i: number): boolean => i < 0 || i >= w * h || wall[i] === 1;
  const dead = new Uint8Array(w * h);
  const isFloor = (i: number): boolean => i >= 0 && i < w * h && wall[i] === 0 && boxWall[i] === 0;
  // Wall runs: a maximal run of floor cells hugging one wall, bounded by
  // walls at both ends, holding no goal: a box there can only slide along
  // the run and never reach a goal.
  const markRun = (cells: number[], bounded: boolean): void => {
    if (!bounded || cells.length === 0 || cells.some((c) => goalSet[c])) return;
    for (const c of cells) dead[c] = 1;
  };
  for (let y = 0; y < h; y++) {
    for (const dy of [-1, 1]) {
      let run: number[] = [];
      let leftBounded = false;
      for (let x = 0; x <= w; x++) {
        const c = y * w + x;
        const hugs = x < w && isFloor(c) && pWall(c + dy * w);
        if (hugs) {
          if (run.length === 0) leftBounded = x === 0 || !isFloor(c - 1);
          run.push(c);
        } else {
          markRun(run, leftBounded && (x >= w || !isFloor(c)));
          run = [];
        }
      }
    }
  }
  for (let x = 0; x < w; x++) {
    for (const dx of [-1, 1]) {
      let run: number[] = [];
      let topBounded = false;
      for (let y = 0; y <= h; y++) {
        const c = y * w + x;
        const hugs = y < h && isFloor(c) && pWall(c + dx);
        if (hugs) {
          if (run.length === 0) topBounded = y === 0 || !isFloor(c - w);
          run.push(c);
        } else {
          markRun(run, topBounded && (y >= h || !isFloor(c)));
          run = [];
        }
      }
    }
  }
  // Fixpoint: a non-goal square is dead when no push from it can reach a
  // live square (the push is impossible, or lands on a dead square). This
  // also marks corners: every push from a corner is impossible.
  let changed = true;
  while (changed) {
    changed = false;
    for (let c = 0; c < w * h; c++) {
      if (!isFloor(c) || goalSet[c] || dead[c]) continue;
      let allDead = true;
      for (const [dx, dy] of DIRS) {
        const dest = c + dy * w + dx;
        const side = c - dy * w - dx;
        const destOk = !solid(dest) && !dead[dest];
        const sideOk = !pWall(side);
        if (destOk && sideOk) { allDead = false; break; }
      }
      if (allDead) { dead[c] = 1; changed = true; }
    }
  }
  return dead;
}

function makeBoard(w: number, h: number, wall: Uint8Array, boxWall: Uint8Array, goals: number[]): Board {
  const goalSet = new Uint8Array(w * h);
  for (const g of goals) goalSet[g] = 1;
  const floor: number[] = [];
  for (let i = 0; i < w * h; i++) if (wall[i] === 0 && boxWall[i] === 0) floor.push(i);
  const b: Board = { w, h, wall, boxWall, goals: [...goals].sort((a, c) => a - c), goalSet, dead: new Uint8Array(w * h), distGoal: [], floor };
  b.dead = computeDead(b);
  b.distGoal = b.goals.map((g) => bfsDist(w, h, (i) => wall[i] === 0 && boxWall[i] === 0, g));
  return b;
}

// Minimum-cost assignment of boxes to goals (Hungarian, O(n^3)); Infinity
// when some box cannot reach any goal. Admissible: one push moves one box
// one square, so the summed assignment distance lower-bounds the pushes left.
function hungarian(b: Board, boxes: number[]): number {
  const n = boxes.length;
  if (n === 0) return 0;
  const M = 1e9;
  const cost: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let j = 0; j < n; j++) {
      const d = b.distGoal[j]![boxes[i]!]!;
      row.push(d < 0 ? M : d);
    }
    cost.push(row);
  }
  const u = new Array(n + 1).fill(0);
  const v = new Array(n + 1).fill(0);
  const p = new Array(n + 1).fill(0);
  const way = new Array(n + 1).fill(0);
  for (let i = 1; i <= n; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = new Array(n + 1).fill(Infinity);
    const used = new Array(n + 1).fill(false);
    do {
      used[j0] = true;
      const i0 = p[j0]!;
      let delta = Infinity;
      let j1 = 0;
      for (let j = 1; j <= n; j++) {
        if (used[j]) continue;
        const cur = cost[i0 - 1]![j - 1]! - u[i0]! - v[j]!;
        if (cur < minv[j]!) { minv[j] = cur; way[j] = j0; }
        if (minv[j]! < delta) { delta = minv[j]!; j1 = j; }
      }
      for (let j = 0; j <= n; j++) {
        if (used[j]) { u[p[j]!] += delta; v[j] -= delta; }
        else minv[j]! -= delta;
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do {
      const j1 = way[j0]!;
      p[j0] = p[j1]!;
      j0 = j1;
    } while (j0 !== 0);
  }
  let total = 0;
  for (let j = 1; j <= n; j++) {
    const c = cost[p[j]! - 1]![j - 1]!;
    if (c >= M / 2) return Infinity;
    total += c;
  }
  return total;
}

// True when some box off a goal can never move again: a 2x2 block of walls
// and boxes, or a box frozen by walls and frozen boxes in every direction.
function isDeadlocked(b: Board, boxes: number[]): boolean {
  const { w, h, wall, boxWall, goalSet } = b;
  const occ = new Uint8Array(w * h);
  for (const x of boxes) occ[x] = 1;
  // A 2x2 block is solid when every cell is a wall or holds a box. (A
  // box-only square does not count: the player may stand on it, so a box
  // beside it can still be pushed.)
  const solid = (i: number): boolean => i < 0 || i >= w * h || wall[i] === 1 || occ[i] === 1;
  for (const x of boxes) {
    if (goalSet[x]) continue;
    if (b.dead[x]) return true;
    // 2x2 blocks containing x: all four cells wall-or-box.
    const cx = x % w;
    const cy = (x / w) | 0;
    const blocks: number[][] = [];
    if (cx + 1 < w && cy + 1 < h) blocks.push([x, x + 1, x + w, x + w + 1]);
    if (cx - 1 >= 0 && cy + 1 < h) blocks.push([x - 1, x, x + w - 1, x + w]);
    if (cx + 1 < w && cy - 1 >= 0) blocks.push([x - w, x - w + 1, x, x + 1]);
    if (cx - 1 >= 0 && cy - 1 >= 0) blocks.push([x - w - 1, x - w, x - 1, x]);
    for (const blk of blocks) {
      if (blk.every((c) => solid(c))) return true;
    }
  }
  // Frozen boxes, to a fixpoint: a box is frozen when every push is blocked
  // by a wall, a box-only cell, or a frozen box, on the destination or on
  // the player's side.
  const frozen = new Uint8Array(w * h);
  let changed = true;
  while (changed) {
    changed = false;
    for (const x of boxes) {
      if (frozen[x]) continue;
      let stuck = true;
      for (const [dx, dy] of DIRS) {
        const dest = x + dy * w + dx;
        const side = x - dy * w - dx;
        const destBlocked = dest < 0 || dest >= w * h || wall[dest] === 1 || boxWall[dest] === 1 || (occ[dest] === 1 && frozen[dest] === 1);
        const sideBlocked = side < 0 || side >= w * h || wall[side] === 1 || (occ[side] === 1 && frozen[side] === 1);
        if (!destBlocked && !sideBlocked) { stuck = false; break; }
      }
      if (stuck) { frozen[x] = 1; changed = true; }
    }
  }
  for (const x of boxes) if (!goalSet[x] && frozen[x]) return true;
  return false;
}

// The smallest cell of the player's reachable region (boxes as obstacles):
// the canonical representative of the player's position for a state key.
function regionMin(b: Board, boxes: number[], player: number): number {
  const occ = new Uint8Array(b.w * b.h);
  for (const x of boxes) occ[x] = 1;
  const seen = new Uint8Array(b.w * b.h);
  const q: number[] = [player];
  seen[player] = 1;
  let min = player;
  for (let qi = 0; qi < q.length; qi++) {
    const c = q[qi]!;
    if (c < min) min = c;
    for (const [dx, dy] of DIRS) {
      const nx = (c % b.w) + dx;
      const ny = ((c / b.w) | 0) + dy;
      if (nx < 0 || nx >= b.w || ny < 0 || ny >= b.h) continue;
      const n = ny * b.w + nx;
      if (b.wall[n] || occ[n] || seen[n]) continue;
      seen[n] = 1;
      q.push(n);
    }
  }
  return min;
}

interface SearchResult {
  moves: string;
  pushes: number;
  player: number;
  boxes: number[];
  expansions: number;
}

// Push-optimal A* on one board. Returns null when the budget runs out or
// the position is unsolvable.
function solveBoard(b: Board, startPlayer: number, startBoxes: number[], cap: number): SearchResult | null {
  const boxes0 = [...startBoxes].sort((a, c) => a - c);
  if (b.goals.length !== boxes0.length) return null;
  if (boxes0.every((x) => b.goalSet[x])) {
    return { moves: '', pushes: 0, player: startPlayer, boxes: boxes0, expansions: 0 };
  }
  if (isDeadlocked(b, boxes0)) return null;
  const h0 = hungarian(b, boxes0);
  if (!isFinite(h0)) return null;

  const sp: number[] = [startPlayer];
  const sb: number[][] = [boxes0];
  const sg: number[] = [0];
  const parent: number[] = [-1];
  const seg: string[] = [''];
  const keyOf = (boxes: number[], player: number): string =>
    boxes.join(',') + '|' + regionMin(b, boxes, player);
  const best = new Map<string, number>();
  best.set(keyOf(boxes0, startPlayer), 0);

  // Binary heap of [f, -g, seq, state], smallest first; seq breaks ties so
  // the order is identical on every run.
  const heap: [number, number, number, number][] = [];
  let seq = 0;
  const less = (a: [number, number, number, number], e: [number, number, number, number]): boolean =>
    a[0] < e[0] || (a[0] === e[0] && (a[1] < e[1] || (a[1] === e[1] && a[2] < e[2])));
  const heapPush = (e: [number, number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      const t = heap[i]!; heap[i] = heap[p]!; heap[p] = t;
      i = p;
    }
  };
  const heapPop = (): [number, number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        const t = heap[i]!; heap[i] = heap[c]!; heap[c] = t;
        i = c;
      }
    }
    return top;
  };
  heapPush([h0, 0, seq++, 0]);

  let expansions = 0;
  const occ = new Uint8Array(b.w * b.h);
  while (heap.length > 0) {
    if (expansions >= cap || spent >= TOTAL_EXPANSIONS) return null;
    const [, ng, , s] = heapPop();
    const boxes = sb[s]!;
    const player = sp[s]!;
    if (best.get(keyOf(boxes, player)) !== s) continue;
    if (boxes.every((x) => b.goalSet[x])) {
      let moves = '';
      for (let t = s; t > 0; t = parent[t]!) moves = seg[t] + moves;
      return { moves, pushes: sg[s]!, player, boxes, expansions };
    }
    expansions++;
    spent++;
    for (let i = 0; i < b.w * b.h; i++) occ[i] = 0;
    for (const x of boxes) occ[x] = 1;
    for (let bi = 0; bi < boxes.length; bi++) {
      const box = boxes[bi]!;
      const bx = box % b.w;
      for (const [dx, dy, ch] of DIRS) {
        if (bx + dx < 0 || bx + dx >= b.w || bx - dx < 0 || bx - dx >= b.w) continue;
        const dest = box + dy * b.w + dx;
        const side = box - dy * b.w - dx;
        if (dest < 0 || dest >= b.w * b.h || side < 0 || side >= b.w * b.h) continue;
        if (b.wall[dest] || b.boxWall[dest] || occ[dest]) continue;
        if (b.wall[side] || occ[side]) continue;
        const walk = bfsPath(b.w, b.h, b.wall, occ, player, side);
        if (walk === null) continue;
        const nb = boxes.slice();
        nb[bi] = dest;
        nb.sort((a, c) => a - c);
        if (isDeadlocked(b, nb)) continue;
        const h = hungarian(b, nb);
        if (!isFinite(h)) continue;
        const g = sg[s]! + 1;
        const k = keyOf(nb, box);
        const seen = best.get(k);
        if (seen !== undefined && sg[seen]! <= g) continue;
        const t = sp.length;
        sp.push(box);
        sb.push(nb);
        sg.push(g);
        parent.push(s);
        seg.push(walk + ch);
        best.set(k, t);
        heapPush([g + h, -g, seq++, t]);
      }
    }
  }
  return null;
}

// Simulate moves on a board; returns the end state or null on an illegal step.
function simulate(w: number, h: number, wall: Uint8Array, goals: number[], boxesIn: number[], playerIn: number, moves: string): { player: number; boxes: number[] } | null {
  const boxes = [...boxesIn];
  let player = playerIn;
  const goalSet = new Uint8Array(w * h);
  for (const g of goals) goalSet[g] = 1;
  for (const ch of moves) {
    let dx = 0;
    let dy = 0;
    if (ch === 'u') dy = -1;
    else if (ch === 'd') dy = 1;
    else if (ch === 'l') dx = -1;
    else if (ch === 'r') dx = 1;
    else continue;
    const nx = (player % w) + dx;
    const ny = ((player / w) | 0) + dy;
    if (nx < 0 || nx >= w || ny < 0 || ny >= h) return null;
    const q = ny * w + nx;
    if (wall[q]) return null;
    const at = boxes.indexOf(q);
    if (at >= 0) {
      const bx = nx + dx;
      const by = ny + dy;
      if (bx < 0 || bx >= w || by < 0 || by >= h) return null;
      const dest = by * w + bx;
      if (wall[dest] || boxes.includes(dest)) return null;
      boxes[at] = dest;
    }
    player = q;
  }
  boxes.sort((a, c) => a - c);
  return { player, boxes };
}

interface Room {
  cells: number[];
  cellSet: Uint8Array;
  boxes: number[];
  goals: number[];
  door: number;
  entry: number;
}

// Split a level into its rooms: the hallway is the player's row; every
// connected component of open cells off that row is one room, entered from
// the hallway cell beside its door.
function findRooms(level: { w: number; h: number; wall: Uint8Array; goals: number[]; boxes: number[]; player: number }): Room[] | null {
  const { w, h, wall, goals, boxes, player } = level;
  const hallY = (player / w) | 0;
  const seen = new Uint8Array(w * h);
  const rooms: Room[] = [];
  for (let start = 0; start < w * h; start++) {
    if (wall[start] || seen[start]) continue;
    if ((start / w | 0) === hallY) continue;
    // Flood-fill one component.
    const cells: number[] = [start];
    seen[start] = 1;
    for (let qi = 0; qi < cells.length; qi++) {
      const c = cells[qi]!;
      for (const [dx, dy] of DIRS) {
        const nx = (c % w) + dx;
        const ny = ((c / w) | 0) + dy;
        if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
        const n = ny * w + nx;
        if (wall[n] || seen[n] || (n / w | 0) === hallY) continue;
        seen[n] = 1;
        cells.push(n);
      }
    }
    const cellSet = new Uint8Array(w * h);
    for (const c of cells) cellSet[c] = 1;
    const rBoxes = boxes.filter((x) => cellSet[x]);
    const rGoals = goals.filter((x) => cellSet[x]);
    // The door: the component cell touching the hallway row.
    let door = -1;
    let entry = -1;
    for (const c of cells) {
      const cy = (c / w) | 0;
      if (Math.abs(cy - hallY) === 1) {
        const e = hallY * w + (c % w);
        if (!wall[e]) { door = c; entry = e; break; }
      }
    }
    if (door < 0 || rBoxes.length === 0) continue;
    if (rBoxes.length !== rGoals.length) return null;
    rooms.push({ cells, cellSet, boxes: rBoxes, goals: rGoals, door, entry });
  }
  if (rooms.length === 0) return null;
  const covered = rooms.reduce((a, r) => a + r.boxes.length, 0);
  if (covered !== boxes.length) return null;
  rooms.sort((a, b) => a.door - b.door);
  return rooms;
}

function solveLevel(text: string): string {
  const level = parseLevel(text);
  const { w, h, wall, goals, boxes, player } = level;
  if (boxes.every((x) => goals.includes(x))) return '';
  const rooms = findRooms(level);
  if (rooms !== null) {
    // Try room orders (door order, then reversed) until one concatenates.
    const orders = [rooms, [...rooms].reverse()];
    for (const order of orders) {
      const solved: { moves: string; player: number; boxes: number[] }[] = [];
      let ok = true;
      for (const room of order) {
        const boxWall = new Uint8Array(w * h).fill(1);
        for (const c of room.cells) boxWall[c] = 0;
        boxWall[room.entry] = 1; // the player may stand here, a box never may
        const board = makeBoard(w, h, wall, boxWall, room.goals);
        const res = solveBoard(board, room.entry, room.boxes, ROOM_EXPANSIONS);
        if (res === null) { ok = false; break; }
        solved.push(res);
      }
      if (!ok) continue;
      // Concatenate: walk to each door through the hallway, then the room.
      let moves = '';
      let curPlayer = player;
      let curBoxes = [...boxes];
      let good = true;
      for (let ri = 0; ri < order.length; ri++) {
        const room = order[ri]!;
        const res = solved[ri]!;
        const occ = new Uint8Array(w * h);
        for (const x of curBoxes) occ[x] = 1;
        const walk = bfsPath(w, h, wall, occ, curPlayer, room.entry);
        if (walk === null) { good = false; break; }
        moves += walk + res.moves;
        const sim = simulate(w, h, wall, goals, curBoxes, curPlayer, walk + res.moves);
        if (sim === null) { good = false; break; }
        curPlayer = sim.player;
        curBoxes = sim.boxes;
      }
      if (!good) continue;
      const final = simulate(w, h, wall, goals, boxes, player, moves);
      if (final !== null && final.boxes.every((x) => goals.includes(x))) return moves;
    }
  }
  // Fallback: one A* over the whole level.
  const board = makeBoard(w, h, wall, new Uint8Array(w * h), goals);
  const res = solveBoard(board, player, boxes, LEVEL_EXPANSIONS);
  if (res !== null) {
    const final = simulate(w, h, wall, goals, boxes, player, res.moves);
    if (final !== null && final.boxes.every((x) => goals.includes(x))) return res.moves;
  }
  return '';
}

for (const id of readdirSync(I).sort()) {
  const moves = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
