// Builds docs/slides.pptx from the report's figures and tables (M8).
// Build-time tool only (not a project dependency):
//   cd <any scratch dir> && npm install pptxgenjs
//   NODE_PATH=<scratch dir>/node_modules node docs/slides/build_deck.js      (run from the repo root)
// All numbers are copied from outputs/chg3_final/results.md (val, 110 pairs, the final method) and docs/REPORT.md.
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..", "..");
const FIG = (f) => path.join(ROOT, "docs", "figures", f);
const OUT = path.join(ROOT, "docs", "slides.pptx");

// palette: deep slate (dominant on title / closing slides), white content, figure colours as accents
const COL = {
  slate: "1F2A36", slate2: "2E3D4D", ink: "0B0B0B", ink2: "52514E", muted: "898781", line: "E1E0D9",
  tint: "F3F5F7", white: "FFFFFF", blue: "2A78D6", orange: "EB6834", aqua: "1BAF7A", ice: "CFE0F5",
};
const HEAD = "Cambria", BODY = "Calibri";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.author = "Erik Sedlák";
pres.title = "Living Scene Graphs, Subproject D: change detection and graph maintenance";
pres.theme = { headFontFace: HEAD, bodyFontFace: BODY };

const W = 13.333, M = 0.6;
pres.defineSlideMaster({
  title: "DARK",
  background: { color: COL.slate },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 2.3, w: W - 2 * M, h: 1.6, fontFace: HEAD,
      fontSize: 44, bold: true, color: COL.white, valign: "bottom", align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: M, y: 4.05, w: W - 2 * M, h: 1.6, fontFace: BODY,
      fontSize: 20, color: COL.ice, valign: "top" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "DARK_LIST",
  background: { color: COL.slate },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.6, w: W - 2 * M, h: 1.1, fontFace: HEAD,
      fontSize: 40, bold: true, color: COL.white, valign: "middle" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: M, y: 2.0, w: W - 2 * M, h: 4.6, fontFace: BODY,
      fontSize: 20, color: COL.white, valign: "top" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: COL.white },
  margin: [0.5, 0.6, 0.5, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.35, w: W - 2 * M, h: 0.95, fontFace: HEAD,
      fontSize: 34, bold: true, color: COL.slate, valign: "middle" }, text: "" } },
  ],
  slideNumber: { x: W - 1.0, y: 7.0, w: 0.5, h: 0.3, fontFace: BODY, fontSize: 10, color: COL.muted },
});

function text(slide, t, o) {
  slide.addText(t, { isTextBox: true, fontFace: BODY, color: COL.ink, margin: 0, ...o });
}
function card(slide, x, y, w, h, fill) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill || COL.tint },
    line: { color: fill || COL.tint } });
}
function stat(slide, x, y, w, big, label, color) {
  card(slide, x, y, w, 2.1);
  text(slide, big, { x: x + 0.3, y: y + 0.2, w: w - 0.6, h: 1.0, fontFace: HEAD, fontSize: 48, bold: true,
    color: color || COL.slate });
  text(slide, label, { x: x + 0.3, y: y + 1.2, w: w - 0.6, h: 0.8, fontSize: 15, color: COL.ink2, valign: "top" });
}
function bullets(slide, items, o) {
  slide.addText(items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1 } })),
    { isTextBox: true, fontFace: BODY, fontSize: 16, color: COL.ink, paraSpaceAfter: 8, valign: "top", margin: 0, ...o });
}
function image(slide, file, x, y, w, pxW, pxH, alt) {
  slide.addImage({ path: FIG(file), x, y, w, h: (w * pxH) / pxW, altText: alt });
}

// 1 — title
let s = pres.addSlide({ masterName: "DARK" });
s.addText("Keeping a scene graph alive", { placeholder: "title" });
s.addText("Object-level change detection and graph maintenance on 3RScan\nLiving Scene Graphs, Subproject D · Erik Sedlák · November 2026",
  { placeholder: "body" });
s.addNotes("Given a scene graph of a room from one visit and the RGB-D frames of a later visit, update the graph: what is unchanged, moved, removed, unknown, new, and the support relations.");

// 2 — the problem
s = pres.addSlide({ masterName: "CONTENT" });
s.addText("Gone, or just not seen?", { placeholder: "title" });
stat(s, M, 1.6, 3.8, "737", "reference objects missing from the rescan annotation but not listed removed (val)", COL.slate);
stat(s, M + 4.15, 1.6, 3.8, "73", "objects listed removed (scored)", COL.orange);
stat(s, M + 8.3, 1.6, 3.8, "95%", "of the 737 deleted by a naive \"unmatched → removed\" baseline (B0)", COL.orange);
text(s, "A rescan covers only part of the room. Absence in the data is not evidence of removal: a method must say " +
  "\"unknown\" for what it did not see, and \"removed\" only when it saw the empty place.",
  { x: M, y: 4.2, w: W - 2 * M, h: 1.2, fontSize: 20, color: COL.ink, valign: "top" });
text(s, "False deletion = share of these 737 objects the method deletes (lower is better; some are real, unlisted removals).",
  { x: M, y: 6.3, w: W - 2 * M, h: 0.5, fontSize: 12, color: COL.muted });

// 3 — task and modes
s = pres.addSlide({ masterName: "CONTENT" });
s.addText("Task: maintain the graph from raw RGB-D", { placeholder: "title" });
const colW = 3.7, gap = 0.45;
const flow = [
  ["Input", "Reference scene graph (objects, boxes, support relations), reference mesh + texture; rescan RGB-D frames, poses, T_align"],
  ["Online walk", "Frames in time order: render the reference at the rescan pose, compare depth and appearance, vote per node"],
  ["Output", "Per node: unchanged / moved / removed / unknown + confidence; new nodes; recomputed on / in relations; snapshots"],
];
flow.forEach(([h, b], i) => {
  const x = M + i * (colW + gap + 0.15);
  card(s, x, 1.6, colW, 2.6);
  text(s, h, { x: x + 0.3, y: 1.8, w: colW - 0.6, h: 0.5, fontFace: HEAD, fontSize: 20, bold: true, color: COL.slate });
  text(s, b, { x: x + 0.3, y: 2.35, w: colW - 0.6, h: 1.8, fontSize: 15, color: COL.ink2, valign: "top" });
  if (i < 2) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + colW + 0.1, y: 2.7, w: 0.4, h: 0.4, fill: { color: COL.muted },
    line: { color: COL.muted } });
});
card(s, M, 4.6, 5.9, 2.0, "E8F0FA");
text(s, "perception  (headline)", { x: M + 0.3, y: 4.75, w: 5.3, h: 0.45, fontFace: HEAD, fontSize: 18, bold: true, color: COL.blue });
text(s, "Finds rescan objects itself from depth + appearance residuals, labels them with CLIP. Never reads rescan annotations (import test).",
  { x: M + 0.3, y: 5.25, w: 5.3, h: 1.2, fontSize: 15, color: COL.ink, valign: "top" });
card(s, M + 6.25, 4.6, 5.9, 2.0);
text(s, "oracle  (upper bound)", { x: M + 6.55, y: 4.75, w: 5.3, h: 0.45, fontFace: HEAD, fontSize: 18, bold: true, color: COL.slate });
text(s, "Rescan objects = GT instances (IDs shuffled). Isolates the change logic; measures what perception costs. Every table has both rows.",
  { x: M + 6.55, y: 5.25, w: 5.3, h: 1.2, fontSize: 15, color: COL.ink, valign: "top" });

// 4 — GT findings
s = pres.addSlide({ masterName: "CONTENT" });
s.addText("What the ground truth really says", { placeholder: "title" });
const gts = [
  ["36%", "of val moves are < 10 cm → \"don't care\" (235 moves scored)"],
  ["171", "scored new objects of 224 raw \"new\" IDs; noisy, audited"],
  ["4", "real support-parent swaps among 278 moved things"],
  ["12", "door swings: articulation, out of scope (not scored)"],
];
gts.forEach(([b, l], i) => stat(s, M + i * 3.1, 1.7, 2.85, b, l, i === 2 ? COL.blue : COL.slate));
text(s, "Consequences for the evaluation", { x: M, y: 4.3, w: 6, h: 0.45, fontFace: HEAD, fontSize: 20, bold: true, color: COL.slate });
bullets(s, [
  "Absent-unlisted objects are excluded from removed P / R and reported as the false-deletion rate",
  "Most \"relation changes\" are annotation gaps, and \"cup moves from table to shelf\" is rare: relation updates are also scored on 40 training pairs with the most parent changes (relchange, never tuned on)",
  "Thresholds chosen on 33 training pairs (dev_tune), each with a rule written before the run; val (110 pairs, 47 rooms) reported once",
], { x: M, y: 4.85, w: W - 2 * M, h: 1.9 });

// 5 — method
s = pres.addSlide({ masterName: "CONTENT" });
s.addText("Method: evidence from free space, decided late", { placeholder: "title" });
const steps = [
  ["1", "Render", "reference mesh at the rescan pose; T_align refined online by ICP from the frames seen so far"],
  ["2", "Vote", "free = seen through (d_obs > d_exp + 0.15 m); seen only with contrast behind; thin nodes: DINOv2 texture vs photo"],
  ["3", "Decorrelate", "a vote counts only after the camera moved ≥ 0.30 m or turned ≥ 20°"],
  ["4", "Candidates", "perception: nearer-than-reference voxels (see-through votes subtract), co-segmented, CLIP-labelled"],
  ["5", "Decide (every 20 frames, final at the end)", "confirmed in place; Hungarian; a move needs evidence at the old place; removed only with free evidence"],
  ["6", "Confidence + relations", "vote weights measured on the tuning set, delete if p < 0.4; re-check support of what changed"],
];
steps.forEach(([n, h, b], i) => {
  const col = i % 2, row = Math.floor(i / 2);
  const x = M + col * 6.15, y = 1.55 + row * 1.75;
  card(s, x, y, 5.95, 1.5);
  s.addShape(pres.shapes.OVAL, { x: x + 0.25, y: y + 0.4, w: 0.65, h: 0.65, fill: { color: COL.slate }, line: { color: COL.slate } });
  text(s, n, { x: x + 0.25, y: y + 0.4, w: 0.65, h: 0.65, fontFace: HEAD, fontSize: 20, bold: true, color: COL.white, align: "center", valign: "middle" });
  text(s, h, { x: x + 1.15, y: y + 0.15, w: 4.6, h: 0.45, fontFace: HEAD, fontSize: 17, bold: true, color: COL.slate });
  text(s, b, { x: x + 1.15, y: y + 0.6, w: 4.6, h: 0.85, fontSize: 14, color: COL.ink2, valign: "top" });
});

// 6 — free-space evidence: native chart
s = pres.addSlide({ masterName: "CONTENT" });
s.addText("Free-space evidence removes 90% of false deletions", { placeholder: "title" });
s.addChart(pres.charts.BAR, [
  { name: "oracle", labels: ["B0 naive", "B1 + votes", "B2 + association", "FULL"], values: [95.0, 9.5, 9.5, 9.1] },
  { name: "perception", labels: ["B0 naive", "B1 + votes", "B2 + association", "FULL"], values: [83.6, 8.0, 7.9, 7.2] },
], {
  x: M, y: 1.5, w: 7.4, h: 5.2, barDir: "col", barGrouping: "clustered", chartColors: [COL.blue, COL.orange],
  showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 12, dataLabelColor: COL.ink2, dataLabelFontFace: "+mn-lt",
  dataLabelFormatCode: "0.0",
  showTitle: true, title: "False deletion on val, % (n = 737, lower is better)", titleFontSize: 14, titleColor: COL.ink2, titleFontFace: "+mn-lt",
  showLegend: true, legendPos: "b", legendFontSize: 12, legendFontFace: "+mn-lt", legendColor: COL.ink2,
  catAxisLabelColor: COL.ink2, catAxisLabelFontSize: 12, catAxisLabelFontFace: "+mn-lt",
  valAxisLabelColor: COL.muted, valAxisLabelFontSize: 11, valAxisLabelFontFace: "+mn-lt", valAxisMaxVal: 100,
  valGridLine: { color: COL.line, size: 0.5 }, catGridLine: { style: "none" },
});
text(s, "Rule", { x: 8.5, y: 1.7, w: 4.2, h: 0.45, fontFace: HEAD, fontSize: 20, bold: true, color: COL.slate });
bullets(s, [
  "Only seeing through the place a node occupied lowers its confidence",
  "Occluded, out of view, invalid depth: no information → unknown",
  "Removed precision 22 → 61% (oracle), 4 → 16% (perception) from B0 to B1, same matching",
  "FULL keeps 93.4% of unchanged objects in perception, 90.9% in oracle",
], { x: 8.5, y: 2.25, w: 4.2, h: 4.3, fontSize: 15 });

// 7 — results table
s = pres.addSlide({ masterName: "CONTENT" });
s.addText("Results on val: both modes, every method", { placeholder: "title" });
const hdr = ["method", "mode", "removed P / R", "false del. ↓", "unch. kept", "moved F1", "new P / R", "new, any label"];
const rows = [
  ["B0", "oracle", "21.5 / 71.2", "95.0", "80.7", "30.6", "85.0 / 59.6", "85.0 / 59.6"],
  ["B1", "oracle", "60.8 / 42.5", "9.5", "92.4", "30.6", "85.0 / 59.6", "85.0 / 59.6"],
  ["B2", "oracle", "55.6 / 41.1", "9.5", "85.7", "56.3", "82.1 / 59.1", "82.1 / 59.1"],
  ["FULL", "oracle", "58.3 / 28.8", "9.1", "90.9", "56.5", "67.5 / 66.7", "67.5 / 66.7"],
  ["B0", "perception", "4.0 / 74.0", "83.6", "6.3", "14.1", "5.6 / 7.6", "7.9 / 10.5"],
  ["B1", "perception", "15.5 / 46.6", "8.0", "80.4", "14.1", "5.6 / 7.6", "7.9 / 10.5"],
  ["B2", "perception", "18.5 / 46.6", "7.9", "86.2", "38.1", "3.4 / 3.5", "6.9 / 7.0"],
  ["FULL", "perception", "24.3 / 37.0", "7.2", "93.4", "42.5", "3.1 / 3.5", "6.9 / 7.6"],
  ["+ detector", "perception", "29.3 / 32.9", "7.1", "92.1", "45.2", "6.1 / 21.1", "10.1 / 34.5"],
];
const cell = (t, o = {}) => ({ text: t, options: { fontFace: BODY, fontSize: 14, color: COL.ink, valign: "middle", ...o } });
const tbl = [hdr.map((h, j) => cell(h, { bold: true, color: COL.white, fill: { color: COL.slate }, align: j < 2 ? "left" : "right" }))].concat(
  rows.map((r) => r.map((c, j) => cell(c, {
    bold: r[0] === "FULL", fill: { color: r[0] === "FULL" ? "E8F0FA" : COL.white }, align: j < 2 ? "left" : "right",
  }))));
s.addTable(tbl, { x: M, y: 1.45, w: W - 2 * M, colW: [1.25, 1.45, 1.75, 1.35, 1.3, 1.25, 1.6, 2.18], rowH: 0.43,
  border: { type: "solid", pt: 0.5, color: COL.line } });
text(s, "%, micro-averaged over 110 pairs. Support: 73 removed, 737 absent-unlisted, 1,401 unchanged, 235 moved, 171 new. " +
  "95% bootstrap CIs in the report (e.g. FULL perception false deletion 7.2 [4.6, 10.8]). + detector = ablation (Grounding DINO + SAM 2).",
  { x: M, y: 6.05, w: W - 2 * M, h: 0.7, fontSize: 12, color: COL.muted, valign: "top" });

// 8 — qualitative
s = pres.addSlide({ masterName: "CONTENT" });
s.addText("One room, before and after (oracle)", { placeholder: "title" });
image(s, "topdown_val_8eabc45f_oracle.png", M, 1.45, W - 2 * M, 2070, 720,
  "Top-down view of val pair 8eabc45f to 5341b7b3: reference graph, updated graph, ground truth");
text(s, "Val pair chosen by rule (most GT changes in the 20-pair subset). The two pillows removed from the sofa and several moved " +
  "pillows are found; the sofa and couch-table moves are missed and a bench is moved by mistake. Perception deletes the same pillows.",
  { x: M, y: 5.85, w: W - 2 * M, h: 0.9, fontSize: 15, color: COL.ink2, valign: "top" });

// 9 — ablations
s = pres.addSlide({ masterName: "CONTENT" });
s.addText("Deciding late matters most", { placeholder: "title" });
stat(s, M, 1.5, 3.7, "58 → 16%", "oracle removed precision with immediate commitment (delete as soon as the place looks empty)", COL.orange);
stat(s, M, 3.85, 3.7, "57 → 22%", "oracle moved F1: moved objects are deleted and reappear as false new nodes", COL.orange);
image(s, "m7_fraction.png", M + 4.1, 1.55, 8.0, 1950, 480, "Observed-fraction ablation: metrics vs share of frames, both modes");
text(s, "Less observation → more unknown (oracle 31 → 78% at 10% of frames), not more false deletions (9.1 → 1.4%).",
  { x: M + 4.1, y: 3.65, w: 8.0, h: 0.7, fontSize: 15, color: COL.ink2, valign: "top" });
bullets(s, [
  "A move needs evidence at the old place: false moves 4.3 → 1.9% (oracle), unchanged kept 85.9 → 90.9%",
  "Appearance off: ≈ no change on val, only 4 scored removals are thin; catches the 2 cm book in the controlled test",
  "τ_free 0.05–0.30 m: a monotone recall / false-deletion trade-off; 0.15 m kept",
], { x: M + 4.1, y: 4.45, w: 8.0, h: 2.3, fontSize: 15 });

// 9b — the graph during the walk (anytime curve, val)
s = pres.addSlide({ masterName: "CONTENT" });
s.addText("The graph during the walk", { placeholder: "title" });
image(s, "anytime_val.png", M, 1.45, 8.2, ANYTIME_W, ANYTIME_H, "Metrics of the tentative graph over the walk, val, both modes");
bullets(s, ANYTIME_BULLETS, { x: M + 8.5, y: 1.6, w: 3.6, h: 5.0, fontSize: 15 });

// 10 — relations
s = pres.addSlide({ masterName: "CONTENT" });
s.addText("Relations: keeping the old parent is a strong baseline", { placeholder: "title" });
const rh = ["set (FULL, oracle)", "keep old (R0)", "geometric (R1)", "ours (update)"];
const rr = [
  ["val (a) all things, n = 995", "91.9", "79.7", "93.3"],
  ["val (b) moved things, n = 168", "94.0", "63.7", "86.3"],
  ["relchange (b) moved, n = 93", "48.4", "40.9", "47.3"],
  ["relchange (c) parent swapped, n = 14", "0.0", "42.9", "35.7"],
];
s.addTable([rh.map((h, j) => cell(h, { bold: true, color: COL.white, fill: { color: COL.slate }, align: j ? "right" : "left" }))].concat(
  rr.map((r) => r.map((c, j) => cell(c, { align: j ? "right" : "left", bold: j === 3 })))),
  { x: M, y: 1.55, w: 7.6, colW: [3.4, 1.4, 1.4, 1.4], rowH: 0.55, border: { type: "solid", pt: 0.5, color: COL.line } });
text(s, "Parent accuracy of things, %, GT = rescan 3DSSG.", { x: M, y: 4.45, w: 7.6, h: 0.4, fontSize: 12, color: COL.muted });
card(s, 8.6, 1.55, 4.13, 4.6);
bullets(s, [
  "Most moved things keep their parent: R0 is right 94% of the time on val",
  "Ours matches R0 overall and wins only where parents really change",
  "Keeping a still-supporting parent: val (b) 73.8 → 86.3%",
  "on / in type correct in 95–97% given the parent",
], { x: 8.9, y: 1.8, w: 3.6, h: 4.2, fontSize: 15 });

// 11 — perception gap + audit
s = pres.addSlide({ masterName: "CONTENT" });
s.addText("Where perception fails, and what the GT misses", { placeholder: "title" });
image(s, "topdown_val_4a9a43d2_perception_failure.png", M, 1.4, 7.6, 2070, 676,
  "Perception failure case 4a9a43d2: large false new nodes, a moved stool deleted");
bullets(s, [
  "62 depth-residual candidates per pair, 29% match any GT object; no signal separates them → 6 of 171 new objects",
  "Detector ablation (Grounding DINO + SAM 2): new R 3.5 → 21%, moved F1 42.5 → 45.2, but loses removals on the tuning set",
  "Moved F1 42.5 vs 56.5% (oracle): small moves and same-label swaps are missed",
], { x: M, y: 4.05, w: 7.6, h: 2.6, fontSize: 15 });
card(s, 8.6, 1.4, 4.13, 5.3, "E8F0FA");
text(s, "GT audit (§7.4)", { x: 8.9, y: 1.6, w: 3.6, h: 0.45, fontFace: HEAD, fontSize: 18, bold: true, color: COL.slate });
text(s, "30 false positives per mode (10 moved / removed / new) + the confidence-veto cases, labelled by hand from photo panels.\n\n" +
  "[Share of GT gaps: pending labelling]\n\nDev examples: a robot vacuum really gone but unlisted (caught only by appearance); " +
  "a pillow replaced by a different pillow (missed).",
  { x: 8.9, y: 2.15, w: 3.6, h: 4.4, fontSize: 14, color: COL.ink, valign: "top" });

// 12 — takeaways
s = pres.addSlide({ masterName: "DARK_LIST" });
s.addText("Takeaways", { placeholder: "title" });
s.addText([
  { text: "Free-space evidence + deferred commitment: false deletion 95% → 7–9%, unknown instead of wrong", options: { bullet: true, breakLine: true } },
  { text: "A move needs evidence at the old place, as a removal does: fewer false moves, more unchanged kept", options: { bullet: true, breakLine: true } },
  { text: "The graph is maintained during the walk; perception keeps it safe but finds few new objects: candidates are the bottleneck", options: { bullet: true, breakLine: true } },
  { text: "Limits: the confidence veto costs removed recall; doors, rotations, multi-visit chains out of scope", options: { bullet: true, breakLine: true } },
  { text: "Reproducible: one command per table, metrics.json with config hash + git SHA; README runs one pair end-to-end", options: { bullet: true } },
], { placeholder: "body", fontSize: 20, paraSpaceAfter: 14 });

// write, then drop the ZIP's folder entries (they break opening / co-editing through cloud services)
pres.writeFile({ fileName: OUT }).then(async (f) => {
  const fs = require("fs");
  const JSZip = require("jszip");
  const zip = await JSZip.loadAsync(fs.readFileSync(f));
  const out = new JSZip(); // copy file entries only (JSZip.remove() on a folder would delete its contents)
  const names = Object.keys(zip.files).filter((n) => !zip.files[n].dir);
  names.sort((a, b) => (b === "[Content_Types].xml") - (a === "[Content_Types].xml")); // content types first
  for (const n of names) out.file(n, await zip.files[n].async("nodebuffer"), { createFolders: false });
  fs.writeFileSync(f, await out.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
  console.log("wrote " + f);
});
