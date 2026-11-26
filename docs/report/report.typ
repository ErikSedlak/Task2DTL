// Living Scene Graphs, Subproject D: report in the structure of the author's bachelor thesis.
// Compile:  uv run --with typst python -c "import typst; typst.compile('docs/report/report.typ', output='docs/report/report.pdf')"
// Numbers: outputs/chg3_final (val, one run), outputs/chg3_final_anytime, outputs/chg3_audit.

#set page(paper: "a4", margin: (x: 1.6cm, top: 1.4cm, bottom: 1.5cm), numbering: "1")
#set text(font: "New Computer Modern", size: 9.3pt, lang: "en")
#set par(justify: true, leading: 0.5em, spacing: 0.6em, first-line-indent: 0pt)
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => block(above: 0.9em, below: 0.45em, text(size: 12pt, weight: "bold", it))
#show heading.where(level: 2): it => block(above: 0.7em, below: 0.3em, text(size: 10pt, weight: "bold", it))
#set figure(gap: 0.4em)
#set figure.caption(separator: [: ])
#show figure.caption: set text(size: 8.5pt)
#show figure: set block(above: 0.7em, below: 0.7em)
#set table(stroke: 0.4pt + luma(120), inset: (x: 3pt, y: 2.2pt))
#show table: set text(size: 8pt)
#show link: set text(fill: rgb("#1f4e8c"))

#align(center)[
  #text(size: 15pt, weight: "bold")[Change Detection and Graph Maintenance over Time on 3RScan] \
  #v(0.2em)
  #text(size: 10pt)[Erik Sedlák · Living Scene Graphs, Subproject D · November 2026]
]

#block(inset: (x: 1.2cm), above: 0.6em)[
  #text(weight: "bold")[Abstract.] A scene graph of a room is wrong as soon as someone moves a chair or takes a cup away.
  This report describes a method that keeps the graph of a room up to date when the room is visited again. It gets the
  graph of the first visit and the RGB-D frames of the second one, and decides for every object whether it is
  _unchanged_, _moved_, _removed_ or _unknown_, adds _new_ objects, and re-checks relations such as "pillow on sofa". The main idea
  is that an object counts as removed only when the sensor looks _through_ the place where it used to be; an object that
  is merely out of view or hidden stays _unknown_. On the 110 validation pairs of 3RScan this lowers the false deletion rate of a
  naive baseline from 95% to 7–9%. Moves are found with an F1 of 57% (given object instances) and 43% (the method finds the objects
  itself). Finding new objects from depth alone works poorly (3.5% precision, 4.1% recall); an open-vocabulary detector
  helps recall (22%) but costs removals.
  #linebreak()
  #text(weight: "bold")[Keywords.] change detection, scene graph, 3RScan, RGB-D, long-term mapping
]

#heading(numbering: none)[Introduction]

Robots and AR systems often keep a map of a room as a _scene graph_: the objects are nodes, and relations such as
"on" or "in" are edges. Such a graph is built once, but rooms change. People rearrange furniture, take things away and
bring new ones. If the graph is not updated, it slowly becomes a record of a room that no longer exists.

This report studies one concrete version of the problem on the 3RScan dataset: a graph of a first visit
(the _reference_) and the RGB-D frames of a later visit (the _rescan_) are given, and the graph has to be
corrected. The difficulty is not to see what is new, but to tell _"gone"_ from _"not seen"_. A rescan usually covers
only part of the room. If every object that was not observed were deleted, most of the graph would be destroyed. In the
validation split, 737 reference objects are missing from the rescan annotation without being listed as removed, while only 73 scored objects are listed as removed.

Section 1 defines the task and reviews related work. Section 2 describes the data and the preprocessing, Section 3
explains the method, and Section 4 evaluates it with tables and visualizations. Section 5 concludes.

= Problem definition and related work

== The task

Input: the reference graph (nodes with a class, a 3D box and support relations, from 3DSSG), the reference mesh and its
texture, and, for the rescan, the RGB-D frames with camera poses and a rough alignment `T_align` to the reference. Output: the
updated graph.
- every reference node gets a state: _unchanged_, _moved_ (with a new pose), _removed_ or _unknown_;
- new nodes are added;
- the support relation (parent and type) of changed objects is recomputed;
- every node has a _confidence_ that grows when the object is seen again and falls when it should be visible but is not.

The graph is maintained _during_ the walk: after every 20 processed frames the method writes a tentative graph, and the last one is the result.
Two observation modes share one interface. In *oracle* mode, the rescan objects are the annotated instances of the rescan (with shuffled ids); it is an upper bound that
isolates the change logic. In *perception* mode, the method finds the objects itself from depth and appearance and names them with CLIP. Every
result is reported for both modes.

== Related work

*Data.* 3RScan @wald2019rio contains 1482 RGB-D scans of 478 rooms, many of them re-scanned after people changed the room, with annotations of
which objects moved, disappeared or appeared. 3DSSG @wald2020sg adds scene graphs (objects, support and proximity relations) on top of it.
The original 3RScan task is to re-localize objects in a changed room; graph maintenance is a different use of the same data.

*Scene graphs over time.* SceneGraphFusion @wu2021sgfusion builds a graph incrementally from an RGB-D sequence, but does not model changes between visits.
3D dynamic scene graphs @rosinol2020dsg add places, objects and humans to a map built by SLAM; they focus on people and places rather than on objects that
were rearranged between visits.

*Change detection on 3RScan.* The closest work is "Has Anything Changed?" by Adam et al. @adam2023changed. They render the depth of both reconstructed meshes,
take depth differences as seeds, and let SAM masks grow the seeds to whole objects with a graph optimization. They report class-agnostic recall at
different box IoU thresholds on the same 110 validation pairs, offline and on fused meshes. They do not associate objects across visits, decide between removed
and not seen, or update relations. Their numbers are therefore not comparable with the node-level precision and recall used here (Section 4.7).

*Models used.* The method uses frozen models for appearance only: CLIP @radford2021clip to name candidates, DINOv2 @oquab2023dinov2 to compare reference and rescan
textures, and, for one ablation, Grounding DINO @liu2023gdino with SAM 2 @ravi2024sam2 as an open-vocabulary detector. Rendering and ICP use Open3D @zhou2018open3d.

= Data and preprocessing

== Splits

Three sets of reference–rescan pairs are used. *Tuning set*: 33 training pairs (the 20 rooms with the most changes, 10 random rooms and 3 early development pairs). Every
threshold was chosen there, each with a selection rule written down before the run. *Validation*: 110 pairs in 47 rooms, run once for all tables. *Relation set*:
40 training pairs with the most changes of support parents (73 events), never tuned on. Early validation runs influenced some decisions; they are listed in Section 5.

== What the ground truth really says

Checking the annotation before building anything changed the evaluation:
- *Absent is not removed.* 1,018 reference objects of the validation rooms are missing in the rescan annotation, but only 96 are listed as removed. The others are scored separately
  as _absent-unlisted_; deleting one is a _false deletion_ (lower is better), although some of them really are gone (Section 4.7).
- *Many moves are tiny.* 134 of 370 moves are shorter than 10 cm. Moves below 10 cm are "don't care", which leaves 235 scored moves.
- *"New" is noisy.* Of 224 new ids, 171 are things and furniture; structure such as walls is dropped.
- *Relation changes are rare.* Only 4 of 278 moved things swap their support parent, so a second set was needed for relations.

Scored ground truth on validation: 235 moved, 73 removed, 171 new, 1,401 unchanged and 737 absent-unlisted objects.

== Preprocessing

Units are metres, the reference mesh frame is the common frame, and `T_align` maps the rescan into it. The provided alignment is about 5 cm off, which is enough
to confuse depth comparison at object borders. It is therefore refined during the walk: every 20 processed frames one point-to-plane ICP registers the depth of all frames
so far to the reference walls, floor and ceiling, and the estimate is accepted only if it is close to the raw one and not degenerate (one wall cannot fix a shift along itself).
No future frame is used.

Every third frame is processed, in time order. For each frame the reference mesh is rendered at the rescan pose with Open3D ray casting
(about 5 ms per frame, on the CPU), giving the depth the camera _would_ see if nothing had changed. Frozen CLIP and DINOv2 features are computed on the GPU (GTX 1650) and
cached. 

= Method

== Evidence from free space

The method compares the observed depth $d_"obs"$ with the rendered depth $d_"exp"$ on the pixels where a reference node is first hit. A frame gives one _vote_ per node:
- *free*: the sensor sees _through_ the node ($d_"obs" > d_"exp" + tau_"free"$ with $tau_"free" = 0.15$ m on at least half of the valid pixels);
- *seen*: the surface is at the expected place ($|d_"obs" - d_"exp"| <= 0.10$ m);
- otherwise _no information_. Occlusion, out of view and invalid depth say nothing.

Two refinements keep this honest. A "seen" vote counts only if the surface behind the node is farther than $tau_"free"$: a pillow lying on a sofa cannot be
seen in place by depth, because the sofa is right behind it. For thin objects (below 15 cm) depth cannot see a book vanish, so the reference _texture_ is rendered at the rescan pose and
compared with the photo through DINOv2 features. This is the "geometry + appearance" variant of the assignment. Finally, a vote counts only after _viewpoint
decorrelation_: the camera must have moved 0.30 m or turned 20° since the last counted vote. A person who stands still and films one spot casts one vote, not thirty.

== New-object candidates (perception mode)

Pixels that are clearly nearer than the rendered reference (more than 10 cm) are back-projected, and points near the reference surface or structure are dropped. The rest is accumulated in a 4 cm voxel grid with a
log-odds per voxel; a pixel that sees through a voxel subtracts, so a person who walks away leaves no object behind. Neighbouring voxels are linked only if no frame
puts them in different image segments (_co-segmentation_), which keeps a new sofa apart from its cushions. Each cluster is named by CLIP from its five largest crops and
gets a confidence from the votes of the whole walk. It becomes a `new` node only above a threshold chosen on the tuning set; below it, it can still explain a move.

== Association and states

At the end of the walk a node with at least three counted _seen_ votes and no _free_ vote is *confirmed in place* and is `unchanged`; it also absorbs the observations lying inside it. The remaining nodes and
the observations are matched by the Hungarian algorithm on label, size and position. A matched node is `moved` if ICP of the partial observation onto the node moves its centroid by more than 10 cm.
*A move needs evidence at the old place*: a node whose old place is net-seen, or was never observed, may only be matched to an observation inside its own place. This is the mirror of the removal rule: an object is not
moved away from a place where the sensor still sees it, nor from one that nobody looked at. An unmatched node is `removed` only with free evidence (two counted free votes); otherwise it is `unknown`.

== Confidence and deletion

Every node starts at $p = 0.9$. A _seen_ vote adds 0.63 to the log-odds and a _free_ vote subtracts 1.47; these weights are likelihood ratios measured on the tuning set, not chosen by hand.
A node is deleted only if it has free evidence _and_ $p < 0.4$. `moved` and `removed` are final only at the end of the walk (_deferred commitment_): a moved
object can show up at its new place long after its old place was seen empty.

== Relations and the graph during the walk

Moved and new things, and things whose parent moved or vanished, are re-checked. A thing keeps its old parent if that parent still supports it at the new pose; otherwise geometric contact and containment rules
choose a new parent (an anchor, another thing, or the floor). The same method runs on the state after every 20 processed frames, which gives the tentative graphs of the _anytime_ curve in Section 4.5.

== Baselines and ablations

Four methods build on each other: *B0* matches by class and position and deletes whatever is unmatched; *B1* adds the online votes (a node is removed only with free evidence, otherwise it is unknown); *B2* adds
association and the ICP move test; *FULL* adds the old-place evidence, thin-object appearance, the measured confidence and the cleaner candidates. Each part can be switched off in an ablation row.

= Evaluation

== Metrics

Precision and recall are computed per change type against the 3RScan ground truth: removed, moved and new (a new node must match a ground-truth new object in position and, in the labelled variant, in label; _any label_ is
also reported). Unchanged objects are scored as _unchanged kept_ (the share that stays `unchanged` or `unknown`). *False deletion* is the share of absent-unlisted objects that were deleted. Intervals are 95% bootstrap over pairs.

== Main results

#figure(
  table(
    columns: (auto, auto, 1.15fr, 0.9fr, 0.9fr, 0.9fr, 1.1fr, 1.1fr),
    align: (left, left, right, right, right, right, right, right),
    table.header([*method*], [*mode*], [*removed P / R*], [*false del. ↓*], [*unch. kept*], [*moved F1*], [*new P / R*], [*new, any label*]),
    [B0], [oracle], [21.1 / 69.9], [95.0], [80.8], [31.2], [85.0 / 59.6], [85.0 / 59.6],
    [B1], [oracle], [65.3 / 43.8], [9.1], [92.8], [31.2], [85.0 / 59.6], [85.0 / 59.6],
    [B2], [oracle], [59.6 / 42.5], [9.1], [85.8], [55.8], [81.0 / 59.6], [81.0 / 59.6],
    [*FULL*], [oracle], [*58.5 / 32.9*], [*8.7*], [*90.3*], [*57.4*], [*68.3 / 65.5*], [*68.3 / 65.5*],
    [B0], [perception], [3.9 / 72.6], [83.6], [6.3], [12.3], [5.8 / 8.2], [7.6 / 10.5],
    [B1], [perception], [14.7 / 45.2], [7.9], [80.7], [12.3], [5.8 / 8.2], [7.6 / 10.5],
    [B2], [perception], [17.3 / 43.8], [8.0], [86.2], [38.6], [3.3 / 3.5], [5.6 / 5.8],
    [*FULL*], [perception], [*22.3 / 34.2*], [*7.3*], [*93.4*], [*42.6*], [*3.5 / 4.1*], [*5.6 / 6.4*],
    [FULL + detector], [perception], [28.9 / 32.9], [6.9], [91.9], [44.9], [6.2 / 21.6], [10.1 / 35.1],
  ),
  caption: [Change detection on the 110 validation pairs, in %. Support: 73 removed, 737 absent-unlisted, 1,401 unchanged, 235 moved, 171 new objects.
    95% CIs of FULL: perception false deletion 7.3 [4.7, 10.9], moved F1 42.6 [35.6, 48.6]; oracle moved F1 57.4 [50.4, 63.7].],
) <tab-main>

The table says four things.
- *Free-space evidence is the biggest win.* B0 deletes 95% of the absent-unlisted objects; B1, with the same matching, deletes 9.1% (oracle) and 7.9% (perception). Removed precision rises from 21 to 65% (oracle).
- *Association with ICP fixes moves* (oracle moved F1 31 → 56%), and the old-place evidence makes them safe: it nearly halves the false moves of absent-unlisted objects (oracle 4.3 → 2.3%, perception 6.4 → 3.7%) and keeps
  more unchanged objects (oracle 85.8 → 90.3%). The price is recall on large moves whose old place was never observed.
- *The confidence costs removed recall* (FULL 33% against B2's 43% in oracle) in exchange for fewer wrong deletions.
- *Perception loses most on new objects.* From depth residuals it finds 7 of 171 new objects with the right label (11 with any label). Of 155 visible new objects, only 62 get any candidate, and most of the 56 candidates per pair are
  static scan differences or clutter (only 29% match a ground-truth object). The detector raises new recall to 22% but lost 6 points of removed F1 on the tuning set, so it is not the default.

== Visualization of the graph before and after

#figure(
  image("/figures/topdown_val_8eabc45f_perception.png", width: 100%),
  caption: [Validation room 8eabc45f → 5341b7b3 (perception). Left: the reference graph. Middle: the updated graph, with boxes coloured by state (grey unchanged, dotted unknown, blue moved with an arrow, hatched red removed,
    green new). Right: the ground truth. The two pillows removed from the sofa are deleted and some pillow moves are found, but the sofa stays _unknown_, the lamp is moved to a wrong place and two false new objects appear.],
) <fig-room>

Figure 1 shows a room with many changes, chosen by a fixed rule (the most scored changes in the milestone subset), not by how it looks. The oracle run on the same room also finds the moved sofa. The repository also replays a pair as a time line in Rerun (`lsg viz rerun`).

== Ablations

#figure(
  table(
    columns: (1.9fr, 1.25fr, 1.25fr, 0.9fr),
    align: (left, right, right, right),
    table.header([*row* (each differs from FULL in one part)], [*oracle: rem. R / false del. / moved F1*], [*perception: rem. R / false del. / moved F1*], [*perc. new P / R*]),
    [FULL], [32.9 / 8.7 / 57.4], [34.2 / 7.3 / 42.6], [3.5 / 4.1],
    [− old-place evidence], [37.0 / 8.7 / 55.5], [37.0 / 7.6 / 37.8], [3.2 / 3.5],
    [− measured vote weights], [30.1 / 8.1 / 55.6], [30.1 / 6.9 / 40.1], [4.9 / 5.8],
    [− co-segmentation], [as FULL], [34.2 / 6.9 / 38.7], [4.4 / 2.3],
    [− appearance (depth only)], [32.9 / 8.4 / 57.0], [32.9 / 7.1 / 42.6], [3.6 / 4.1],
    [− online alignment (pre-pass over all frames)], [28.8 / 9.0 / 54.6], [35.6 / 7.9 / 40.4], [3.6 / 4.1],
    [− viewpoint decorrelation], [34.2 / 9.2 / 57.6], [37.0 / 8.4 / 42.0], [2.2 / 6.4],
    [\+ immediate commitment], [52.1 / 10.2 / 23.8], [52.1 / 10.2 / 15.3], [5.2 / 7.6],
    [the earlier method (class priors, strict candidates)], [27.4 / 8.1 / 54.6], [32.9 / 7.7 / 31.7], [2.0 / 8.8],
  ),
  caption: [Ablations on validation, in %. Removed recall, false deletion rate and moved F1 per mode, and perception new precision / recall.],
) <tab-abl>

*Deciding late is the clearest result.* Deleting a node as soon as its place looks empty removes objects that merely moved; their observations at the new place then become false new nodes. Removed precision falls from 58% to 17%
(oracle), and oracle moved F1 from 57% to 24%. *Geometry versus geometry + appearance:* appearance has almost no effect on validation, because only 4 of the 73 scored removals are thin. In a synthetic test a 2 cm book removed from a
table is kept by depth alone and deleted with appearance, and on a development pair it catches a real, unlisted removal (a robot vacuum); it is a targeted tool, not an overall gain. Alignment during the walk is as good as the offline
pre-pass on validation (perception moved F1 42.6 vs 40.4%), although it was 7.6 points worse on the tuning set; the choice was made on principle, since a graph built during the walk may not use future frames.

== How much of the room is observed

#figure(
  image("/figures/m7_fraction.png", width: 92%),
  caption: [FULL with only one contiguous chunk of 10 / 25 / 50% of the frames (mean ± sd over three chunk positions). Oracle blue, perception orange.],
) <fig-frac>

With less observation the method says _unknown_ instead of guessing (Figure 2). From the full sequence down to 10% of the frames, the unknown share grows from 31 to 78% (oracle) and from 49 to 92% (perception), while the false deletion rate falls
from 8.7 to 1.4% and from 7.3 to 1.4%: the intended behaviour, since less is seen through. During the walk (`lsg anytime`) perception moved F1 is 11 / 23 / 43% at 30 / 50 / 100% of the frames. 52% of the reference nodes change their
state at least once; moved nodes settle last (median at 56% of the walk), since they need the late view of their new place. This is the partial-observation case of the assignment; poses are taken as given and only the alignment is estimated.

== Relations

Parent accuracy of things for R0 (keep the reference relation), R1 (recompute geometrically from scratch) and ours (update). On validation, for all things the three give 94.9 / 79.8 / 92.2% in perception and
91.8 / 79.8 / 92.8% in oracle; for moved things 94.0 / 69.6 / 85.7% and 94.0 / 66.7 / 86.9%. R0 is a strong baseline because most moved things keep their parent ("cup moves from table to shelf" is rare in 3RScan). The update wins where parents really change (relation set, parent swapped, oracle: 0 / 42.9 / 35.7% with R0 at 0%, n = 14).
Keeping a parent that still supports the thing matters: without that rule moved things drop from 86.9 to 75.6%. The type (on / in) is right in 95–97% of cases.

== Audit of the ground truth, and comparison to related work

Because the annotation is incomplete, 60 false positives of FULL (10 per mode and change type) and 9 cases where the confidence kept an object that the ground truth lists as removed were checked by hand on photo panels.
A fraction of 28% [19, 41] of the false positives are real changes the annotation lacks. Of the sampled false deletions of unlisted objects, 7 of 8 (oracle) and 3 of 4 (perception) really are gone, so the false deletion rates above are upper bounds.
The false new objects, in contrast, are almost always method errors (1 of 20 real). In 8 of 9 veto cases the object really was gone, so the confidence rule costs real removals. The samples are small (10 per cell, one labeller). The method of Adam et al. @adam2023changed solves a different problem (offline, class-agnostic, regions instead of graph nodes); comparing with it would need a separate protocol at their box-IoU thresholds, which was not done.


= Conclusion

The report presented a method that maintains an object graph across two visits of a room. Free-space evidence, viewpoint decorrelation and deferred commitment make it cautious: it keeps unseen objects as unknown instead of deleting them (false deletion 95% → 7–9%),
and requires evidence at the old place before it calls an object moved. With the annotated instances, moves are found with an F1 of 57%; with perception from depth and CLIP, 43%. All numbers come from one run on the validation split,
after thresholds were chosen on 33 training pairs.

*Limitations.* New objects are the weak point: depth residuals mix real objects with static differences between the two scans, and no confidence separates them (3.5% precision). The detector raises recall but re-detected objects near emptied places take over removed nodes.
The confidence rule trades removed recall for safety, small moves (under 30 cm) and swaps of identical objects are often missed, and relation changes in 3RScan are few, so their evaluation has small counts.
Validation was looked at between early development steps; the decisions it influenced are listed in `docs/REPORT.md` (Section 8).

*Future work.* The most promising step is a better test for new objects: tracking detector masks and comparing them with the reference at the object level, instead of residual depth. A second step is a proper comparison with offline change detection
at the box-IoU thresholds of Adam et al., and estimating the rescan poses instead of using the given ones. Door swings, rotation-only moves and chains of several visits are out of scope.

*Use of AI tools.* Claude Code (Anthropic) was used for code, experiments and drafting; it is logged per milestone in `docs/AI_USAGE.md`.

#show bibliography: set text(size: 8pt)
#set par(leading: 0.4em, spacing: 0.4em)
#bibliography("/report/refs.bib", style: "ieee", title: "Bibliography")
