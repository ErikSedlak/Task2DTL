// Living Scene Graphs, Subproject D: short report. Sections 3 and 4 follow the points of the assignment.
// Compile:  uv run --with typst python docs/report/build.py
// Numbers: outputs/chg3_final (val, one run), outputs/chg3_final_anytime, outputs/chg3_audit.

#set page(paper: "a4", margin: (x: 1.5cm, top: 1.3cm, bottom: 1.4cm), numbering: "1")
#set text(font: "New Computer Modern", size: 9pt, lang: "en")
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

#heading(numbering: none)[Introduction]

The hard part of keeping a 3RScan scene graph up to date is not to see a change but to tell an object that is _gone_ from one that was _not seen_.
A rescan covers only part of the room: on validation, 737 reference objects are missing from the rescan annotation without being listed as removed, against 73 scored removals.
The method below therefore deletes an object only when the sensor looks _through_ its old place. Sections 3 and 4 follow the points of the assignment;
the choices the assignment leaves open are marked _(our choice)_ in Section 2 and collected in Section 3.5.
Code: `src/lsg/` (change detection in `change/` and `perception/`, graph update in `change/update.py`, evaluation in `eval/`, figures in `viz/`).

= Related work

*Data.* 3RScan @wald2019rio contains 1482 RGB-D scans of 478 rooms, many of them re-scanned after people changed the room, with annotations of
which objects moved, disappeared or appeared. 3DSSG @wald2020sg adds scene graphs (objects, support and proximity relations) on top of it.
The original 3RScan task is to re-localize objects in a changed room; graph maintenance is a different use of the same data.

*Scene graphs over time.* SceneGraphFusion @wu2021sgfusion builds a graph incrementally from an RGB-D sequence, but does not model changes between visits.
3D dynamic scene graphs @rosinol2020dsg add places, objects and humans to a map built by SLAM; they focus on people and places rather than on objects that
were rearranged between visits.

*Change detection on 3RScan.* The closest work is "Has Anything Changed?" by Adam et al. @adam2023changed. They render the depth of both reconstructed meshes,
take depth differences as seeds, and grow them to whole objects with SAM masks. They report class-agnostic recall at box-IoU thresholds on the same 110 validation pairs,
offline and on fused meshes, and do not associate objects across visits, decide between removed and not seen, or update relations.

*Models used.* Frozen models are used for appearance only: CLIP @radford2021clip to name candidates, DINOv2 @oquab2023dinov2 to compare reference and rescan
textures, and, for one ablation, Grounding DINO @liu2023gdino with SAM 2 @ravi2024sam2. Rendering and ICP use Open3D @zhou2018open3d.

= Data and preprocessing

== 3RScan, 3DSSG and the splits

The reference graph is built from 3DSSG (objects with class and 3D box, support relations `on` / `in`); the rescan's RGB-D frames, poses and the rough alignment `T_align` are the input.
Three sets of reference–rescan pairs are used _(our choice)_. *Tuning set*: 33 training pairs (the 20 rooms with the most changes, 10 random rooms, 3 development pairs); every
threshold was chosen there, each with a selection rule written down before the run. *Validation*: 110 pairs in 47 rooms, run once for all tables. *Relation set*:
40 training pairs with the most changes of support parents (73 events), never tuned on.

== What the ground truth says, and how it is scored

The annotation was checked before building anything, and the scoring follows from it:
- *Absent is not removed.* 1,018 reference objects of the validation rooms are missing in the rescan annotation, but only 96 are listed as removed. _Our choice:_ the others are
  scored separately as _absent-unlisted_; deleting one is a _false deletion_ (lower is better), although some really are gone (Section 4.8).
- *Many moves are tiny.* 134 of 370 annotated moves are shorter than 10 cm, comparable to the 5 cm error of the given alignment. _Our choice:_ moves below 10 cm are "don't care" (neither moved nor unchanged), which leaves 235 scored moves.
- *"New" is noisy.* Of 224 new ids, 171 are things and furniture. _Our choice:_ walls, floors and other structure are not scored as new.
- *Relation changes are rare.* Only 4 of 278 moved things swap their support parent, so _(our choice)_ the relation set above was added.

Scored ground truth on validation: 235 moved, 73 removed, 171 new, 1,401 unchanged and 737 absent-unlisted objects.

== Preprocessing

Units are metres and the reference mesh frame is the common frame. The provided `T_align` is about 5 cm off, enough
to confuse depth comparison at object borders, so it is refined during the walk: every 20 processed frames one point-to-plane ICP registers the depth of all frames
so far to the reference surfaces, accepted only if it is close to the raw one and not degenerate. No future frame is used.
Every third frame is processed, in time order. For each frame the reference mesh is rendered at the rescan pose with Open3D ray casting
(about 5 ms on the CPU), giving the depth the camera _would_ see if nothing had changed. CLIP and DINOv2 features are computed on the GPU (GTX 1650) and cached.

= Method

Two observation modes share one interface _(our choice)_. In *oracle* mode the rescan objects are the annotated rescan instances with shuffled ids, an upper bound that
isolates the change logic. In *perception* mode the method finds the objects itself from depth and appearance and names them with CLIP. Every result is reported for both.

== Detecting unchanged, moved, removed and new objects

*Votes from free space.* On the pixels where a reference node is first hit, the observed depth $d_"obs"$ is compared with the rendered $d_"exp"$. A frame gives one vote per node:
*free* when the sensor sees _through_ the node ($d_"obs" > d_"exp" + tau_"free"$, $tau_"free" = 0.15$ m, on half of the valid pixels); *seen* when the surface is in place ($|d_"obs" - d_"exp"| <= 0.10$ m);
otherwise _no information_: occlusion, out of view and invalid depth say nothing. A seen vote counts only if the surface behind the node is farther than $tau_"free"$ (a pillow on a sofa cannot be seen in place by depth).
A vote counts only after _viewpoint decorrelation_: the camera must have moved 0.30 m or turned 20° since the last counted vote, so a person filming one spot casts one vote, not thirty.

*Appearance for thin objects.* Depth cannot see a 2 cm book vanish. For nodes thinner than 15 cm the reference _texture_ is rendered at the rescan pose and
compared with the photo through DINOv2 patch features; a mismatch is a _free-like_ vote. This is the "geometry + appearance" variant; without it the method uses geometry only.

*New-object candidates (perception).* Pixels more than 10 cm nearer than the rendered reference are back-projected; points near the reference surface or structure are dropped. The rest
is accumulated in a 4 cm voxel grid with a log-odds per voxel; seeing through a voxel lowers it, so a person who walks away leaves nothing behind. Voxels are linked only if no frame
puts them in different image segments, which keeps a new sofa apart from its cushions. Each cluster is named by CLIP and becomes a `new` node only above a confidence threshold.

*States.* At the end of the walk a node with three counted seen votes and no free vote is *confirmed* `unchanged`. The other nodes and the observations are matched by the
Hungarian algorithm on label, size and position; a matched node is `moved` if ICP of the observation onto the node moves its centroid by more than 10 cm.
*A move needs evidence at the old place*: a node whose old place is net-seen, or was never observed, may only be matched inside its own place. An unmatched node is `removed` only
with free evidence (Section 3.3), otherwise `unknown`.

== Updating the graph

*Move nodes:* a moved node gets the ICP transform; its box and centroid are moved with it. *Change anchor and relation:* moved and new things, and things whose parent moved or vanished, are re-checked.
A thing keeps its old parent if that parent still supports it at the new pose; otherwise contact and containment rules on the points choose a new parent (an anchor, another thing or the floor)
and the type `on` or `in`, and the `anchor` field is recomputed up the chain. *Delete removed nodes:* a removed node stays in the file, marked `removed` with its confidence, and never acts as a parent;
deletion is a view filter _(our choice, so a later visit could restore it)_. *Add new nodes:* with the CLIP label, box, points and a parent from the same rules.
The graph is maintained _during_ the walk: after every 20 processed frames the same steps give a tentative graph (Section 4.5); the last one is the result.

== Update policy: confidence and when to delete

Every node starts at $p = 0.9$. A counted seen vote adds 0.63 to the log-odds (the object is re-observed); a counted free vote subtracts 1.47 (it should be visible and is not there).
The weights are likelihood ratios measured on the tuning set, not set by hand. *Deletion rule:* a node is deleted only with free evidence (two counted free votes, or three appearance votes for thin nodes) _and_ $p < 0.4$;
two free votes delete an untouched node, one seen plus two free do not. *Occluded and out-of-view objects* get no vote, so their confidence does not change and they end as `unknown`,
never `removed`. "Not seen although it should be visible" is thus read strictly as "seen through" _(our choice)_: a node in view but hidden behind a new object is not penalized.
`moved` and `removed` are final only at the end of the walk (_deferred commitment_): a moved object can appear at its new place long after its old place was seen empty.

== Baselines

*B0* matches by class and position and deletes whatever is unmatched; *B1* adds the votes; *B2* adds association and the ICP move test; *FULL* adds old-place evidence,
thin-object appearance, the measured confidence and the cleaner candidates.

== Decisions not fixed by the assignment

Besides the scoring choices of Section 2.2, the following were our decisions. Thresholds live in `configs/default.yaml`; the rationale and the alternatives are in `docs/DECISIONS.md`.

#figure(
  table(
    columns: (1.25fr, 1.1fr, 2.2fr),
    align: (left, left, left),
    table.header([*decision*], [*value*], [*why*]),
    [frames used], [every 3rd], [consecutive frames are nearly identical],
    [free / seen margins], [0.15 m / 0.10 m], [above depth and alignment noise; $tau_"free"$ sweep 0.05–0.30 m in `docs/REPORT.md`],
    [viewpoint decorrelation], [0.30 m or 20°], [a camera that stands still must not cast many votes],
    [thin objects (appearance)], [< 15 cm], [below $tau_"free"$ depth cannot see through them],
    [confirmed / removed], [3 seen / 2 free votes], [one view is never enough],
    [confidence start, deletion], [p = 0.9, p < 0.4], [an unseen node stays; $p_"del"$ from a sweep on the tuning set],
    [move test], [ICP > 10 cm], [same as the "don't care" threshold of the scoring],
    [new-node gate (perception)], [≥ 7 cluster views], [best new F1 on the tuning set],
    [detector (Grounding DINO + SAM 2)], [off, ablation row], [lost 6 points of removed F1 on the tuning set (pre-written rule)],
    [new matched to GT], [≤ 0.30 m or IoU ≥ 0.25, same class], [boxes of new objects are partial; class = label or NYU40 class],
    [relations scored], [support parent and type of things], [`next_to` is computed but not scored],
    [poses], [given, `T_align` refined], [the second harder case (own poses) was not attempted],
  ),
  caption: [Choices not fixed by the assignment (method and protocol).],
) <tab-dec>

= Evaluation

== Precision and recall for each change type

Precision and recall are computed per change type against the 3RScan ground truth: removed, moved and new (the labelled variant needs the class; _any label_ is
also reported). Unchanged objects are scored as _unchanged kept_ (the share that stays `unchanged` or `unknown`). *False deletion* is the share of absent-unlisted objects that were deleted. Intervals are 95% bootstrap over pairs.

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

- *Free-space evidence is the biggest win.* B0 deletes 95% of the absent-unlisted objects; B1, with the same matching, 9.1% (oracle) and 7.9% (perception). Removed precision rises from 21 to 65% (oracle).
- *Association with ICP fixes moves* (oracle moved F1 31 → 56%), and old-place evidence makes them safe: it nearly halves the false moves of absent-unlisted objects (oracle 4.3 → 2.3%, perception 6.4 → 3.7%) and keeps
  more unchanged objects (oracle 85.8 → 90.3%), at the price of large moves whose old place was never observed.
- *The confidence costs removed recall* (FULL 33% against B2's 43% in oracle) in exchange for fewer wrong deletions.
- *Perception loses most on new objects.* From depth residuals it finds 7 of 171 new objects with the right label (11 with any label). Of 155 visible new objects only 62 get any candidate, and most of the 56 candidates per pair are
  static scan differences or clutter. The detector raises new recall to 22% but costs removals, so it is not the default.

== Accuracy of the updated relations vs. 3DSSG

The support parent and its type of every thing in the updated graph are compared with the rescan's 3DSSG. Three policies: R0 keeps the reference relation, R1 recomputes all relations geometrically from scratch, and ours updates them (Section 3.2).
On validation, parent accuracy over all things is 94.9 / 79.8 / 92.2% (R0 / R1 / ours) in perception and 91.8 / 79.8 / 92.8% in oracle; for moved things 94.0 / 69.6 / 85.7% and 94.0 / 66.7 / 86.9%.
R0 is strong because most moved things keep their parent ("cup moved from table to shelf" is rare in 3RScan). The update wins where parents really change: on the relation set, for things whose parent was swapped, oracle gives 0 / 42.9 / 35.7% (n = 14).
Keeping a parent that still supports the thing matters: without that rule moved things drop from 86.9 to 75.6%. Given the right parent, the type (`on` / `in`) is right in 95–97% of cases.

== Ablation: geometry only vs. geometry + appearance

#figure(
  table(
    columns: (1.6fr, 1.4fr, 1.4fr, 1fr),
    align: (left, right, right, right),
    table.header([*variant*], [*oracle: rem. R / false del. / moved F1*], [*perception: rem. R / false del. / moved F1*], [*perc. thin: rem. R / kept*]),
    [geometry + appearance (FULL)], [32.9 / 8.7 / 57.4], [34.2 / 7.3 / 42.6], [1 of 4 / 91.9],
    [geometry only (depth)], [32.9 / 8.4 / 57.0], [32.9 / 7.1 / 42.6], [0 of 4 / 95.4],
  ),
  caption: [Appearance ablation on validation, in %. Thin = thinner than 15 cm (4 scored removals, 173 unchanged objects).],
) <tab-app>

Appearance has almost no effect overall, because only 4 of the 73 scored removals are thin. On those it catches one removal that depth misses, and in perception costs 3.5 points of thin unchanged objects kept. In a synthetic test a 2 cm book removed from a table is kept by depth alone and deleted with appearance, and on a tuning pair it catches a real, unlisted removal (a robot vacuum):
a targeted tool for objects depth cannot see, not an overall gain.

== Ablation: how much of the room is observed

#figure(
  image("/figures/m7_fraction.png", width: 60%),
  caption: [FULL with only one contiguous chunk of 10 / 25 / 50% of the frames (mean ± sd over three chunk positions). Oracle blue, perception orange.],
) <fig-frac>

FULL is run on one contiguous chunk of 10, 25 or 50% of the frames, at three positions in the sequence. With less observation the method says _unknown_ instead of guessing (@fig-frac): from the full sequence down to 10%, the unknown share grows from 31 to 78% (oracle) and from 49 to 92% (perception),
while the false deletion rate falls from 8.7 to 1.4% and from 7.3 to 1.4%. This is the intended behaviour: what is not seen through is not deleted.

== Harder case: partial observation and poses

The 10% chunk above is the "only a few images" case: a short piece of the walk instead of the full sequence. A second view of partial observation is the graph _during_ the walk (`lsg anytime`): perception moved F1 is 11 / 23 / 43% at
30 / 50 / 100% of the frames, and false deletion 1.2 / 3.3 / 7.3%. 52% of the reference nodes change their state at least once; moved nodes settle last (median at 56% of the walk), since they need the late view of their new place,
which is why `moved` is final only at the end. The rescan poses were taken as given; only the alignment to the reference is estimated, online (Section 2.3).

== Other ablations

#figure(
  table(
    columns: (1.9fr, 1.25fr, 1.25fr, 0.9fr),
    align: (left, right, right, right),
    table.header([*row* (each differs from FULL in one part)], [*oracle: rem. R / false del. / moved F1*], [*perception: rem. R / false del. / moved F1*], [*perc. new P / R*]),
    [FULL], [32.9 / 8.7 / 57.4], [34.2 / 7.3 / 42.6], [3.5 / 4.1],
    [− old-place evidence], [37.0 / 8.7 / 55.5], [37.0 / 7.6 / 37.8], [3.2 / 3.5],
    [− measured vote weights], [30.1 / 8.1 / 55.6], [30.1 / 6.9 / 40.1], [4.9 / 5.8],
    [− co-segmentation], [as FULL], [34.2 / 6.9 / 38.7], [4.4 / 2.3],
    [− online alignment (pre-pass over all frames)], [28.8 / 9.0 / 54.6], [35.6 / 7.9 / 40.4], [3.6 / 4.1],
    [− viewpoint decorrelation], [34.2 / 9.2 / 57.6], [37.0 / 8.4 / 42.0], [2.2 / 6.4],
    [\+ immediate commitment], [52.1 / 10.2 / 23.8], [52.1 / 10.2 / 15.3], [5.2 / 7.6],
    [the earlier method (class priors, strict candidates)], [27.4 / 8.1 / 54.6], [32.9 / 7.7 / 31.7], [2.0 / 8.8],
  ),
  caption: [Further ablations on validation, in %.],
) <tab-abl>

*Deciding late is the clearest result.* Deleting a node as soon as its place looks empty removes objects that merely moved; their observations at the new place then become false new nodes. Removed precision falls from 58% to 17%
(oracle), and oracle moved F1 from 57% to 24%. Alignment during the walk is as good as the offline pre-pass on validation (perception moved F1 42.6 vs 40.4%), although it was 7.6 points worse on the tuning set;
it was kept because a graph built during the walk may not use future frames.

== Visualization of the graph before and after

#figure(
  placement: auto,
  image("/figures/topdown_val_8eabc45f_perception.png", width: 76%),
  caption: [Validation room 8eabc45f → 5341b7b3 (perception), top-down. Left: reference graph. Middle: updated graph (grey unchanged, dotted unknown, blue moved, hatched red removed, green new). Right: ground truth.
    The two pillows removed from the sofa are deleted and some pillow moves are found; the sofa stays _unknown_, the lamp is moved wrongly and two false new objects appear.],
) <fig-room>

#figure(
  placement: auto,
  image("/figures/graph_before_after.png", width: 66%),
  caption: [The same room as a graph, changed part only, before and after the update. Arrows go from a thing to its support parent (solid `on`, dashed `in`). Removed pillows and a vase, moved pillows and a lamp,
    and two new nodes ("clothes" in the sofa, "table lamp" in a cabinet; both false) appear; the sofa and a table stay unknown.],
) <fig-graph>

@fig-room and @fig-graph show the graph before and after the update in one room, chosen by a fixed rule (the most scored changes in the milestone subset), not by how it looks. The oracle run on the same room also finds the moved sofa.
Figures for any pair: `lsg viz topdown`.

== Audit of the ground truth

Because the annotation is incomplete, 60 false positives of FULL (10 per mode and change type) and 9 cases where the confidence kept an object that the ground truth lists as removed were checked by hand on photo panels.
28% [19, 41] of the false positives are real changes the annotation lacks. Of the sampled false deletions of unlisted objects, 7 of 8 (oracle) and 3 of 4 (perception) really are gone, so the false deletion rates above are upper bounds.
False new objects are almost always method errors (1 of 20 real). In 8 of 9 veto cases the object really was gone, so the confidence rule costs real removals. The samples are small (10 per cell, one labeller).
Comparing with Adam et al. @adam2023changed would need their offline, class-agnostic protocol at box-IoU thresholds, which was not done.

= Conclusion

The method maintains an object graph across two visits of a room. Free-space evidence, viewpoint decorrelation and deferred commitment make it cautious: it keeps unseen objects as unknown instead of deleting them (false deletion 95% → 7–9%),
and requires evidence at the old place before it calls an object moved. With the annotated instances, moves are found with an F1 of 57%; with perception from depth and CLIP, 43%. All numbers come from one run on the validation split,
after thresholds were chosen on 33 training pairs.

*Limitations.* New objects are the weak point: depth residuals mix real objects with static differences between the two scans, and no confidence separates them (3.5% precision). The detector raises recall but re-detected objects near emptied places take over removed nodes.
The confidence rule trades removed recall for safety, small moves (under 30 cm) and swaps of identical objects are often missed, and relation changes are few. Validation was looked at between early development steps; the decisions it influenced are listed in `docs/REPORT.md` (Section 8).

*Future work.* A better test for new objects: tracking detector masks and comparing them with the reference at the object level, instead of residual depth. Also the box-IoU comparison with Adam et al.,
and estimating the rescan poses instead of using the given ones. Door swings, rotation-only moves and chains of several visits are out of scope.

*Use of AI tools.* Claude Code (Anthropic) was used for code, experiments and drafting; it is logged per milestone in `docs/AI_USAGE.md`.

#show bibliography: set text(size: 8pt)
#set par(leading: 0.4em, spacing: 0.4em)
#bibliography("/report/refs.bib", style: "ieee", title: "Bibliography")
