// Living Scene Graphs, Subproject D: short report. Sections 2 and 3 follow the points of the assignment.
// Compile:  uv run --with typst python docs/report/build.py   (or Tinymist / typst CLI with the project root at the repo or docs/: typst compile --root . docs/report/report.typ)
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

Keeping the scene graph of a room up to date sounds like a matter of spotting what changed. In practice the hard part is the opposite: deciding that an object
is really _gone_ and not just _unseen_. A rescan rarely covers the whole room. In the validation rooms of 3RScan, 737 reference objects are missing from the
rescan annotation without being listed as removed, while only 73 scored objects are listed as removed. A method that deletes everything it does not see would wipe out most of the graph.
My method therefore deletes an object only when the camera looks _through_ the place where it used to be.

The sections below follow the points of the assignment, and @sec-decisions collects the decisions I had to make myself. The code is in `src/lsg/`: change detection in
`change/` and `perception/`, the graph update in `change/update.py`, the evaluation in `eval/` and the figures in `viz/`.

= Data and preprocessing

== 3RScan, 3DSSG and how I split them

3RScan @wald2019rio has 1482 RGB-D scans of 478 rooms, many of them scanned again after people changed something, and 3DSSG @wald2020sg adds a scene graph to each scan.
I build the reference graph from 3DSSG: objects with a class and a 3D box, and the support relations `on` and `in`. From the rescan, the method gets only the RGB-D frames,
the camera poses and the rough alignment `T_align` to the reference.

I split the pairs into three sets. All thresholds were chosen on a *tuning set* of 33 training pairs (the 20 rooms with the most changes, 10 random rooms and 3 pairs I used during
development), and for each choice I wrote down the selection rule before running it. The *validation set* (110 pairs in 47 rooms) was run once for all tables. Because relation
changes turned out to be rare, I added a *relation set* of 40 training pairs with the most changed support parents (73 events).
== What the ground truth says <sec-gt>

Before building anything I looked at what the annotation actually says, and it changed how I score the results:
- *Missing is not the same as removed.* 1,018 reference objects are missing from the rescan annotation, but only 96 are listed as removed. I score the others separately as
  _absent-unlisted_. Deleting one of them counts as a _false deletion_, even though some of them really are gone (@sec-audit).
- *Many moves are tiny.* 134 of 370 annotated moves are shorter than 10 cm, close to the error of the given alignment. I treat moves below 10 cm as "don't care", which leaves 235 scored moves.
- *"New" is noisy.* Of 224 new ids, 171 are things or furniture. I do not score walls, floors and other structure as new.
- *Relation changes are rare.* Only 4 of 278 moved things change their support parent, which is why I added the relation set.

In total, validation has 235 moved, 73 removed, 171 new, 1,401 unchanged and 737 absent-unlisted objects.

== Preprocessing <sec-pre>

Everything is in metres, in the frame of the reference mesh. The given `T_align` is about 5 cm off, which is enough to confuse depth comparisons at object borders, so the method
refines it while walking: every 20 processed frames it registers the depth seen so far to the reference surfaces with point-to-plane ICP, and it accepts the result only if it stays
close to the given transform and is well constrained.
I process every third frame, in time order. For each one, Open3D @zhou2018open3d ray casting renders the reference mesh from the rescan pose (about 5 ms per frame on the CPU), which gives
the depth the camera _would_ see if nothing had changed. CLIP and DINOv2 features are computed once and cached.

= Method

I chose to run every experiment in two modes. In *oracle* mode the method gets the annotated rescan objects (with shuffled ids), so it only has to reason about the change; this is an
upper bound. In *perception* mode it has to find the objects itself from depth and appearance, and it names them with CLIP @radford2021clip. All results are reported for both.

== Detecting unchanged, moved, removed and new objects

*Votes from free space.* The core idea is to compare the observed depth $d_"obs"$ with the rendered depth $d_"exp"$ on the pixels where a reference node is the first surface hit.
Each frame gives one of three votes per node:
- *free*: the camera sees _through_ the node, $d_"obs" > d_"exp" + tau_"free"$ with $tau_"free" = 0.15$ m, on at least half of the valid pixels;
- *seen*: the surface is where it should be, $|d_"obs" - d_"exp"| <= 0.10$ m;
- *no information* otherwise. Occlusion, being out of view and missing depth say nothing about the object.

A seen vote only counts if there is something to contrast it with: a pillow lying on a sofa looks "in place" to depth even when it is gone, because the sofa is right behind it.
A vote also only counts if the camera has moved at least 0.30 m or turned 20° since the last counted vote. Otherwise someone filming one corner for ten seconds would cast thirty identical votes.

*Appearance for thin objects.* Depth cannot see a 2 cm book disappear from a table. For objects thinner than 15 cm, I therefore render the reference _texture_ from the rescan pose and
compare it with the photo using DINOv2 @oquab2023dinov2 patch features. A clear mismatch counts like a free vote. This is the "geometry + appearance" version; without it the method uses geometry only.

*New-object candidates.* In perception mode, new objects come from pixels that are more than 10 cm _nearer_ than the reference. I back-project them, drop the points that lie on the
reference surface or on walls, and collect the rest in a 4 cm voxel grid where each voxel keeps a log-odds. Seeing through a voxel lowers it, so a person who walked through the room
leaves nothing behind. Two neighbouring voxels are joined only if no frame puts them in different image segments, which keeps a new sofa apart from the cushions lying on it. Each cluster
is named by CLIP and becomes a `new` node only if its confidence is high enough.

*Deciding the states.* At the end of the walk, a node with three counted seen votes and no free vote is *confirmed* as `unchanged`. The other nodes are matched to the observations with the
Hungarian algorithm on label, size and position, and a matched node is `moved` if ICP of the observation onto the node shifts it by more than 10 cm. I added one rule that mirrors the
removal rule: *a move needs evidence at the old place*. If the old place still looks occupied, or nobody looked at it, the node may only be matched to an observation inside its own box.
An unmatched node becomes `removed` only with free evidence (@sec-policy); otherwise it stays `unknown`.

== Updating the graph <sec-update>

The update does the four things the assignment asks for.
*Moving nodes:* a moved node takes the ICP transform, and its box and centroid move with it.
*Changing anchor and relation:* moved and new things, and things whose parent moved or disappeared, are checked again. If the old parent still supports the thing at its new pose, it is
kept; otherwise contact and containment rules choose a new parent (an anchor, another thing or the floor) and the type `on` or `in`, and the anchor is recomputed up the chain.
*Deleting removed nodes:* I decided to keep a removed node in the file, marked `removed` together with its confidence, instead of erasing it. It never acts as a parent, and a later visit could bring it back.
*Adding new nodes:* each gets its CLIP label, box, points and a parent found by the same rules.
The graph is kept up to date _during_ the walk: every 20 processed frames the same steps produce a tentative graph (@sec-harder), and the last one is the result.

== Update policy: confidence and when to delete <sec-policy>

Every node starts with confidence $p = 0.9$. Each counted seen vote adds 0.63 to its log-odds (the object was seen again), and each counted free vote subtracts 1.47 (it should have been
visible and was not there). I did not pick these weights by hand; they are likelihood ratios measured on the tuning set.
A node is deleted only if it has free evidence (two counted free votes, or three appearance mismatches for thin objects) _and_ $p < 0.4$. Two free votes delete an untouched node;
one seen and two free votes do not.

Occluded and out-of-view objects get no votes at all, so their confidence does not move and they end up `unknown`, never `removed`. I read "should be visible but is not seen" strictly
as "the camera saw through it", so an object hidden behind something new is not penalized.
Finally, `moved` and `removed` are decided only at the end of the walk. A moved object can show up at its new place long after its old place was seen empty, and deciding early would turn
it into a removal plus a false new object.

== Baselines

To see what each idea is worth, I compare four methods that build on each other. *B0* matches objects by class and position and deletes everything that is unmatched. *B1* adds the free-space votes.
*B2* adds Hungarian matching and the ICP move test. *FULL* adds the old-place rule, appearance for thin objects, the measured confidence and the cleaner candidates.

== Decisions I made along the way <sec-decisions>

Apart from the scoring choices in @sec-gt, these are the decisions the assignment left to me. The values live in `configs/default.yaml`.

#figure(
  table(
    columns: (1.25fr, 1.1fr, 2.2fr),
    align: (left, left, left),
    table.header([*decision*], [*my choice*], [*why*]),
    [observation modes], [oracle and perception], [separates errors of the change logic from errors of perception],
    [frames used], [every 3rd], [consecutive frames are nearly identical],
    [free / seen margins], [0.15 m / 0.10 m], [above depth and alignment noise; I also tried $tau_"free"$ from 0.05 to 0.30 m, which trades removed recall against false deletion],
    [new viewpoint], [0.30 m or 20°], [a camera standing still should not cast many votes],
    [thin objects], [< 15 cm], [depth cannot see through anything thinner than $tau_"free"$],
    [confirmed / removed], [3 seen / 2 free votes], [one view is never enough],
    [confidence], [start 0.9, delete below 0.4], [an unseen node stays; the deletion threshold comes from a sweep on the tuning set],
    [move test], [ICP shift > 10 cm], [the same as the "don't care" threshold of the scoring],
    [new nodes (perception)], [≥ 7 views of the cluster], [best new F1 on the tuning set],
    [detector (Grounding DINO + SAM 2)], [off, tried as a variant], [it lost 6 points of removed F1 on the tuning set],
    [new object matches GT], [≤ 0.30 m or IoU ≥ 0.25, same class], [boxes of new objects are only partly seen; class = label or NYU40 class],
    [relations scored], [support parent and type of things], [`next_to` is computed but not scored],
    [poses], [given; only `T_align` refined], [I did not attempt the second harder case],
  ),
  caption: [Decisions the assignment did not fix.],
) <tab-dec>

= Evaluation

== Precision and recall for each change type

I compute precision and recall for each change type against the 3RScan ground truth. For new objects I report two versions: one that also requires the right class, and one that accepts
any label. Unchanged objects count as _kept_ if they stay `unchanged` or `unknown`, and the _false deletion_ rate is the share of absent-unlisted objects the method deleted.
Intervals are 95% bootstrap over pairs.

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

What I take from @tab-main:
- *Looking through old places is what matters most.* B0 deletes 95% of the absent-unlisted objects. With the same matching, B1 deletes only 9.1% (oracle) and 7.9% (perception), and removed
  precision rises from 21 to 65% (oracle).
- *ICP matching fixes moves* (oracle moved F1 31 → 56%), and the old-place rule makes them safer: false moves of absent-unlisted objects nearly halve (oracle 4.3 → 2.3%, perception 6.4 → 3.7%)
  and more unchanged objects are kept (oracle 85.8 → 90.3%). The price is large moves whose old place nobody looked at.
- *The confidence rule trades removals for safety:* FULL finds 33% of the removals in oracle, against 43% for B2.
- *New objects are where perception fails.* From depth residuals it finds only 7 of the 171 new objects with the right class (11 with any label). Of the 155 visible ones, only 62 get a
  candidate at all, and most of the 56 candidates per pair are static differences between the two scans, or clutter. An open-vocabulary detector (Grounding DINO @liu2023gdino with SAM 2
  @ravi2024sam2) raises new recall to 22%, but it cost removals on the tuning set, so I left it off.

== Are the updated relations right?

I compare the support parent and its type of every thing with the rescan's 3DSSG, for three policies: R0 keeps the reference relations, R1 recomputes all of them from scratch, and my
update (@sec-update) re-checks only what changed. Over all things, parent accuracy on validation is 94.9 / 79.8 / 92.2% (R0 / R1 / update) in perception and 91.8 / 79.8 / 92.8% in oracle;
for moved things it is 94.0 / 69.6 / 85.7% and 94.0 / 66.7 / 86.9%.
R0 is hard to beat here, because in 3RScan moved things almost always keep their parent; a cup moving from a table to a shelf is rare. Where parents really change (things with a swapped
parent in the relation set, oracle), the update gets 35.7% and R1 42.9%, while R0 gets 0% (n = 14). Keeping the old parent when it still carries the thing matters: without that rule,
moved things drop from 86.9 to 75.6%. When the parent is right, the type (`on` / `in`) is right in 95–97% of cases.

== Geometry only vs. geometry + appearance

#figure(
  table(
    columns: (1.6fr, 1.4fr, 1.4fr, 1fr),
    align: (left, right, right, right),
    table.header([*version*], [*oracle: rem. R / false del. / moved F1*], [*perception: rem. R / false del. / moved F1*], [*perc. thin: rem. R / kept*]),
    [geometry + appearance (FULL)], [32.9 / 8.7 / 57.4], [34.2 / 7.3 / 42.6], [1 of 4 / 91.9],
    [geometry only (depth)], [32.9 / 8.4 / 57.0], [32.9 / 7.1 / 42.6], [0 of 4 / 95.4],
  ),
  caption: [Geometry only vs. geometry + appearance on validation, in %. Thin = thinner than 15 cm (4 scored removals, 173 unchanged objects).],
) <tab-app>

Over the whole validation set, appearance changes almost nothing, simply because only 4 of the 73 scored removals are thin. On those it catches one removal that depth misses, and in
perception it costs 3.5 points of thin unchanged objects kept. It does matter in the case it was built for: in a synthetic test, a 2 cm book removed from a table is kept by depth alone and
deleted with appearance, and on a tuning pair it found a real removal that the annotation does not list (a robot vacuum). So appearance is a targeted tool for objects that depth cannot see,
not an overall gain.

== How much of the room has to be seen?

#figure(
  image("../figures/m7_fraction.png", width: 60%),
  caption: [FULL with only one contiguous piece of 10 / 25 / 50% of the frames (mean ± sd over three positions in the walk). Oracle blue, perception orange.],
) <fig-frac>

To test this, I run FULL on a single contiguous piece of the walk, 10, 25 or 50% of the frames, at three positions in the sequence (@fig-frac). With less to look at, the method says
_unknown_ instead of guessing. Going from the full walk down to 10%, the share of unknown nodes grows from 31 to 78% (oracle) and from 49 to 92% (perception), while false deletions fall from
8.7 to 1.4% and from 7.3 to 1.4%. That is the behaviour I wanted: what the camera has not seen through is not deleted.

== The harder case: few images and poses <sec-harder>

The 10% piece above is the "only a few images" case. The graphs written during the walk give a second view of it: while the camera is still walking, perception moved F1 is 11% after 30% of the
frames, 23% after 50% and 43% at the end, and false deletion grows from 1.2 to 3.3 to 7.3%. Half of the reference nodes (52%) change their state at least once during the walk. Moved nodes
settle last (median at 56% of the walk), because they need a late view of their new place; that is why `moved` is final only at the end. I did not estimate the rescan poses myself; only the
alignment to the reference is estimated, online (@sec-pre).

== Which parts of the method matter

#figure(
  table(
    columns: (1.9fr, 1.25fr, 1.25fr, 0.9fr),
    align: (left, right, right, right),
    table.header([*variant* (differs from FULL in one part)], [*oracle: rem. R / false del. / moved F1*], [*perception: rem. R / false del. / moved F1*], [*perc. new P / R*]),
    [FULL], [32.9 / 8.7 / 57.4], [34.2 / 7.3 / 42.6], [3.5 / 4.1],
    [without the old-place rule], [37.0 / 8.7 / 55.5], [37.0 / 7.6 / 37.8], [3.2 / 3.5],
    [hand-set vote weights], [30.1 / 8.1 / 55.6], [30.1 / 6.9 / 40.1], [4.9 / 5.8],
    [without image segments], [as FULL], [34.2 / 6.9 / 38.7], [4.4 / 2.3],
    [alignment from all frames first], [28.8 / 9.0 / 54.6], [35.6 / 7.9 / 40.4], [3.6 / 4.1],
    [every frame votes], [34.2 / 9.2 / 57.6], [37.0 / 8.4 / 42.0], [2.2 / 6.4],
    [deciding immediately], [52.1 / 10.2 / 23.8], [52.1 / 10.2 / 15.3], [5.2 / 7.6],
    [my earlier method (class priors, strict candidates)], [27.4 / 8.1 / 54.6], [32.9 / 7.7 / 31.7], [2.0 / 8.8],
  ),
  caption: [FULL with one part changed at a time, on validation, in %.],
) <tab-abl>

*Deciding late matters most* (@tab-abl). If a node is deleted as soon as its place looks empty, objects that merely moved are removed, and their observations at the new place turn into
false new objects: removed precision falls from 58% to 17% (oracle), and oracle moved F1 from 57% to 24%. Aligning during the walk turned out as good as aligning with all frames in advance
(perception moved F1 42.6 vs 40.4%), although it was 7.6 points worse on the tuning set. I kept it anyway, because a graph built during the walk should not use frames that have not been recorded yet.

== The graph before and after

#figure(
  placement: auto,
  image("../figures/topdown_val_8eabc45f_perception.png", width: 76%),
  caption: [Validation room 8eabc45f → 5341b7b3 (perception), top-down. Left: reference graph. Middle: updated graph (grey unchanged, dotted unknown, blue moved, hatched red removed, green new). Right: ground truth.
    The two pillows removed from the sofa are deleted and some pillow moves are found; the sofa stays _unknown_, the lamp is moved to a wrong place and two false new objects appear.],
) <fig-room>

#figure(
  placement: auto,
  image("../figures/graph_before_after.png", width: 66%),
  caption: [The same room as a graph, changed part only, before and after the update. Arrows go from a thing to its support parent (solid `on`, dashed `in`). Removed pillows and a vase, moved pillows and a lamp,
    and two new nodes ("clothes" in the sofa, "table lamp" in a cabinet; both false) appear; the sofa and a table stay unknown.],
) <fig-graph>

@fig-room and @fig-graph show the graph of one room before and after the update. I picked the room by a fixed rule (the most scored changes in the milestone subset), not by how good it
looks. With the annotated instances, the method also finds the moved sofa. `lsg viz topdown` draws the same figures for any pair.

== Is the ground truth right? <sec-audit>

Since the annotation is incomplete, I checked 60 false positives of FULL by hand on photo panels (10 per mode and change type), plus 9 objects that the ground truth lists as removed but the
confidence rule kept. In 28% [19, 41] of the false positives the method was right and the annotation misses the change. Of the sampled false deletions of unlisted objects, 7 of 8 (oracle) and
3 of 4 (perception) really are gone, so the false deletion rates above are upper bounds. False new objects, on the other hand, are almost always real errors (1 of 20 was a real new object).
In 8 of the 9 kept cases the object really was gone, so the confidence rule does cost real removals. These samples are small and were labelled by one person. A direct comparison with
Adam et al. @adam2023changed, who report class-agnostic box-IoU recall on the same split, would need their protocol, and I did not do it.

= Conclusion

I built a method that keeps an object graph up to date across two visits of a room. Looking through old places, counting only votes from new viewpoints and deciding at the end make it
cautious: it keeps unseen objects as unknown instead of deleting them (false deletion 95% → 7–9%), and it wants evidence at the old place before it calls something moved. With the annotated
instances it finds moves with an F1 of 57%, with its own perception 43%. All numbers come from a single run on validation, after every threshold was fixed on 33 training pairs.

*Limitations.* New objects are the weak spot. Depth residuals mix real new objects with static differences between the two scans, and nothing in the confidence separates them (3.5% precision).
The detector finds more of them, but objects it re-detects near emptied places then take over removed nodes. The confidence rule gives up some removals for safety, small moves (under 30 cm)
and swaps of identical objects are often missed, and there are few relation changes to learn from. I did look at validation results between early development steps. They led to excluding the catch-all classes when matching new objects, to "seen needs contrast", to the
relaxed candidate rules, to dropping the class prior and to trying the detector; after the tuning set was introduced, every choice was made there with a rule written beforehand.

*Future work.* The most useful next step would be a better test for new objects: tracking detector masks and comparing them with the reference object by object, instead of looking at
residual depth. After that, a proper comparison with Adam et al. at their box-IoU thresholds, and estimating the rescan poses instead of taking them as given. Door swings, rotation-only
moves and chains of several visits are out of scope.

*Use of AI tools.* I used Claude Code (Anthropic) for code, experiments and drafting.

#show bibliography: set text(size: 8pt)
#set par(leading: 0.4em, spacing: 0.4em)
#bibliography("refs.bib", style: "ieee", title: "Bibliography")
