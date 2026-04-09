# MToon Test Matrix

## Purpose

This document defines how MToon-like rendering progress should be evaluated during implementation.

It is a practical validation checklist, not a shader design document.

Related documents:

- [mtoon-compatibility.md](/C:/lib/github/kjranyone/VulVATAR/docs/mtoon-compatibility.md)
- [shader-implementation-notes.md](/C:/lib/github/kjranyone/VulVATAR/docs/shader-implementation-notes.md)

## Core Rule

Evaluate rendering against real representative avatars.

Do not judge compatibility only by whether parameters exist in code.

## Representative Test Areas

Each test avatar should be checked for:

- face readability
- hair transparency behavior
- outline readability
- culling correctness
- overall silhouette

## Must-Pass Early Checks

### Face

Check:

- face is not excessively dark
- face shadow threshold looks stable under slight camera movement
- front-lit face remains readable

### Hair

Check:

- alpha cutout or blend does not visibly break
- culling mode does not create obvious missing surfaces
- outline does not produce major artifacts if enabled

### Outline

Check:

- outline is visible when intended
- width changes are perceptible and stable
- outline does not disappear unpredictably across camera angles

### Alpha

Check:

- cutout behaves like cutout
- blend behaves like blend
- alpha output remains usable for compositing

## Nice-To-Have Early Checks

- simple lighting feels coherent across body and face
- shadow softness changes produce visible but controlled differences
- toon threshold changes are understandable in the UI

## Deferred Checks

These can wait until the basic renderer is stable:

- long-tail stylistic parameters
- edge-case material combinations
- high-fidelity parity on uncommon avatars

## Regression Signals

Treat these as regressions:

- face quality improves on one avatar and breaks on another common one
- hair transparency becomes unstable after outline changes
- alpha output becomes incompatible with compositing expectations
- parameter changes stop producing predictable visible changes

## Test Session Format

For each renderer milestone:

1. open representative avatar set
2. inspect face under front and side lighting
3. inspect hair and transparent regions
4. inspect outline readability
5. inspect compositing or alpha output if enabled
6. record pass or fail with screenshots later if desired

## Failure Modes To Avoid

- validating only against one avatar
- ignoring face and hair quality while pursuing low-priority parameters
- calling a parameter supported when it has no meaningful visible effect
