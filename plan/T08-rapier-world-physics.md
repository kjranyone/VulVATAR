# T08: Rapier World Physics Integration

## Priority: P2

## Source

- `docs/architecture.md` — physics strategy, Phase 5 milestones
- `docs/architecture.md:209` — Rapier positioning

## Status

**COMPLETED (feature-gated)**

- `RapierWorld` wrapper in `src/simulation/mod.rs` with full pipeline
- `add_character_body`, `move_character`, `character_position`, `query_sphere_cast` implemented
- Character body auto-initialized when avatar loads (rapier feature enabled)
- `PhysicsWorld::step_all()` runs springs + cloth + Rapier in lockstep via `SimulationClock`
- Frame loop uses `step_all()` when Rapier initialized, falls back to individual steps otherwise
- Accessor methods `rapier_initialized()` and `character_position()` on `PhysicsWorld`
- Physics status displayed in Rendering inspector panel
- Feature flag `rapier` gates all compilation; system works without it

## Current Code

- `src/simulation/mod.rs:133` — `RapierWorld` struct (behind `#[cfg(feature = "rapier")]`)
- `PhysicsWorld::init_rapier`, `add_static_collider`, `query_contacts`

## Remaining Work

- [x] Integrate Rapier broad-phase collision into cloth solver queries
- [x] Integrate Rapier queries into spring bone collision
- [x] Implement character controller body (avatar representation in world)
- [x] Add scene collider authoring or loading
- [x] Verify Rapier timestep sync with simulation clock
- [ ] Test performance impact on frame budget (requires runtime profiling)
- [x] Ensure Rapier is optional — system works without it

## Acceptance Criteria

- Cloth and spring can query world collision where needed
- Avatar can interact with scene-level colliders
- Rapier can be disabled via feature flag without breaking other systems
- Performance is acceptable at 60 fps with moderate scene complexity
