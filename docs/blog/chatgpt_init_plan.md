
Yes—I understand exactly what you mean. The blog should be a personal engineering story and a practical tutorial, gradually developing RoboBall from a simple moving sphere into a physics-based robot with reusable and composable skills.

## Recommended central idea

The article’s main message should be:

> Complex robot behavior can be built from small, verified motion primitives. First control the physics, then create reusable skills, and finally compose those skills into behaviors such as climbing stairs.

The current article should focus on low-level control, mid-level skills, and skill composition. IL/RL should appear only as the future high-level layer.

## Possible titles

My preferred title:

**Building RoboBall: From an Old Idea to a Robot That Moves, Jumps, and Climbs Stairs**

Alternatives:

- **Teaching a Ball to Move, Jump, and Climb**
- **From 60 Telescoping Rods to Composable Robot Skills**
- **Building a Telescopic Robot Ball in MuJoCo**

## Proposed narrative structure

### 1. Opening: an idea from my bachelor studies

Begin with your personal story:

- You imagined a robotic ball during your early bachelor studies.
- At the time, you did not have a clear application or enough time to develop it.
- The idea remained in the background for several years.
- Coding assistants lowered the implementation barrier and encouraged you to revisit it.
- While building it, possible applications became clearer.

End the introduction with the final stairs video as a hook:

> This article explains how that old idea became a simulated robot that can roll, steer, jump, stop, and combine those abilities to climb stairs.

### 2. What we will build

Briefly tell readers what they will learn:

- How the telescopic RoboBall works.
- How its rods generate motion.
- Why coordinate frames matter.
- How MuJoCo simulates contacts and actuators.
- How movement and jumping become reusable skills.
- How skills are composed into stair traversal.

Also establish the tutorial philosophy:

> Every concept will be introduced only when it becomes necessary.

### 3. Introducing RoboBall

Explain the mechanism in approachable language:

- A central spherical core.
- 60 telescoping rods.
- Rods distributed approximately uniformly using a Fibonacci sphere.
- Each rod has a sleeve, an extending shaft, and a foot.
- Extending rods against the ground displaces the core and generates motion.
- Different rod patterns produce rolling, braking, jumping, and bracing.

Useful equation:

\[
\hat u_i^{body} =
\begin{bmatrix}
\sqrt{1-z_i^2}\cos\phi_i\\
\sqrt{1-z_i^2}\sin\phi_i\\
z_i
\end{bmatrix}
\]

Briefly explain that this distributes the rods around the sphere without requiring a regular latitude-longitude grid.

Suggested visual: [first visible rods](/home/azureuser/telescopic_robot/docs/project_journey/assets/01_first_visible_bars.png).

### 4. Start simple: a ball that follows a path

Before introducing all 60 actuators, begin with the simplest problem:

- Represent the robot as a moving sphere.
- Define a target point or path waypoint.
- Compute the desired direction:

\[
\hat d = \frac{p_{target}-p_{robot}}
{\|p_{target}-p_{robot}\|}
\]

- Continuously update the target direction.
- Show the robot following a simple path and performing a round trip.

This gives readers an immediate result before introducing contact physics.

Suggested video: [round-trip motion](/home/azureuser/telescopic_robot/docs/project_journey/assets/04_roundtrip_telescoping.mp4).

### 5. From abstract motion to physical motion

Now explain why directly changing the ball’s position is insufficient.

Introduce:

- Gravity.
- Mass and inertia.
- Contact forces.
- Friction.
- Actuator force limits.
- Collisions between feet and terrain.
- The MuJoCo simulation timestep.

Explain why MuJoCo was introduced and how it turned animation into an actual control problem.

A small contact equation is enough:

\[
F_t \leq \mu F_n
\]

Explain that a foot can only generate limited tangential force before it slips.

This section can include one instructive failure, such as feet sinking into the floor or rods extending without producing useful motion.

### 6. The control hierarchy

Introduce the three-level architecture with a diagram:

```text
Future high-level controller
IL / RL / task planner
          ↓ selects skills
Mid-level skill library
move, stop, jump, fall, turn
          ↓ produces 60 targets
Low-level controller and physics
rod servos, force limits, contacts, friction
```

Clarify the scope:

- Low level: actuator targets, tracking, contact and physical constraints.
- Mid level: named behaviors such as `move`, `jump_to`, and `fall_down`.
- High level: deciding which skill to use and when. This is where future IL/RL belongs.

### 7. Low-level control: deciding which rods should move

This is the central technical tutorial section.

First rotate every rod from the body frame into the world frame:

\[
\hat u_i^{world}=R(q)\hat u_i^{body}
\]

Then project each rod relative to the requested travel direction:

\[
u_i^\parallel = \hat u_i\cdot(d_x,d_y,0)
\]

\[
u_i^\perp = \hat u_i\cdot(-d_y,d_x,0)
\]

\[
u_i^z = \hat u_{i,z}
\]

Explain these visually:

- \(u^\parallel\): front versus rear.
- \(u^\perp\): left versus right.
- \(u^z\): top versus bottom.

Then show how a smooth weight \(w_i\) becomes a rod-extension target:

\[
\ell_i^* = \ell_{stance} + w_i\ell_{stroke}
\]

If appropriate, briefly explain the actuator position controller:

\[
e_i = \ell_i^*-\ell_i
\]

\[
F_i = K_p e_i + K_i\int e_i\,dt
\]

We should confirm whether the final article should call this PI control or MuJoCo position-servo control, based on the exact implementation.

### 8. Building primitive skills

Introduce a small, carefully chosen subset rather than listing everything.

#### Move and turn

Explain that the robot has no permanent “front.” The commanded direction defines forward at every timestep.

Show how turning is simply changing \(\hat d\), rather than inventing separate left/right mechanisms.

#### Stop

Explain active braking and why setting all commands to zero does not immediately stop a rolling sphere.

#### Jump

Introduce the jump state machine:

```text
crouch → takeoff → airborne → landing
```

Use basic projectile motion:

\[
z(t)=z_0+v_z t-\frac{1}{2}gt^2
\]

\[
x(t)=x_0+v_x t
\]

Explain that the controller does not merely “extend everything.” It must:

- Retract before launch.
- Fire the correct ground-facing rods.
- Maintain forward velocity.
- Tuck during flight.
- Extend compliant landing rods.

Suggested visual: [jump progression](/home/azureuser/telescopic_robot/docs/project_journey/assets/forward_jump_progression_grid.png).

#### Fall safely

Introduce `fall_down`:

```text
edge → freefall → absorb → stop
```

Explain the expected impact-speed relationship:

\[
v_{impact}\approx\sqrt{2gh}
\]

The landing gear extension can therefore scale approximately with \(\sqrt h\).

### 9. The skill interface

Show the clean programming contract:

```python
targets = skill(
    quaternion,
    rod_directions,
    max_extension,
    **parameters,
)
```

Explain the design choice:

- Skills are pure functions.
- They do not directly modify the simulator.
- They return 60 rod targets.
- The runner owns time, phase transitions, contacts, and retries.

This separation makes skills testable and composable.

### 10. Composing skills: climbing stairs

This should be the main case study.

Show that “stairs” is not another independent controller. It is a composition:

```text
move + stop
    ↓
jump_to + landing
    ↓ repeat for each ascending tread
move across plateau
    ↓
fall_down + stop
    ↓ repeat for each descending tread
```

Explain the current course:

- Three 25 cm rises.
- 1.30 m treads.
- 1.80 m width.
- Three upward jumps.
- Three controlled downward drops.

Important lesson: a phase timer does not prove success. A stair counts only after:

- Contact with the correct MuJoCo geometry.
- Position inside the tread boundaries.
- Stable vertical velocity.
- No core collision.

Show the final result:

- Jump attempts: 1, 1, 1.
- Climbs: 3/3.
- Descents: 3/3.
- Core impacts: 0.

Suggested video: [current stair traversal](/home/azureuser/telescopic_robot/storage_local/20260901_1955__local__skills__stairs/renders/stairs_verified_composite.mp4).

### 11. What failed—and why that matters

Include only three or four useful failures:

- A controller appeared successful because it counted time rather than contacts.
- A jump reached sufficient height but lacked horizontal velocity.
- The ball’s orientation changed which rods were available for takeoff.
- A small 8 cm preparatory side roll produced reliable first-attempt jumps.

This makes the post credible and educational without becoming a development log.

Use the repeating pattern:

```text
What I expected → what happened → measurement → correction
```

### 12. Possible real-world applications

Present these as possibilities, not proven claims:

- Inspection inside pipes, tanks, or hazardous structures.
- Search and exploration in cluttered environments.
- Rough-terrain or planetary exploration.
- Robots whose effective shape can change for locomotion or protection.
- Educational research into unconventional locomotion.

Explain that the current work is simulation-first and physical hardware remains future work.

### 13. What comes next

Briefly introduce the future high-level layer:

- A task planner selects named skills.
- Imitation learning can learn from scripted demonstrations.
- Reinforcement learning can choose directions, timings, and skill transitions.
- Eventually the policy should issue commands such as:

```text
move toward the platform
jump onto it
stabilize
turn right
descend safely
```

Do not explain the complete IL/RL implementation in this article. That deserves a second post.

### 14. Conclusion: returning to the old idea

Return to the personal story:

- The original idea did not arrive with a complete application.
- Implementing it created the technical understanding needed to discover applications.
- Coding assistants accelerated experimentation, but measurement and physics determined which ideas worked.
- RoboBall is now a platform for studying hierarchical control and composable robot behavior.

## Recommended visuals

I suggest eight main visuals:

1. Final stair video as the opening hook.
2. Early RoboBall concept or first visible rods.
3. Annotated robot anatomy.
4. Body-frame versus world-frame rod diagram.
5. Path-following video.
6. Jump phase progression.
7. Three-layer control architecture diagram.
8. Stair skill composition diagram and final video.

Too many videos will make the article heavy. Use short MP4 clips for the hero, path following, jumping, and stairs; use still images or diagrams for everything else.

## Source-material map

We can draw the article primarily from:

- [Early robot and RL journey](/home/azureuser/telescopic_robot/docs/project_journey/01_from_telescoping_ball_to_maze_rl.md)
- [Skill library and measured experiments](/home/azureuser/telescopic_robot/docs/project_journey/02_skill_library_and_the_skill_course.md)
- [Low-level control and physics investigations](/home/azureuser/telescopic_robot/docs/project_journey/01_hierarchical_imitation_and_60d_rl_locomotion.md)
- [Stairs implementation](/home/azureuser/telescopic_robot/docs/stairs_skill.md)
- [Current skill API](/home/azureuser/telescopic_robot/skills/README.md)

## Recommended length and style

- Approximately 3,500–4,500 words.
- First-person voice.
- Accessible to programmers without robotics experience.
- Short explanations before every equation.
- Small code snippets rather than complete source files.
- Measurements beside every major claim.
- Technical sidebars for readers who want more depth.

I would write it using a repeated rhythm:

> Show the behavior → explain the problem → introduce the concept → derive the controller → show the measured result.

This gives the post a clear tutorial structure while preserving your personal story.
