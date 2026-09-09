# Control and coordination in RoboBall

RoboBall needs several kinds of decisions. A useful way to understand them is
to ask what each layer receives and what it produces.

## The control hierarchy

| Layer | Main question | Example | Output |
|---|---|---|---|
| High-level planning | What should the robot do next? | Approach the stairs, climb them, then descend | A skill and its goal |
| Mid-level skill control | How should the ball perform that action? | Move forward, turn, jump, or land | A coordinated movement command |
| Rod coordination | Which rods should move, and by how much? | Use the rear-bottom rods to push forward | 60 target extensions |
| Low-level actuator control | How does each rod reach its target? | Apply PD control to one telescoping rod | 60 actuator forces |
| Physics and hardware | What motion results from those forces? | Rolling, contact, friction, and impacts | The next robot state |

The complete flow is

```text
task goal
   ↓
high-level planner
   ↓ chooses and sequences skills
move forward / turn / jump / land
   ↓
rod-coordination algorithm
   ↓ produces one target for every rod
[e₁*, e₂*, ..., e₆₀*]
   ↓
60 low-level PD controllers
   ↓ produce actuator forces
[F₁, F₂, ..., F₆₀]
   ↓
robot mechanics and environment
```

## High-level planning

The high-level planner receives a task such as **reach the top of the
staircase**. It decides which skills to use and in what order. For example:

```text
approach the stairs
→ slow down
→ jump onto the first step
→ stabilize
→ jump again
→ descend
```

The planner does not calculate individual rod forces. It selects mid-level
skills and gives them goals, directions, or other parameters.

## Mid-level skills

A mid-level skill describes a complete movement pattern. RoboBall's primitive
skills include moving forward, turning, jumping, and landing.

Each skill coordinates many rods:

- moving forward creates a travelling pattern of rear-bottom rods;
- turning produces different patterns on the two sides of the ball;
- jumping extends several downward rods together; and
- landing prepares rods to absorb contact.

The command **move right** is high-level compared with a motor command, but it
is still a mid-level command in the complete robot architecture. A high-level
command is closer to **reach the top of the staircase**.

## Rod coordination

The rod-coordination algorithm translates a movement command into 60 target
extensions. For forward motion, it evaluates which rods point behind the
requested direction, remain near the travel centreline, and point sufficiently
downward.

This is not a strictly binary selection. Every rod receives a continuous drive
weight

$$
0\leq w_i\leq1.
$$

A weight near one means that the rod is useful for the current movement. A
weight near zero means that it should remain near its minimum extension. The
weight is converted into a target extension $e_i^*$.

The current coordination rules are expert-designed: a human chose the geometry
and conditions used to distribute the rod targets. This describes how the
controller was designed, not its level in the hierarchy. A learned policy could
replace the expert-designed rule while remaining at the same coordination
level.

## Low-level actuator control

Rod coordination decides the desired extension $e_i^*$. It does not determine
the motor force directly. Each rod therefore has a low-level PD controller:

$$
F_i=K_p\left(e_i^*-e_i\right)-K_v\dot e_i.
$$

The proportional part corrects extension error, while the derivative part
damps fast motion. The output $F_i$ is the force applied to actuator $i$.
Therefore, the low-level controller performs

```text
target extension eᵢ*
→ compare with measured extension eᵢ
→ compute actuator force Fᵢ
```

PD does not decide whether a rod is useful for moving or jumping. It only makes
the rod follow the target supplied by the layer above it.

## Physics closes the loop

The actuator forces interact with the robot and its environment. MuJoCo
computes joint motion, ground contact, friction, gravity, and impacts. Sensors
then measure the resulting state, and the controllers repeat their decisions.

```text
plan → skill → rod targets → actuator forces → physical motion
  ↑                                                   ↓
  └────────────── measured robot state ───────────────┘
```

This feedback loop is what turns a sequence of abstract decisions into physical
movement.
