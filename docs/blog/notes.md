# Notes on RoboBall Physics & Mechanics

## Contact Torque & Rolling Locomotion

### Q: What does "The force line misses $O$, so it creates a turning effect" mean in the contact torque diagram?

**A:** 

In mechanics, this means the **line of action** of the net ground reaction force $\mathbf{F}_i$ does not pass directly through the robot's center of mass $O$. 

Here is the breakdown of why this happens and why it is critical for RoboBall's movement:

---

#### 1. Line of Action and Lever Arm
- **If a force points directly through the center $O$:**
  The perpendicular distance from the pivot point $O$ to the line of action of the force (known as the **lever arm** $d$) is zero:
  $$\boldsymbol{\tau}_i = \mathbf{r}_i \times \mathbf{F}_i = \mathbf{0}$$
  A force passing through the center only produces linear acceleration (translational push) without spinning the ball.

- **When the force line misses $O$:**
  There is a nonzero lever arm ($d > 0$). Consequently, the vector cross product produces a nonzero **torque** $\boldsymbol{\tau}_i$:
  $$\lVert\boldsymbol{\tau}_i\rVert = \lVert\mathbf{r}_i \times \mathbf{F}_i\rVert = d \cdot \lVert\mathbf{F}_i\rVert \neq 0$$
  This torque is the "turning effect" that causes rotational acceleration about the center $O$.

---

#### 2. Why the Force Line Misses $O$ in RoboBall
The total contact reaction force $\mathbf{F}_i$ exerted by the ground on the rod foot is the vector sum of two orthogonal components:

1. **Normal force ($\mathbf{F}_{n,i}$):** Acts perpendicular to the ground (upward) to support the robot's weight.
2. **Friction force ($\mathbf{F}_{t,i}$):** Acts tangential to the ground (horizontal) in response to the rod pushing backward against the floor.

Because of the friction component, the resultant reaction vector $\mathbf{F}_i = \mathbf{F}_{n,i} + \mathbf{F}_{t,i}$ is tilted relative to the rod. When you trace this force vector forward, its line of action passes to the side of the core center $O$ rather than straight through it.

---

#### 3. Why This Matters for Locomotion
This turning effect is precisely what enables RoboBall to roll:
- If rods only pushed straight through the center, the robot would slide or stall rather than roll smoothly.
- By selectively extending rods positioned downward and slightly behind the intended direction of travel, the robot commands ground reaction forces whose lines of action continuously miss $O$.
- This generates the sustained torque needed to drive forward rolling locomotion.
