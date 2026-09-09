"""High-level skills: emit skill commands, never rod targets.

Empty for now. Reserved for planning and RL policies that choose which
skill to run and with what arguments, rather than computing rod
extensions themselves.

The contract, so this folder does not become a place for leftovers:

* Input  : task state, not rod geometry.
* Output : a skill name plus its arguments, for `execute_skill`.
* Imports: may use mid_level and low_level. Neither may import this.
"""
