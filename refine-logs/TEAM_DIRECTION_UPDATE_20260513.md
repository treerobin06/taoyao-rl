# Team Direction Update

Date: 2026-05-13  
Status: working proposal after team discussion; final assignment still needs meeting confirmation

## Updated Track Split

After discussion, the cleaner working split is to avoid making B-line another implicit-conservatism / IQL-family track. The proposed split is:

| Track | Role | Main comparison question |
|---|---|---|
| A-line | value conservatism | Do conservative Q/value methods stay stable under low-quality replay data? |
| B-line | normal / non-conservative contrast | What happens if we remove explicit conservatism / trusted-action regularization? |
| C-line | policy / behavior regularization | How do behavior regularization and trusted-action selection affect offline and O2O performance? |

This makes the final report cleaner. Instead of three partially overlapping conservative RL families, we now have:

1. conservative value learning;
2. normal non-conservative learning;
3. policy/behavior regularization.

## What B-Line Should Mean Now

B-line should not be judged by whether it is a latest SOTA method. Its role is to provide a contrast:

> Under the same low-quality-data and online fine-tuning setup, how does a normal learner behave without explicit conservatism or trusted-action regularization?

Useful B-line candidates:

- PPO or SAC-style online baseline;
- vanilla TD3-style online fine-tuning;
- another agreed weak-regularization baseline.

Minimum output format:

- method;
- env;
- seed;
- offline steps, if any;
- online steps, if any;
- eval episodes;
- final / best normalized score;
- curve or log path;
- one-sentence interpretation.

## Impact On C-Line

C-line does not need to change its completed experiments. The new B-line role actually strengthens the final story:

- A-line shows what conservative value methods do;
- B-line shows what normal/non-conservative methods do;
- C-line shows what policy regularization and trusted-action selection do.

The current C-line result should be framed as:

> Trusted-action regularization helps offline learning under low-quality replay data, but can conflict with online adaptation if the teacher-label constraint is carried into fine-tuning too strongly.

This pairs naturally with B-line if B-line shows faster online adaptation but weaker offline initialization.
