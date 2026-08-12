# واژه‌نامه — Hard words from the telescopic-robot work

Technical / uncommon English words used while building the RadialSphere robot,
with Persian translations and what they meant in this project.

| English | فارسی | In this project |
|---|---|---|
| telescopic | تلسکوپی، کشویی | bars that slide in/out like a telescope |
| sleeve | غلاف | the fixed guide tube each rod slides through |
| rod | میله | the sliding inner bar |
| foot | کف‌پا، پایه | the rounded cap on the rod tip that touches the ground |
| stub | زائدهٔ کوتاه | the short piece of sleeve that used to stick out of the ball |
| flush (with) | هم‌سطح (با) | the sleeve ports now sit level with the ball surface |
| port | دهانه، درگاه | the flush ring where a bar enters the ball |
| protrude / protrusion | بیرون زدن / بیرون‌زدگی | how far a part sticks out of the sphere |
| retract / retraction | جمع شدن، تو رفتن | a bar pulling back into the ball |
| extend / extension | باز شدن / بازشدگی | a bar pushing out; also the joint's length value |
| stroke / travel | کورسِ حرکت | the full range a bar can move (0 → max_extend) |
| excursion | دامنهٔ جابه‌جایی | how far a joint actually moved during an episode |
| prismatic joint | مفصل کشویی | MuJoCo "slide" joint — moves along a line, no rotation |
| actuator | عملگر، محرک | the motor driving each slide joint |
| stiffness | سفتی | actuator position gain (kp) |
| damping | میرایی | actuator velocity gain (kv) |
| compliance | نرمی، انعطاف‌پذیری | how much the actuators give way under load |
| penetration | فرورفتن، نفوذ | feet sinking into the floor (the contact bug) |
| traction | چسبندگی، کشش | grip between the feet and the floor |
| settle | نشستن، جاگیر شدن | letting gravity seat the ball during reset |
| spawn | نقطهٔ شروع، پیدایش | where the ball starts an episode |
| gait | طرز راه رفتن، گام | the rolling/pushing movement pattern |
| stance | ایستایی، تکیه‌گاه | the downward bars that carry the ball's weight |
| breadcrumbs | خرده‌نان؛ نشانه‌های مسیر | the small red path markers |
| waypoint | نقطهٔ میانی مسیر | one point of the discretised path |
| look-ahead | پیش‌نگری | aiming at a point further along the path |
| turnaround | دوربرگردان | the semicircle where the roundtrip course reverses |
| lane | باند، خط حرکت | the out and return tracks of the roundtrip |
| roundtrip | رفت‌وبرگشت | the new out-and-back scenario |
| truncation | قطع زودهنگام | episode ending by hitting max_steps, not the goal |
| hue | فام، رنگ‌مایه | the unique color given to each bar |
| silhouette | سایه‌نما، نیم‌رخ | the ball's outline — why identical bars looked frozen |
| chase camera | دوربین تعقیب‌کننده | the camera following behind the ball |
| tilt | شیب دادن، کج کردن | angling the camera down toward the ground |
| tangent | مماس | the path's initial direction, used to aim the camera |
| vestigial | زائد، به‌جامانده و بی‌استفاده | dead config (the old `demo:` section) |
| streaming | جریانی، نوشتن تدریجی | writing video frames to disk one by one (vs buffering in RAM) |
