# cam2 rejudge 2026-07-23

## Verdict
Partially — task-region scale/centering improved, but camera pitch still far from agentview; domain gap unchanged

## Recommendation
Ready for WAM retest (dryrun / reliable_pick) to measure policy effect; optional further cam2 raise+tilt down if agentview match is priority. Restart franka_robot_state_broadcaster for live proprio dump.

## Key numbers (scene 224)
- live brightness=111.4, mass_cy=131.8, lower-upper=39.6
- train brightness=71.1, mass_cy=81.9, lower-upper=-34.0
- prev_morning mass_cy=157.2
- prev_pose_fix2 mass_cy=125.0
- FK ee_xyz=[0.6557, 0.0904, 0.2805] aligned_norm=[0.9869999885559082, 0.16699999570846558, -0.5989999771118164, -0.1770000010728836, -0.19099999964237213, 0.10899999737739563, 0.8939999938011169, -0.9020000100135803] OOD=[]

## Artifacts
/tmp/kairos_cam2_rejudge_20260723_163355