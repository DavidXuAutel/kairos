#!/usr/bin/env bash
# Reliable pick-up-a-pen: recover → WAM micro-approach → descend → grasp → lift.
# Does NOT resume GELLO unless --resume-gello.
set -eo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate kairos
set +u
source /opt/ros/humble/setup.bash
source ~/franka_ros2_ws/install/setup.bash 2>/dev/null || true
source ~/kairos/scripts/env_libero_franka.sh
set -u
export KAIROS_ROOT="${KAIROS_ROOT:-$HOME/kairos}"
export WAM_URL="${WAM_URL:-http://127.0.0.1:8005}"
export PYTHONPATH="${KAIROS_ROOT}/benchmarks/common:${KAIROS_ROOT}/scripts/phase2:${PYTHONPATH:-}"
# ensure hold helper exists
if [[ ! -f /tmp/kairos_hold_q.py ]]; then
  cat > /tmp/kairos_hold_q.py <<'PY'
import numpy as np, rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
JOINTS=["fr3_joint1","fr3_joint2","fr3_joint3","fr3_joint4","fr3_joint5","fr3_joint6","fr3_joint7"]
class Hold(Node):
    def __init__(self):
        super().__init__("kairos_hold_q"); self.q=None
        self.create_subscription(JointState,"/franka/joint_states",self.on_j,10)
        self.pub=self.create_publisher(JointState,"/gello/joint_states",10)
        self.create_timer(0.02,self.tick)
    def on_j(self,m):
        mp={n:p for n,p in zip(m.name,m.position)}
        try: self.q=np.array([mp[n] for n in JOINTS],float)
        except KeyError: pass
    def tick(self):
        if self.q is None: return
        msg=JointState(); msg.header.stamp=self.get_clock().now().to_msg()
        msg.name=list(JOINTS); msg.position=[float(x) for x in self.q]; self.pub.publish(msg)
rclpy.init(); rclpy.spin(Hold())
PY
fi
exec python "${KAIROS_ROOT}/scripts/phase2/run_reliable_pick.py" "$@"
