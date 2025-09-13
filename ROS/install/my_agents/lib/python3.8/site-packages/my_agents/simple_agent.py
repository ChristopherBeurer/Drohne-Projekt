import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import csv

TARGET_X = 5.0
TARGET_Y = 5.0
TARGET_Z = 2.0

class SimpleAgent(Node):
    def __init__(self):
        super().__init__('simple_agent')
        self.publisher = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', 10)
        self.subscription = self.create_subscription(Odometry, '/mavros/local_position/odom', self.odom_callback, 10)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.current_pose = None
        self.episode_done = False
        self.step = 0

        # Logging initialisieren
        self.logfile = open('episode_log.csv', 'w', newline='')
        self.writer = csv.writer(self.logfile)
        self.writer.writerow(['step', 'x', 'y', 'z', 'reward'])

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose

    def timer_callback(self):
        if self.episode_done:
            return

        target = PoseStamped()
        target.header.frame_id = "map"
        target.pose.position.x = TARGET_X
        target.pose.position.y = TARGET_Y
        target.pose.position.z = TARGET_Z
        self.publisher.publish(target)

        if self.current_pose:
            dx = TARGET_X - self.current_pose.position.x
            dy = TARGET_Y - self.current_pose.position.y
            dz = TARGET_Z - self.current_pose.position.z
            dist = (dx**2 + dy**2 + dz**2)**0.5
            print(f"Aktuelle Distanz zum Ziel: {dist:.2f} m")

            # Reward-Berechnung
            reward = -dist
            if dist < 0.5:
                reward += 10
                print("Ziel erreicht! Episode abgeschlossen.")
                self.episode_done = True

            # Logging
            self.writer.writerow([
                self.step,
                self.current_pose.position.x,
                self.current_pose.position.y,
                self.current_pose.position.z,
                reward
            ])
            self.step += 1

    def destroy_node(self):
        self.logfile.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = SimpleAgent()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
