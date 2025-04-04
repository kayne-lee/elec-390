import networkx as nx
import math
from time import sleep
from picarx import Picarx  # Ensure you have this library installed

def euclidean_distance(node1, node2):
    """Calculate the Euclidean distance between two points."""
    return round(math.sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2), 2)

def calculate_angle(p1, p2):
    """Calculate angle (in degrees) from p1 to p2."""
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))

def drive_picarx(path):
    """Drive Picar-X along the calculated path."""
    px = Picarx()
    current_angle = 5  # Initial heading

    for i in range(len(path) - 1):
        start = path[i]
        end = path[i + 1]

        target_angle = calculate_angle(start, end)
        distance_to_travel = euclidean_distance(start, end)

        if i == 0:
            px.set_dir_servo_angle(5)
        else:
            angle_diff = target_angle - current_angle

            # Normalize the angle difference to range [-180, 180]
            angle_diff = ((angle_diff + 180) % 360 - 180  ) *  -1
            print("Angle Diff:", angle_diff)
            # Turn the car
            if angle_diff > 0:  # Turn right
                px.set_dir_servo_angle(30)
                px.forward(50)
                sleep(abs(angle_diff) / 30)  # Sleep proportional to the angle difference
                px.set_dir_servo_angle(0)
                sleep(0.2)
            elif angle_diff < 0:  # Turn left
                px.set_dir_servo_angle(-30)
                px.forward(50)
                sleep(abs(angle_diff) / 30)  # Sleep proportional to the angle difference
                px.set_dir_servo_angle(0)
                sleep(0.2)
            else:  # Go straight if no angle change
                px.set_dir_servo_angle(5)
                sleep(0.2)

        # Move forward
        px.forward(500)
        sleep(distance_to_travel / 50)

        # Stop after each segment
        px.forward(0)
        sleep(0.2)

        # Update current angle
        current_angle = target_angle

    px.set_cam_tilt_angle(0)
    px.set_cam_pan_angle(0)  
    px.set_dir_servo_angle(5)  
    px.stop()
    sleep(.2)

def find_shortest_path(graph, start, end):
    """Find the shortest path using Dijkstra's algorithm."""
    return nx.shortest_path(graph, source=start, target=end, weight="weight", method='dijkstra')

if __name__ == "__main__":
    # Create Graph
    unknown = 266
    # Define intersections from the table as (X, Y) coordinates
    intersections = {
        (452, 29): "Aquatic Ave. & Beak St.",
        (305, 29): "Aquatic Ave. & Feather St.",
        (129, 29): "Aquatic Ave. & Waddle Way",
        (213, 29): "Aquatic Ave. & Waterfoul Way",
        (284, 393): "Breadcrumb Ave. & The Circle",
        (181, 459): "Breadcrumb Ave. & Waddle Way",
        (305, unknown): "The Circle & Feather St.",
        (273, 307): "The Circle & Waterfoul Way",
        (452, 293): "Dabbler Dr. & Beak St.",
        (350, 324): "Dabbler Dr. & The Circle",
        (585, 293): "Dabbler Dr. & Mallard St.",
        (452, 402): "Drake Dr. & Beak St.",
        (576, 354): "Drake Dr. & Mallard St.",
        (452, 474): "Duckling Dr. & Beak St.",
        (593, 354): "Duckling Dr. & Mallard St.",
        (452, 135): "Migration Ave. & Beak St.",
        (305, 135): "Migration Ave. & Feather St.",
        (585, 135): "Migration Ave. & Mallard St.",
        (29, 135): "Migration Ave. & Quack St.",
        (129, 135): "Migration Ave. & Waddle Way",
        (213, 135): "Migration Ave. & Waterfoul Way",
        (452, 233): "Pondside Ave. & Beak St.",
        (305, 233): "Pondside Ave. & Feather St.",
        (585, 233): "Pondside Ave. & Mallard St.",
        (28, 329): "Pondside Ave. & Quack St.",
        (214, 241): "Pondside Ave. & Waterfoul Way",
        (157, 266): "Pondside Ave. & Waddle Way",
        (452, 465): "Tail Ave. & Beak St.",
        (335, 387): "Tail Ave. & The Circle"
    }

    levels = {
        ((181, 459), (157, 266)): 2,
        ((28, 329), (157, 266)): 2,
        ((157, 266), (28, 329)): 2,
        ((305, 233), (452, 233)): 2,
        ((452, 233), (305, 233)): 2,
        ((29, 135), (129, 135)): 3,
        ((129, 135), (29, 135)): 3,
        ((129, 135), (213, 135)): 3,
        ((213, 135), (129, 135)): 3,
        ((213, 135), (305, 135)): 3,
        ((305, 135), (213, 135)): 3,
        
    }

    # Create graph
    G = nx.DiGraph()
    for (x, y), name in intersections.items():
        G.add_node((x, y), label=name)

    # Add edges based on the horizontal and vertical roads
    edges = [
        ((181, 459), (28, 329)), ((28, 329), (181, 459)),
        ((181, 459), (284, 393)), ((284, 393), (181, 459)),
        ((28, 329), (157, 266)), ((157, 266), (28, 329)),
        ((29, 135), (129, 135)), ((129, 135), (29, 135)),
        ((28, 329), (29, 135)), ((29, 135), (28, 329)),
        ((29, 135), (129, 29)), ((129, 29), (29, 135)),
        ((157, 266), (214, 241)), ((214, 241), (157, 266)),
        ((129, 135), (213, 135)), ((213, 135), (129, 135)),
        ((129, 29), (213, 29)), ((213, 29), (129, 29)),
        ((213, 29), (305, 29)), ((305, 29), (213, 29)),
        ((305, 29), (452, 29)), ((452, 29), (305, 29)),
        ((585, 135), (585, 233)), ((585, 233), (585, 135)),
        ((585, 233), (585, 293)), ((585, 293), (585, 233)),
        ((452, 474), (452, 465)), ((452, 465), (452, 474)),
        ((452, 465), (452, 402)), ((452, 402), (452, 465)),
        ((452, 402), (452, 293)), ((452, 293), (452, 402)),
        ((452, 293), (452, 233)), ((452, 233), (452, 293)),
        ((452, 233), (452, 135)), ((452, 135), (452, 233)),
        ((452, 135), (452, 29)), ((452, 29), (452, 135)),
        ((213, 135), (305, 135)), ((305, 135), (213, 135)),
        ((305, 135), (452, 135)), ((452, 135), (305, 135)),
        ((452, 135), (585, 135)), ((585, 135), (452, 135)),
        ((585, 135), (452, 29)), ((452, 29), (585, 135)),
        ((214, 241), (305, 233)), ((305, 233), (214, 241)),
        ((305, 233), (452, 233)), ((452, 233), (305, 233)),
        ((452, 233), (585, 233)), ((585, 233), (452, 233)),
        ((395, 29), (305, 135)), ((305, 135), (395, 29)),
        ((305, 135), (305, 233)), ((305, 233), (305, 135)),
        ((305, 233), (305, 266)), ((305, 266), (305, 233)),
        ((335, 387), (452, 474)), ((452, 474), (335, 387)),
        ((335, 387), (452, 465)), ((452, 465), (335, 387)),
        ((157, 266), (129, 135)),
        ((129, 135), (129, 29)),
        ((181, 459), (157, 266)),
        ((213, 135), (214, 241)),
        ((213, 29), (213, 135)),
        ((214, 241), (273, 307)),
        ((273, 307), (305, unknown)),
        ((305, unknown), (350, 324)),
        ((350, 324), (335, 387)),
        ((335, 387), (284, 393)),
        ((284, 393), (273, 307)),
        ((452, 293), (350, 324)),
        ((585, 293), (452, 293)),
        ((585, 293), (593, 354)),
        ((576, 354), (585, 293)),
        ((452, 402), (576, 354)),
        ((593, 354), (452, 474)),
    ]

    for u, v in edges:
        distance = euclidean_distance(u, v)
        level = levels.get((u, v), 1)  # Default level is 1 if not specified
        weight = distance * level  # Weight is distance multiplied by pedestrian level
        G.add_edge(u, v, weight=round(weight, 2))

    G.add_edge(u, v, weight=weight)
    # Define Start & End
    start = (181, 459)
    end = (452, 474)
    # Get Shortest Path
    shortest_path = find_shortest_path(G, start, end)
    print("Shortest Path:", shortest_path)

    # Drive Picar-X along the path
    drive_picarx(shortest_path)