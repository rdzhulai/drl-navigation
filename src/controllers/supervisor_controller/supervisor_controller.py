import threading
from controller import Supervisor

TIME_STEP = 32

# create Supervisor instance
supervisor = Supervisor()

def trajectory_movement(start_position: list[float], translation_field: object, wheels: list[object], amplitude: int | float, speed: int | float) -> None:
    # Get initial axis coordinates
    x_coord, y_coord, z_coord = start_position[0], start_position[1], start_position[2]
    
    while True:
        # Update x coordinate
        upd_x_coord = translation_field.getSFVec3f()[0]
        
        # Forward
        while upd_x_coord <= x_coord + amplitude:
            wheels[0].setVelocity(speed)
            wheels[1].setVelocity(speed)
            supervisor.step(TIME_STEP)
            upd_x_coord = translation_field.getSFVec3f()[0]  # Update x coordinate inside the loop

        # Backward
        while upd_x_coord >= x_coord:
            wheels[0].setVelocity(-speed)
            wheels[1].setVelocity(-speed)
            supervisor.step(TIME_STEP)
            upd_x_coord = translation_field.getSFVec3f()[0]  # Update x coordinate inside the loop


# initialize dynamic objects (cube)
cube_1_node = supervisor.getFromDef('CUBE-1')
cube_1_translation_field = cube_1_node.getField('translation')
cube_1_start_position = cube_1_translation_field.getSFVec3f()

cube_2_node = supervisor.getFromDef('CUBE-2')
cube_2_translation_field = cube_2_node.getField('translation')
cube_2_start_position = cube_2_translation_field.getSFVec3f()

cube_3_node = supervisor.getFromDef('CUBE-3')
cube_3_translation_field = cube_3_node.getField('translation')
cube_3_start_position = cube_3_translation_field.getSFVec3f()

# initialize motors
speed = 0.0  # [rad/s]
wheels = []
wheelsNames = ['wheel1', 'wheel2']

for i in range(2):
    wheels.append(supervisor.getDevice(wheelsNames[i]))
    wheels[i].setPosition(float('inf'))
    wheels[i].setVelocity(speed)

# Create and launch threads
move_cube_1 = trajectory_movement(cube_1_start_position, cube_1_translation_field, wheels, 0.5, 2)
thread1 = threading.Thread(target=move_cube_1)
move_cube_2 = trajectory_movement(cube_2_start_position, cube_2_translation_field, wheels, 0.5, 2)
thread2 = threading.Thread(target=move_cube_2)
move_cube_3 = trajectory_movement(cube_3_start_position, cube_3_translation_field, wheels, 0.5, 2)
thread3 = threading.Thread(target=move_cube_3)

thread1.start()
thread2.start()
thread3.start()

while supervisor.step(TIME_STEP) != -1:
    pass