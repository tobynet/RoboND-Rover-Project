import numpy as np
import time

def is_back_mode(Rover):
    # Prevent stucked rover
    if Rover.vel < 0.001:
        if (time.time() - Rover.stucked_time >= Rover.slipout_time_on_stucked):
            print("      stucked, go back ")
            Rover.throttle = -Rover.throttle_set
            Rover.brake = 0
            Rover.steer = 0
            Rover.mode = 'back'
            Rover.stucked_time = time.time()
    else:
        Rover.stucked_time = time.time()


# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function
def decision_step(Rover):

    # Implement conditionals to decide what to do given perception data
    # Here you're all set up with some basic functionality but you'll need to
    # improve on this decision tree to do a good job of navigating autonomously!
    print("decision_step() begin...")

    # Example:
    # Check if we have vision data to make decisions with
    if Rover.nav_angles is not None:
        # Check for Rover.mode status
        if Rover.mode == 'forward':
            print("  @mode = forward")

            # Check the extent of navigable terrain
            if (len(Rover.nav_angles) >= Rover.stop_forward):# \
                    #or (Rover.found_rock):
                
                print("    In throttle and turn:", len(Rover.nav_angles))

                # If mode is forward, navigable terrain looks good 
                # and velocity is below max, then throttle 
                if Rover.vel < Rover.max_vel:
                    # Set throttle value to throttle setting
                    Rover.throttle = Rover.throttle_set
                else: # Else coast
                    Rover.throttle = 0
                Rover.brake = 0

                if not is_back_mode(Rover):
                    # right hand side...
                    #add_angle = +0.05 * 180/np.pi
                    # Set steering to average angle clipped to the range +/- 15
                    #Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi + add_angle), -20, 20)
                    Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -10, 10)

                print("      :", {
                    'vel': Rover.vel, 'throttle': Rover.throttle, 
                    'brake': Rover.brake, 'steer': Rover.steer})

            # If there's a lack of navigable terrain pixels then go to 'stop' mode
            elif len(Rover.nav_angles) < Rover.stop_forward:
                print("    go stop mode!!:", len(Rover.nav_angles))

                # Set mode to "stop" and hit the brakes!
                Rover.throttle = 0
                # Set brake to stored brake value
                Rover.brake = Rover.brake_set
                Rover.steer = 0
                Rover.mode = 'stop'

                print("      :", {
                    'vel': Rover.vel, 'throttle': Rover.throttle, 
                    'brake': Rover.brake, 'steer': Rover.steer})

        # If we're already in "stop" mode then make different decisions
        elif Rover.mode == 'stop':
            print("  @mode = stop")

            # If we're in stop mode but still moving keep braking
            if Rover.vel > 0.2:
                print("    more stop: vel:", Rover.vel)

                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0

            # If we're not moving (vel < 0.2) then do something else
            elif Rover.vel <= 0.2:
                print("    already stoped: vel:", Rover.vel)

                # Now we're stopped and we have vision data to see if there's a path forward
                if len(Rover.nav_angles) < Rover.go_forward:
                    
                    print("      stucked, and turn here:", len(Rover.nav_angles))

                    Rover.throttle = 0
                    # Release the brake to allow turning
                    Rover.brake = 0
                    # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                    Rover.steer = -30 # Could be more clever here about which way to turn

                # If we're stopped but see sufficient navigable terrain in front then go!
                if len(Rover.nav_angles) >= Rover.go_forward:
                    print("      go forward!!:", len(Rover.nav_angles))

                    # Set throttle back to stored value
                    Rover.throttle = Rover.throttle_set
                    # Release the brake
                    Rover.brake = 0
                    # Set steer to mean angle
                    Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
                    Rover.stucked_time = time.time()
                    Rover.mode = 'forward'

                    print("      :", {
                        'vel': Rover.vel, 'throttle': Rover.throttle, 
                        'brake': Rover.brake, 'steer': Rover.steer})

        elif Rover.mode == 'back':
            print("  @mode = back")
            if (time.time() - Rover.stucked_time < 4):
                print("      go back! ")
                Rover.throttle = 0
                Rover.brake = 0
                Rover.steer = -30
            else:
                Rover.throttle = Rover.throttle_set
                Rover.stucked_time = time.time()
                Rover.mode = 'forward'
            
            # print("  @mode = back")
            # if (time.time() - Rover.stucked_time < 2):
            #     print("      go back! ")
            #     Rover.throttle = -Rover.throttle_set
            #     Rover.brake = 0
            #     Rover.steer = 0
            # else:
            #     Rover.throttle = Rover.throttle_set
            #     Rover.stucked_time = time.time()
            #     Rover.mode = 'forward'
            print("      :", {
                'vel': Rover.vel, 'throttle': Rover.throttle, 
                'brake': Rover.brake, 'steer': Rover.steer})


    # Just to make the rover do something 
    # even if no modifications have been made to the code
    else:
        print("Rover.nav_angles is NONE")
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0


    # If in a state where want to pickup a rock send pickup command
    if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
        print("begin pickup.")
        Rover.send_pickup = True

    print("pickup flags: ", {
        'near_sample': Rover.near_sample, 
        'picking_up': Rover.picking_up, 
        'send_pickup': Rover.send_pickup})
    print("decision_step() DONE.")
    
    return Rover

