class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
def print_obv(env, obv, size):
    for i in range(size):
        for j in range(size):
            if i == obv[env.player1][0][0] and j == obv[env.player1][0][1]:
                print(f"{bcolors.FAIL}{obv[env.player1][2][0]}{bcolors.ENDC}", end = "")
            elif i == obv[env.player2][0][0] and j == obv[env.player2][0][1]:
                print(f"{bcolors.OKBLUE}{obv[env.player2][2][0]}{bcolors.ENDC}", end = "")
            else:
                print("-", end = "")
        print("")

def simulate(env, trainer1, trainer2, size):
    obv = env.reset()
    done = {"__all__" : False}
    
    while done["__all__"] == False:
        a1 = trainer1.compute_action(obv[env.player1])
        a2 = trainer2.compute_action(obv[env.player2])
        
        
        print_obv(env, obv, size)
        print("".join(['*']*(size)) , end = " action ")
        print(f"{bcolors.FAIL}{a1}{bcolors.ENDC}", end = " , ")
        print(f"{bcolors.OKBLUE}{a2}{bcolors.ENDC}")
        
        obv, reward, done, info = env.step({env.player1: a1, env.player2: a2})
    
    print_obv(env, obv, size)