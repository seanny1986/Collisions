import numpy as np

# Class for the environment container that our physics simulation will be run in
class Environment():
    
    # class constructor
    def __init__(self, DIM, GRAVITY, dt):
        self.DIM = DIM                                              # simulation dimensions [x,y] or [x,y,z]
        self.GRAVITY = GRAVITY                                      # gravity acceleration vector [x,y] or [x,y,z]
        self.dt = dt                                                # simulation timestep
        self.particles = []                                         # list of particle objects in the environment
        self.players = []                                           # list of player characters in the environment
        self.targets = []                                           # list of goals in the environment

    # update function for the environment. Checks for collisions and updates the state of all particles and players
    def update(self):
        if self.particles:
            for p1 in self.particles:
                p1.stateUpdate()
                self.bounce(p1)
                for p2 in self.particles:
                    if p1 != p2:
                        self.elasticCollision(p1, p2)
        
        if self.players:
            for p in self.players:
                p.stateUpdate()
                self.bounce(p)
                for p2 in self.particles:
                    self.collision(p, p2)

        if self.targets:
            for t in self.targets:
                t.stateUpdate()
                self.bounce(t)
                for p in self.particles:
                    self.elasticCollision(t, p)

    # add particle to the environment's particle list
    def addParticle(self, p):
        self.particles.append(p)
        p.addAcceleration(self.GRAVITY)

    # add player to the environment's player list
    def addPlayer(self, p):
        self.players.append(p)
        p.addAcceleration(self.GRAVITY)

    # add a target to the environment
    def addTarget(self, t):
        self.targets.append(t)

    # bounce function to handle particles and players boucing off of walls
    def bounce(self, p):
        for p in self.particles:
            i = 0
            for x in p.X[0]:
                if x > self.DIM[i]-p.radius:
                    dist = p.radius-(self.DIM[i]-x)
                    p.addPosition(-dist)
                    tmp = np.zeros(np.size(p.V))
                    tmp[i] = -2*p.V[0][i]
                    p.addVelocity(tmp)
                elif x < p.radius: 
                    dist = p.radius-x
                    p.addPosition(dist)
                    tmp = np.zeros(np.size(p.X))
                    tmp[i] = -2*p.V[0][i]
                    p.addVelocity(tmp)
                i += 1

    # bounce function to handle particles bouncing off of each other (elastic collision)
    def elasticCollision(self, p1, p2):
        dX = p1.X-p2.X
        dist = np.sqrt(np.sum(dX**2))
        if dist < p1.radius+p2.radius:
            offset = dist-(p1.radius+p2.radius)
            p1.addPosition((-dX/dist)*offset/2)
            p2.addPosition((dX/dist)*offset/2)
            total_mass = p1.mass+p2.mass
            dv1 = -2*p2.mass/total_mass*np.inner(p1.V-p2.V,p1.X-p2.X)/np.sum((p1.X-p2.X)**2)*(p1.X-p2.X)
            dv2 = -2*p1.mass/total_mass*np.inner(p2.V-p1.V,p2.X-p1.X)/np.sum((p2.X-p1.X)**2)*(p2.X-p1.X)
            p1.addVelocity(dv1)
            p2.addVelocity(dv2)

    # function to handle particles colliding and sticking to one another (plastic collision)
    def plasticCollision(self):
        pass
    
    # function to handle players colliding with particles, targets, and each other
    def collision(self, player, p):
        dX = player.X-p.X
        dist = np.sqrt(np.sum(dX**2))
        if dist < player.radius+p.radius:
            for t in self.targets:
                if p == t:
                    player.addPoints(-1000)
                    self.targets.remove(t)
                    if not self.targets:
                        player.reset()
                else:
                    player.addPoints(100)
                    player.reset()


# class for handling particles. Contains particle environment, state, and physical properties
class Particle():

    # class constructor
    def __init__(self, env, X, V, A, radius, mass, density):
        self.env = env                                              # particle environment container
        self.X = X                                                  # particle position [x,y] or [x,y,z]
        self.V = V                                                  # particle velocity [u,v] or [u,v,w]
        self.A = A                                                  # particle acceleration [a,b] or [a,b,c]
        self.radius = radius                                        # particle radius
        self.mass = mass                                            # particle mass
        self.density = density                                      # particle density
        self.colour = (0, 0, 255-int((density-10)/90*240+15))       # particle color
    
    # function to add force to the particle. Takes vector argument [Fx,Fy,Fz]
    def addForce(self, F):
        self.A += F/self.mass

    # function to add acceleration to the particle. Takes vector argument [ax,ay,az]
    def addAcceleration(self, acc):
        self.A += acc

    # function to add velocity to the particle. Takes vector argument [vx,vy,vz]
    def addVelocity(self, vel):
        self.V += vel
    
    # function to add position to the particle. Takes vector argument [x,y,z]
    def addPosition(self, pos):
        self.X += pos

    # function to handle gravitational attraction between particles
    def attract(self, particle):
        r = self.X-particle.X
        self.A += 6.67408e-11*particle.mass/r**2

    # state update function using semi-implicit Euler integration
    def stateUpdate(self): 
        self.V += self.A*self.env.dt
        self.X += self.V*self.env.dt

# class for handling players. Contains player environment, state, and properties
class Player():

    # class constructor
    def __init__(self, env, X, target, mass):
        self.env = env                                              # player environment
        self.X = X                                                  # player position [x,y] or [x,y,z]
        self.mass = mass                                            # player mass (for force calculations)
        self.V = np.asarray([0, 0])                                 # player velocity initialized to zero
        self.A = np.asarray([0, 0])                                 # player acceleration initialized to zero
        self.radius = 5                                             # give player a small circular radius
        self.colour = (255, 0, 0)                                   # make player red so easy to see
        self.collision = False                                      # set collision to false
        self.points = 0                                             # set player points

    # player accumulates negative points over time 
    def addPoints(self, points):
        self.points -= points

    # function to add a force vector to the player. Takes vector argument
    def addForce(self, F):
        self.A += F/self.mass

    # function to add acceleration vector to the player. Takes vector argument
    def addAcceleration(self, acc):
        self.A += acc

    # function to add velocity vector to the player. Takes vector argument
    def addVelocity(self, vel):
        self.V += vel
    
    # function to add position vector to the player. Takes vector argument
    def addPosition(self, pos):
        self.X += pos

    # function to add gravitational force to player
    def attract(self, particle):
        r = self.X-particle.X
        self.A += 6.67408e-11*particle.mass/r**2

    def reset(self):
        self.X = env.DIM/2
        self.points = 0

    # player state update function. Also includes points
    def stateUpdate(self): 
        self.V += self.A*self.env.dt
        self.X += self.V*self.env.dt
        self.addPoints(1)

# target class for the player to reach
class Target():

    # class constructor
    def __init__(self, env, X, V, A, radius, mass, density):
        self.env = env                                              # target environment container
        self.X = X                                                  # target position [x,y] or [x,y,z]
        self.V = V                                                  # target velocity [u,v] or [u,v,w]
        self.A = A                                                  # target acceleration [a,b] or [a,b,c]
        self.radius = radius                                        # target radius
        self.mass = mass                                            # target mass (for force calculations)
        self.colour = (0, 255, 0)                                   # make target green

    # function to add force to the target. Takes vector argument [Fx,Fy,Fz]
    def addForce(self, F):
        self.A += F/self.mass

    # function to add acceleration to the target. Takes vector argument [ax,ay,az]
    def addAcceleration(self, acc):
        self.A += acc

    # function to add velocity to the target. Takes vector argument [vx,vy,vz]
    def addVelocity(self, vel):
        self.V += vel
    
    # function to add position to the target. Takes vector argument [x,y,z]
    def addPosition(self, pos):
        self.X += pos

    # function to handle gravitational attraction between target and particles, player
    def attract(self, particle):
        r = self.X-particle.X
        self.A += 6.67408e-11*particle.mass/r**2

    # state update function using semi-implicit Euler integration
    def stateUpdate(self): 
        self.V += self.A*self.env.dt
        self.X += self.V*self.env.dt