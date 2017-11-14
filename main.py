import physics as pf
import pygame
import pygame.gfxdraw
import pygame.image as image
import numpy as np
import dae
import math
import torch
from torch.autograd import Variable

# pygame display function for the simulation
def display(env):
    if env.particles:
        for p in env.particles:
            pygame.gfxdraw.filled_circle(screen, int(p.X[0][0]), int(p.X[0][1]), p.radius, p.colour)
    if env.players:
        for p in env.players:
            pygame.gfxdraw.filled_circle(screen, int(p.X[0][0]), int(p.X[0][1]), p.radius, p.colour)
    if env.targets:
        for t in env.targets:
            pygame.gfxdraw.filled_circle(screen, int(t.X[0][0]), int(t.X[0][1]), t.radius, t.colour)

# initialize physics simulation container
DIM = np.asarray([400, 400])
GRAVITY = np.asarray([0, 0])
dt = 0.01
env = pf.Environment(DIM, GRAVITY, dt)
number_of_particles = np.random.randint(15,20)

# generate random radii, masses and densities for particles, add them to the environment container
for n in range(number_of_particles):
    radius = np.random.randint(10, 20)
    density = np.random.randint(10, 100)
    mass = (4/3)*density*np.pi*radius**3
    X = np.random.rand(1, 2)*(DIM-radius)+radius
    V = np.random.rand(1, 2)*75
    A = np.zeros((1,2))
    particle = pf.Particle(env, X, V, A, radius, mass, density)
    env.addParticle(particle)

# initialize pygame to view the simulation
pygame.init()
screen = pygame.display.set_mode((DIM[0], DIM[1]))
pygame.display.set_caption('Elastic Collision Particle Simulation')

# generate raw collision data from the simulation
seconds = 5
simtime = seconds/dt
t = 0
running = True
DATA = np.ndarray((9,int(simtime),number_of_particles))
while running:
    
    i = 0
    for p in env.particles:
        DATA[0][t][i] = p.X[0][0]
        DATA[1][t][i] = p.X[0][1]
        DATA[2][t][i] = p.V[0][0]
        DATA[3][t][i] = p.V[0][1]
        DATA[4][t][i] = p.A[0][0]
        DATA[5][t][i] = p.A[0][1]
        DATA[6][t][i] = p.radius
        DATA[7][t][i] = p.mass
        DATA[8][t][i] = p.density
        i += 1

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    screen.fill([255, 255, 255])
    env.update()
    display(env)
    pygame.display.flip()
    t += 1
    if t == int(simtime):
        running = False

def cutData(DATA,thresh):
    print('Prepping data...')

    # chop data like uncle chop chop
    l,m,n = DATA.shape
    clist = []          # context object
    xlist = []          # input data
    ylist = []          # output data
    X = []              # aggregated input data
    Y = []              # aggregated output data

    for i in range(0,m-1):
        for j in range(0,n):
            for k in range(0,n):
                if j != k:
                    rel_x = DATA[0,i,j]-DATA[0,i,k]
                    rel_y = DATA[1,i,j]-DATA[1,i,k]
                    rel_dist = math.sqrt(rel_x**2+rel_y**2)
                    total_rad = DATA[6,i,j]+DATA[6,i,k]
                    if rel_dist-total_rad <= thresh:
                        clist.append(DATA[:,i,k])
            ylist.append(DATA[:,i+1,j])
            xlist.append([DATA[:,i,j], clist])
            clist = []
        X.append(xlist)
        Y.append(ylist)
        xlist = []
        ylist = []
    return X, Y

X, Y = cutData(DATA,7)
print('Data prepped, training network...')

# initialize neural net
encoder_topology = [18, 2500, 1]
decoder_topoogy = [10, 2500, 9]
encoder = dae.FullyConnectedNetwork(encoder_topology)
decoder = dae.FullyConnectedNetwork(decoder_topoogy)
model = dae.DynamicEncoder(encoder, decoder)

criterion = torch.nn.MSELoss(size_average=False)              
optimizer = torch.optim.SGD(model.parameters(), lr=1e-16, momentum=0.9)
iterations = 10
for i in range(0,iterations):
    total_loss = 0
    time = 0
    for t in X:
        c = 0
        for obj in t:
            focus = obj[0]
            context = obj[1]
            f = Variable(torch.FloatTensor(focus))
            x = Variable(torch.FloatTensor(context))
            y = Y[time][c]
            y = Variable(torch.FloatTensor(y))
        
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(f,x)

            # Compute and print loss
            loss = criterion(y_pred, y)
            total_loss = total_loss+loss.data[0]
        
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            c += 1
        time += 1
    print('Iteration: ' + str(i+1) + '/' + str(iterations) + ', Loss: ' + str(total_loss))

# initialize simulation with neural engine running the update function
running = True
thresh = 7
t = 0
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    screen.fill([255, 255, 255])
    for p in env.particles:
        focus = [p.X[0][0], p.X[0][1], p.V[0][0], p.V[0][1], p.A[0][0], p.A[0][1], p.radius, p.mass, p.density]
        context = []
        for q in env.particles:
            if p != q:
                rel_x = p.X[0][0]-q.X[0][0]
                rel_y = p.X[0][1]-q.X[0][1]
                rel_dist = math.sqrt(rel_x**2+rel_y**2)
                total_rad = p.radius+q.radius
                if rel_dist-total_rad <= thresh:
                    c = [q.X[0][0], q.X[0][1], q.V[0][0], q.V[0][1], q.A[0][0], q.A[0][1], q.radius, q.mass, q.density]
                    context.append(c) 
        focus = Variable(torch.FloatTensor(focus))
        context = Variable(torch.FloatTensor(context))
        out = model(focus, context).data.numpy()[2:4]
        p.addVelocity(out/1000)
    env.update()
    display(env)
    pygame.display.flip()
    #image.save(screen,'frame'+str(t)+'.jpeg')
    t += 1
    if t == int(simtime):
        running = False
