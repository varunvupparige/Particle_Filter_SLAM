
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import time
import matplotlib.animation as animation
from tqdm import tqdm


def read_data_from_csv(filename):
  '''
  INPUT 
  filename        file address

  OUTPUT 
  timestamp       timestamp of each observation
  data            a numpy array containing a sensor measurement in each row
  '''
  data_csv = pd.read_csv(filename, header=None)
  data = data_csv.values[:, 1:]
  timestamp = data_csv.values[:, 0]
  return timestamp, data


def lv_encoder():
    
    enc_ts,enc_ticks = read_data_from_csv('data/sensor_data/encoder.csv')
    
    enc_t = np.diff(enc_ts/1e9)
    
    N = 10
    
    Lticks_diff = np.diff(enc_ticks[:,0],axis=0)
    Rticks_diff = np.diff(enc_ticks[:,1],axis=0)
    
    Lv = (Lticks_diff * 0.623479 * np.pi)/(4096 * enc_t)
    Rv = (Rticks_diff * 0.622806 * np.pi)/(4096 * enc_t)
    
    avg_vel = (Lv+Rv)/2
    
    lin_vel = np.zeros((len(enc_t),N))
    
    for j in range(N):
        for i in range(len(avg_vel)):
            lin_vel[i,j] = avg_vel[i] + np.random.normal(0,1e-2)
    
    return enc_ts,enc_t,avg_vel,lin_vel,N


def w_fog():
    
    _ ,fog_yaw = read_data_from_csv('data/sensor_data/fog.csv')
    fog_yaw = fog_yaw[:,2]
    
    _,enc_t,temp1,temp2,N = lv_encoder()

    
    w_fog = np.zeros((len(enc_t),N))
    yaw_fog = np.zeros((len(enc_t),N))
    
    for j in range(N):
        for i in range(len(enc_t)):
            w_fog[i,j] = np.sum(fog_yaw[10*i:10*i+10])/enc_t[i] + np.random.normal(0,1e-3)
            yaw_fog[i,j] = np.sum(fog_yaw[:10*i+10]) + np.random.normal(0,1e-3)
        
    return w_fog,yaw_fog 


def motion_model(lin_vel,w_fog,yaw_fog,enc_t,i):
    
    #this function returns delta state space of each particle for a particular timestamp
    linvel_time_a = lin_vel[i,:]
    wfog_time_a = w_fog[i,:]
    yawfog_time_a = yaw_fog[i,:]
    
    x_t = np.zeros((len(lin_vel[i,:]),3))
    
    for k in range(len(linvel_time_a)):
        x_t[k,0] = enc_t[i]*linvel_time_a[k]*np.cos(yawfog_time_a[k])
        x_t[k,1] = enc_t[i]*linvel_time_a[k]*np.sin(yawfog_time_a[k])
        x_t[k,2] = enc_t[i]*yawfog_time_a[k]
       
    #x = np.insert(x,0,np.array([0,0,0]).T,axis=0)
    
    return x_t


def sigmoid(z):
    #if -z > np.log(np.finfo(type(z)).max):
        #return 0.0    
    a = np.exp(-z)
    return 1 / (1 + a)


def bresenham2D(sx, sy, ex, ey):
  '''
  Bresenham's ray tracing algorithm in 2D.
  Inputs:
	  (sx, sy)	start point of ray
	  (ex, ey)	end point of ray
  '''
  sx = int(np.round(sx))
  sy = int(np.round(sy))
  ex = int(np.round(ex))-1
  ey = int(np.round(ey))-1
  dx = abs(ex-sx)
  dy = abs(ey-sy)
  steep = abs(dy)>abs(dx)
  if steep:
    dx,dy = dy,dx # swap 

  if dy == 0:
    q = np.zeros((dx+1,1))
  else:
    q = np.append(0,np.greater_equal(np.diff(np.mod(np.arange( np.floor(dx/2), -dy*dx+np.floor(dx/2)-1,-dy),dx)),0))
  if steep:
    if sy <= ey:
      y = np.arange(sy,ey+1)
    else:
      y = np.arange(sy,ey-1,-1)
    if sx <= ex:
      x = sx + np.cumsum(q)
    else:
      x = sx - np.cumsum(q)
  else:
    if sx <= ex:
      x = np.arange(sx,ex+1)
    else:
      x = np.arange(sx,ex-1,-1)
    if sy <= ey:
      y = sy + np.cumsum(q)
    else:
      y = sy - np.cumsum(q)
  return np.vstack((x,y))


def lidar_in_world(lidar_data,yaw_fog,j,i):
    
    #Converts lidar scan at time i to the world frame
    angles = np.linspace(-5, 185, 286) / 180 * np.pi
    yaw = yaw_fog[i,j]
    ranges = lidar_data[i,:]

    # filter out too close and too far lidar points
    indValid = np.logical_and((ranges < 60),(ranges> 0.1))
    ranges = ranges[indValid]
    angles = angles[indValid]

    # convert from polar to cartesian coordinates in lidar frame
    xs0 = ranges*np.cos(angles)
    ys0 = ranges*np.sin(angles)

    # construct homogenous state space coordinates in lidar frame
    lidar_sensorframe = np.zeros((4,(len(xs0))))
    lidar_sensorframe[0,:] = xs0
    lidar_sensorframe[1,:] = ys0
    lidar_sensorframe[2,:] = 0
    lidar_sensorframe[3,:] = 1

    # transform from lidar frame to vehicle frame
    T_v2l = np.array([[0.00130201,0.796097,0.60517,0.8349],[0.99999,-0.000419027,-0.00160026,-0.01266],[-0.00102038,0.605169,-0.796097,1.76416],[0,0,0,1]])
    lidar_vehicleframe = np.matmul(T_v2l,lidar_sensorframe)
    
    xL_vehicleframe = lidar_vehicleframe[0,:]
    yL_vehicleframe = lidar_vehicleframe[1,:]
    zL_vehicleframe = lidar_vehicleframe[2,:]

    # filter z values from lidar data in vehicle frame
    thresh = 1.5
    xL_vehicleframe = xL_vehicleframe[zL_vehicleframe<=thresh]
    yL_vehicleframe = yL_vehicleframe[zL_vehicleframe<=thresh]

    # transform from vehicle frame to world frame
    T_w2v = np.array([[np.cos(yaw),-np.sin(yaw),0,x[j,0]],[np.sin(yaw),np.cos(yaw),0,x[j,1]],[0,0,1,0],[0,0,0,1]])
    lidar_worldframe = np.matmul(T_w2v,lidar_vehicleframe)
    
    return lidar_worldframe


def log_odds(tempMAP,lidar_worldframe,particle_index):
    
    
    #converts world frame cartesian coordinates to cells
    xL_worldframe = lidar_worldframe[0,:]
    yL_worldframe = lidar_worldframe[1,:]
    zL_worldframe = lidar_worldframe[2,:]
    
    # convert from meters to cells for lidar data
    xis = np.ceil((xL_worldframe - tempMAP['xmin']) / tempMAP['res'] ).astype(np.int16)-1
    yis = np.ceil((yL_worldframe - tempMAP['ymin']) / tempMAP['res'] ).astype(np.int16)-1

    # convert from meters to cells for motion model
    xmm = np.ceil((x[particle_index,0] - tempMAP['xmin']) / tempMAP['res'] ).astype(np.int16)-1
    ymm = np.ceil((x[particle_index,1] - tempMAP['ymin']) / tempMAP['res'] ).astype(np.int16)-1
    #print(xmm.shape)

    # update log odds
    indGood = np.logical_and(np.logical_and(np.logical_and((xis > 0), (yis > 0)), (xis < MAP['sizex'])), (yis < MAP['sizey']))

    for k in range((len(xis))):
        non_occupied = bresenham2D(xmm,ymm,xis[k],yis[k]).astype(np.int16)
        #print(non_occupied)
        tempMAP['map'][non_occupied[0,:-1],non_occupied[1,:-1]] -= np.log(9)
        tempMAP['map'][non_occupied[0,-1],non_occupied[1,-1]] += np.log(9)

    return tempMAP['map']


def mapCorrelation(im, x_im, y_im, vp, xs, ys):
    nx = im.shape[0]
    ny = im.shape[1]
    xmin = x_im[0]
    xmax = x_im[-1]
    xresolution = (xmax-xmin)/(nx-1)
    ymin = y_im[0]
    ymax = y_im[-1]
    yresolution = (ymax-ymin)/(ny-1)
    nxs = xs.size
    nys = ys.size
    cpr = np.zeros((nxs, nys))
    for jy in range(0,nys):
        y1 = vp[1,:] + ys[jy] # 1 x 1076
        iy = np.int16(np.round((y1-ymin)/yresolution))
        for jx in range(0,nxs):
            x1 = vp[0,:] + xs[jx] # 1 x 1076
            ix = np.int16(np.round((x1-xmin)/xresolution))
            valid = np.logical_and( np.logical_and((iy >=0), (iy < ny)), \
                                    np.logical_and((ix >=0), (ix < nx)))
            cpr[jx,jy] = np.sum(im[ix[valid],iy[valid]])
    return cpr


# Particle Filter Slam

_, lidar_data = read_data_from_csv('data/sensor_data/lidar.csv')
enc_ts,enc_t,_,lin_vel,N = lv_encoder()
w_fog,yaw_fog = w_fog()


# init MAP
MAP = {}
MAP['res']   = 1 #meters
MAP['xmin']  = -1500  #meters
MAP['ymin']  = -1500
MAP['xmax']  =  1500
MAP['ymax']  =  1500 
MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.float64)
MAP['binmap'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8)

# init BIN MAP for generating video
BINMAP = []
frames = []
fig = plt.figure()

#Particles parameters
N = 10
N_threshold = 5

#Trajectory tracker
historyX=[]
historyY=[]

#Parameters for mapCorrelation
x_im = np.arange(MAP['xmin'],MAP['xmax']+MAP['res'],MAP['res']) #x-positions of each pixel of the map
y_im = np.arange(MAP['ymin'],MAP['ymax']+MAP['res'],MAP['res']) #y-positions of each pixel of the map
x_range = np.arange(-0.4,0.4+0.1,0.1)
y_range = np.arange(-0.4,0.4+0.1,0.1)

c = 11100 #number of iterations

# init state space matrix of motion model
x = np.zeros((N,3))
weights = 1/(N)*np.ones(N)

# init first lidar scan and update log odds
lidar_worldframe = lidar_in_world(lidar_data,yaw_fog,0,0)
MAP['map'] = log_odds(MAP,lidar_worldframe,0)


for i in tqdm(range(1,50000)):
    scores = np.zeros(N)
    scores_max = np.zeros(N)

    x_t_i = motion_model(lin_vel,w_fog,yaw_fog,enc_t,i)
    x += x_t_i
    

    for j in range(N):


        lidar_worldframe = lidar_in_world(lidar_data,yaw_fog,j,i)
        lidar_stack = np.vstack((lidar_worldframe[0,:],lidar_worldframe[1,:]))
        cpr = mapCorrelation(MAP['binmap'],x_im,y_im,lidar_stack,x_range,y_range)
        ind = np.argmax(cpr)
        scores[j]=np.max(cpr)

    scores_max=(np.multiply(weights, np.exp((np.array(scores)-max(scores))))).astype(float)
    W = (np.divide(scores_max, sum(scores_max))).astype(float)

    highcorr_particle_index = np.argmax(W)
    highcorr_worldcord = lidar_in_world(lidar_data,yaw_fog,highcorr_particle_index,i)

    #Trajectory tracking
    historyX_t = np.ceil((x[highcorr_particle_index,0] - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    historyY_t = np.ceil((x[highcorr_particle_index,1] - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
    historyX.append(historyX_t)
    historyY.append(historyY_t)

    MAP['map'] = log_odds(MAP,highcorr_worldcord,highcorr_particle_index)
    MAP['binmap'][MAP['map']>0] = 1
    MAP['binmap'][MAP['map']<0] = 2
    MAP['binmap'][MAP['map']==0] = 0
    #BINMAP = MAP['binmap'].tolist()

    N_eff= 1 / np.sum(np.square(W))
    if N_eff < N_threshold:
        x = x[np.random.choice(np.arange(N),N,True,W)]
        W = (np.zeros(N))+(1.0/N)

        #frames.append([plt.imshow(BINMAP, cmap=cm.Greys_r,animated=True)])


# Generate plots and visualization

fig = plt.figure()
plt.title("Robot trajectory in world frame")
plt.axis('equal')


fig1 = plt.figure()
#print(np.unique(sigmoid(b)))
#plt.imshow(sigmoid(it20000),cmap="hot")

plt.title("Occupancy grid map")
plt.xlabel("x-displacement")
plt.ylabel("y-displacement")
plt.xlim([0,2000])
plt.ylim([1250,3000])
plt.show()

#print(np.unique(sigmoid(b)))
fig2 = plt.figure()
plt.imshow(MAP['binmap'],cmap="hot");
plt.scatter(historyY[:50000],historyX[:50000],s=0.1,c='b')
plt.title("Occupancy grid binary map after 50000 iterations")
plt.xlabel("x-displacement")
plt.ylabel("y-displacement")
plt.xlim([0,2000])
plt.ylim([1250,3000])

plt.show()