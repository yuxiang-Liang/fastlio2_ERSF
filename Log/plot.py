# import matplotlib
# matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def euler_to_rotation_matrix(roll, pitch, yaw):
    # Calculate sin and cosine values of the angles
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    # Define rotation matrix
    R_roll = np.array([[1, 0, 0],
                       [0, cr, -sr],
                       [0, sr, cr]])

    R_pitch = np.array([[cp, 0, sp],
                        [0, 1, 0],
                        [-sp, 0, cp]])

    R_yaw = np.array([[cy, -sy, 0],
                      [sy, cy, 0],
                      [0, 0, 1]])

    # Combine individual rotation matrices
    R = np.dot(R_yaw, np.dot(R_pitch, R_roll))
    print (R)
    r = np.linalg.inv(R)

    return r

#######for ikfom
fig, axs = plt.subplots(4,2)
lab_pre = ['', 'pre-x', 'pre-y', 'pre-z']
lab_out = ['', 'out-x', 'out-y', 'out-z']

lab_point = ['', 'point-x', 'point-y', 'point-z']
plot_ind = range(7,10)
a_fast_rs=np.loadtxt('mat_out_fast.txt')
a_fast=np.loadtxt('fast.txt')
a_point=np.loadtxt('point.txt')
a_point_rs=np.loadtxt('point_rs.txt')
time=a_fast_rs[:,0]
axs[0,0].set_title('Attitude')
axs[1,0].set_title('Translation')
axs[2,0].set_title('Extrins-R')
axs[3,0].set_title('Extrins-T')
axs[0,1].set_title('Velocity')
axs[1,1].set_title('bg')
axs[2,1].set_title('ba')
axs[3,1].set_title('Gravity')
for i in range(1,4):
    for j in range(8):
        axs[j%4, j//4].plot(time, a_fast_rs[:,i+j*3],'.-', label=lab_pre[i])

        axs[j%4, j//4].plot(time, a_fast[:,i+j*3],'.-', label=lab_out[i])
for j in range(8):
    # axs[j].set_xlim(386,389)
    axs[j%4, j//4].grid()
    axs[j%4, j//4].legend()
plt.grid()
#######for ikfom#######

groud_truth = pd.read_csv('groundtruth_2013-01-10.csv')
# datax = []
# datay = []
# for row in groud_truth:
#     print(row)
#     datax.append(row[1])
#     datay.append(row[2])
data = groud_truth.iloc[:, 7:10].values
# print(data)
# datay = groud_truth.iloc[:, 8].values
# R0_inv = euler_to_rotation_matrix(-0.029788434570233 - (1 - 0.807 / 180) * np.pi, -0.010193528558576 + 0.166 / 180 * np.pi  , -0.148727726103393 - 90.703 / 180 * np.pi)
# R0_inv = euler_to_rotation_matrix( 0.807/180 * np.pi,   0.166/80 *np.pi  , -90.703 / 180 * np.pi)
R0_inv = euler_to_rotation_matrix( -0.029788434570233 , -0.010193528558576 , -0.148727726103393)

data_T = data.T
for i in range(len(data_T[0])):
    data_T[:, i] = np.dot(R0_inv  ,data_T[:, i])

plt.figure(2)
plt.plot(a_fast_rs[:, 1+3], a_fast_rs[:, 2+3], label='rs')
plt.plot(a_fast[:, 1+3], a_fast[:, 2+3], label='fast')
plt.plot(a_point[:, 1+3], a_point[:, 2+3], label='point')
plt.plot(a_point_rs[:, 1+3], a_point_rs[:, 2+3], label='a_point_rs')
plt.plot(data_T[0, :], data_T[1, :], label='true')
# plt.plot(data[:, 0], data[:, 1], label='true')


length = len(a_fast_rs[:,0])-1
lentrue = len(data_T[0]) -1

print("true ", data_T[:, lentrue], " rs-fast ", a_fast_rs[length, 4:7], " fast ", a_fast[length, 4:7], " point ", a_point[length, 4:7], " rs-point ", a_point_rs[length, 4:7])

plt.legend()
# print(np.cos(-0.147575680110183))

def euler_to_rotation_matrix(roll, pitch, yaw):
    # Calculate sin and cosine values of the angles
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    # Define rotation matrix
    R_roll = np.array([[1, 0, 0],
                       [0, cr, -sr],
                       [0, sr, cr]])

    R_pitch = np.array([[cp, 0, sp],
                        [0, 1, 0],
                        [-sp, 0, cp]])

    R_yaw = np.array([[cy, -sy, 0],
                      [sy, cy, 0],
                      [0, 0, 1]])
    print (R)
    print("11111111")
    # Combine individual rotation matrices
    R = np.dot(R_yaw, np.dot(R_pitch, R_roll))
    r = np.linalg.inv(R)
    
    return r


#### Draw IMU data
# fig, axs = plt.subplots(2)
# imu=np.loadtxt('imu.txt')
# time=imu[:,0]
# axs[0].set_title('Gyroscope')
# axs[1].set_title('Accelerameter')
# lab_1 = ['gyr-x', 'gyr-y', 'gyr-z']
# lab_2 = ['acc-x', 'acc-y', 'acc-z']
# for i in range(3):
#     # if i==1:
#     axs[0].plot(time, imu[:,i+1],'.-', label=lab_1[i])
#     axs[1].plot(time, imu[:,i+4],'.-', label=lab_2[i])
# for i in range(2):
#     # axs[i].set_xlim(386,389)
#     axs[i].grid()
#     axs[i].legend()
# plt.grid()

# #### Draw time calculation
# plt.figure(3)
# fig = plt.figure()
# font1 = {'family' : 'Times New Roman',
# 'weight' : 'normal',
# 'size'   : 12,
# }
# c="red"
# a_out1=np.loadtxt('Log/mat_out_time_indoor1.txt')
# a_out2=np.loadtxt('Log/mat_out_time_indoor2.txt')
# a_out3=np.loadtxt('Log/mat_out_time_outdoor.txt')
# # n = a_out[:,1].size
# # time_mean = a_out[:,1].mean()
# # time_se   = a_out[:,1].std() / np.sqrt(n)
# # time_err  = a_out[:,1] - time_mean
# # feat_mean = a_out[:,2].mean()
# # feat_err  = a_out[:,2] - feat_mean
# # feat_se   = a_out[:,2].std() / np.sqrt(n)
# ax1 = fig.add_subplot(111)
# ax1.set_ylabel('Effective Feature Numbers',font1)
# ax1.boxplot(a_out1[:,2], showfliers=False, positions=[0.9])
# ax1.boxplot(a_out2[:,2], showfliers=False, positions=[1.9])
# ax1.boxplot(a_out3[:,2], showfliers=False, positions=[2.9])
# ax1.set_ylim([0, 3000])

# ax2 = ax1.twinx()
# ax2.spines['right'].set_color('red')
# ax2.set_ylabel('Compute Time (ms)',font1)
# ax2.yaxis.label.set_color('red')
# ax2.tick_params(axis='y', colors='red')
# ax2.boxplot(a_out1[:,1]*1000, showfliers=False, positions=[1.1],boxprops=dict(color=c),capprops=dict(color=c),whiskerprops=dict(color=c))
# ax2.boxplot(a_out2[:,1]*1000, showfliers=False, positions=[2.1],boxprops=dict(color=c),capprops=dict(color=c),whiskerprops=dict(color=c))
# ax2.boxplot(a_out3[:,1]*1000, showfliers=False, positions=[3.1],boxprops=dict(color=c),capprops=dict(color=c),whiskerprops=dict(color=c))
# ax2.set_xlim([0.5, 3.5])
# ax2.set_ylim([0, 100])

# plt.xticks([1,2,3], ('Outdoor Scene', 'Indoor Scene 1', 'Indoor Scene 2'))
# # # print(time_se)
# # # print(a_out3[:,2])
# plt.grid()
# plt.savefig("time.pdf", dpi=1200)
plt.show()
