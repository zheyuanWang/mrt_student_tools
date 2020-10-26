from scipy.spatial.transform import Rotation as R
import h5py
import Tool_read
import matplotlib.pyplot as plt
import numpy as np

PATH_H5_FILE = '/skitti_xxr_pose_h5_intr2/kitti_01_d.h5'
ALL_SEQ = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
SEQ7=['07']
def _euler_winkle_filter(winkle):
    """
    absolute euler winkle output limit in range +-360, 361 transfered to 1,
    which lead to problem when I subtract euler absolute results to calculate the relative winkle
    This function return e.p. -3 for winkle = 357, return 1 for winkle = -359
    """
    if abs(winkle)>180:
        if winkle > 0 :
            filtered= winkle - 360
        else:
            filtered = winkle + 360
        return filtered
    else:
        return winkle

def projection(roottxt, seq, interval_betweenF, rooth5=None ,ifSave=False, ifPlotDistance=False, ifPlotRotation=False,ifPlotEulerwinkel=False,ifPlotAngle_alfa=False):
    """
    update 09.10: use absolute poses of head & tail frames to calculate relative pose of them.
    #todo:
        generate h5 folders for each interval_between_frames
        for frameID < interval, set pose to 0.
            i.e. the length of h5 matrix keeps the same for any interval_between_frames
    produce the (relative) dr 2d pose, output dir = current dir,
    relative means: current pose - the last pose (i.e. if the last pose need to be translated & rotated, this data need to be reversed)
    output h5_matrix would have same len as input, with 0 at the beginning,
        to fill up for the situsation i < interval_between_frames

    :param roottxt: the source folder where the absolute poses stored in txt
    :param rooth5: only used for comparation in the visualization
    """
    # we store the Relative pose to the past Frame in the current Frame R1=F1-F0
    # F0   F1 F2 F3 - frames
    # R0=0 R1 R2 R3 - relative poses

    flat_matrix = Tool_read.read_pose_txt(roottxt, seq)
    if rooth5 is not None:
        dr = Tool_read.read_pose_h5(rooth5, seq)
    np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})

    #init
    frameN=len(flat_matrix)-1  # starts from 0 end at frameN-1, thus total Nummer = frameN = len-1
                            # rows = len(matrix)    columns = len(matrix[0])
    dist_x_ab=np.zeros((1, frameN+interval_betweenF+5)) # +5 for safety reason
    dist_y_ab=np.zeros((1, frameN+interval_betweenF+5))
    dist_x_re=np.zeros((1, frameN))
    dist_y_re=np.zeros((1, frameN))
    rots_AB = np.zeros((1, frameN))
    rots_alfa_AB = np.zeros((1, frameN))
    rots_beta_AB = np.zeros((1, frameN))
    rots_AB_radian = np.zeros((1, frameN))
    cos_AB=np.zeros((1, frameN))
    sin_AB=np.zeros((1, frameN))
    dist_xy=np.zeros((1, frameN))

    rots_absolute=np.zeros((1, frameN+interval_betweenF+5))
    rots=np.zeros((1, frameN))
    rost_re_modified_by_AB=np.zeros((1, frameN))
    rots_other_ab=np.zeros((2, frameN))
    relative_rots_euler=np.zeros((2, frameN))
    rots_without_filter=np.zeros((1, frameN))


    for i in range(frameN):
        matg = np.reshape(flat_matrix[i,:], (3, 4)) # assert matg[0, 3]==flat_matrix[i,3] # assert matg[2, 3]==flat_matrix[i,11]
        # read in the absolute pose from origin data, one by one
            # so in the next step, all element at the tail of dist_x_ab is 0 -> negative index would get 0
        dist_x_ab[0, i] = matg[0, 3]
        dist_y_ab[0, i] = matg[2, 3]
        r3x3 = R.from_matrix(matg[:, 0:3])
        rots_absolute[0, i] = r3x3.as_euler('yxz', degrees=True)[0]  # in 180 degree

        # calculate relative x,y accordingly
        if i<interval_betweenF:
            dist_xy[0, i]=0
            rots[0, i]=0
        else:
            dist_x_re[0, i] = dist_x_ab[0, i] - dist_x_ab[0, i-interval_betweenF]
            dist_y_re[0, i] = dist_y_ab[0, i] - dist_y_ab[0, i-interval_betweenF]
            dist_xy[0, i] = np.sqrt(pow(dist_x_re[0, i],2) + pow(dist_y_re[0, i],2))  # output is Euclidean Distance >=0

            rots[0,i] =_euler_winkle_filter(rots_absolute[0,i]-rots_absolute[0, i-interval_betweenF])
            rots_without_filter[0,i]= rots_absolute[0,i]-rots_absolute[0, i-interval_betweenF]
            # --------plot------
            if ifPlotEulerwinkel:
                rots_other_ab[0, i] = r3x3.as_euler('yxz',degrees=True)[1]
                rots_other_ab[1, i] = r3x3.as_euler('yxz',degrees=True)[2]
                relative_rots_euler[0, i] = _euler_winkle_filter(rots_other_ab[0, i] - rots_other_ab[0, i - interval_betweenF])
                relative_rots_euler[1, i] =_euler_winkle_filter(rots_other_ab[1,i]-rots_other_ab[1, i-interval_betweenF])

            if ifPlotAngle_alfa:
                rots_AB_radian[0, i] = np.arctan2(dist_x_re[0, i], dist_y_re[0, i]) #### Kern tech ####
                rots_AB[0, i]=rots_AB_radian[0, i]*180/np.pi # radian -> 180 drgree

                rots_alfa_AB[0, i]=_euler_winkle_filter(rots_AB[0,i]-rots_absolute[0, i-interval_betweenF]) # shouldn't use, since algin to current frame i
                rots_beta_AB[0, i]=_euler_winkle_filter(rots_absolute[0, i]- rots_AB[0,i])
                # should use beta_with_interval instead of beta of neighboring frames
                cos_AB[0, i] = np.cos(rots_beta_AB[0, i]*np.pi/180) * dist_xy[0, i] # need radian
                sin_AB[0, i] = np.sin(rots_beta_AB[0, i]*np.pi/180) * dist_xy[0, i]
                #rost_re_modified_by_AB=

    # --------plot------distance
    if ifPlotDistance:
        plt.figure("distance_{}".format(seq))
        if False:
            plt.plot(dist_x_re[0, :], "y--", label="dist_x_re relative")
            plt.plot(dist_y_re[0, :], "g.", label="dist_y_re relative")
        plt.plot(dist_xy[0, :], "cx", label="dist_xy relative")
        plt.plot(dr[0, :], "r+", label="dist opensource REFERENCE")
        if ifPlotAngle_alfa:
            plt.plot(cos_AB[0, :] * dist_xy[0, :], "y+", label="cos_AB")
            plt.plot(sin_AB[0, :] * dist_xy[0, :], "b+", label="sin_AB")
        plt.legend()

    # --------plot------rotation
    if ifPlotRotation:
        plt.figure("2d rotation_{}".format(seq))
        plt.plot(dr[1,:],"cx",label="opensource relative")
        plt.plot(rots[0,:],"r+",label="euler relative")
        if 0: # plot dr_absolute
            plt.plot(rots_absolute[0,:],"y.",label="euler absolute")
        #plt.plot(rots_ab_yzx[0,:], 'k--',label="rots_ab_yzx")
        plt.legend()
        print(rots[0,4])



    # --------plot------alfa (angle between AB and B point's driving direction)
    if ifPlotAngle_alfa:
        plt.figure("absolute_angles_{}".format(seq))
        if 0: #if plot absolute
            plt.plot(rots_absolute[0, :], "y+", label="absolute driving direction")
            plt.plot(rots_AB[0, :], "bx", label="absolute AB direction")  #it's wrong, alfa should be near half of the rots
        plt.plot(rots[0,:],"r+",label="euler relative=rots")
        plt.plot(rots_alfa_AB[0,:], "k--", label="alfa")
        plt.plot(rots_beta_AB[0,:], "c+", label="beta")


    plt.legend()

    if ifPlotEulerwinkel:
        plt.figure("euler winkel yxz_{}".format(seq))
        plt.plot(rots_absolute[0, :], "k-", label= "absolute yaw angle") #"t (yaw) absolute")
        plt.plot(rots_other_ab[0, :], "b--", label= "absolute pitch angle") #"x (pitch) absolute")
        plt.plot(rots_other_ab[1, :], "c--", label= "absolute row angle") #"z (row) absolute")
        plt.legend()
        plt.savefig("/home/zwang/euler_winkel_yxz_{}.png".format(seq))
        #-----save---------------------------------------
        plt.figure("relative_euler_{}".format(seq))
        plt.plot(rots[0, :], "k--", label="relative yaw angle") #tmp
        #plt.plot(rots_without_filter[0, :], "r--", label="relative yaw angle without filter") #tmp
        axes = plt.gca()
        #axes.set_ylim([-360, 360])

        #plt.plot(relative_rots_euler[0,:],"c,",label="euler relativex")
        #plt.plot(relative_rots_euler[1,:],"b,",label="euler relativez")
        plt.legend()
        #plt.savefig("/home/zwang/relative_euler_withfilter_{}.png".format(seq))
        #-----save---------------------------------------

    if ifPlotRotation or ifPlotDistance or ifPlotEulerwinkel:

        plt.show()

    if ifSave: # save to yaml
        filename = "xpose" + str(seq) + ".h5"
        h5f = h5py.File(filename, "w")
        h5f.create_dataset('distances', data=dist_xy)
        h5f.create_dataset('rot_r', data=rots)
        h5f.create_dataset('rot_r', data=rots)

        h5f.close()
        print("wrote to", filename)

if __name__ == '__main__':
    # since the output is where this function is,
    # the py for generating h5 file should be in the corresponding output folders
    TXTROOT = "/home/zwang/deeplab/research/deeplab/datasets/skitti_posestxt/"  # original data
    H5ROOT = "/home/zwang/deeplab/research/deeplab/datasets/relative_poses/skitti_xxr_pose_h5_intr2"  # only as validation's plot reference
    # this py is only for valitation :
    #for seq in ALL_SEQ:
    #seq="07"
for seq in ALL_SEQ:
    projection(roottxt=TXTROOT,
               seq=seq,
               interval_betweenF=1,
               rooth5=H5ROOT,
               ifSave=False,
               ifPlotDistance=True,
               ifPlotRotation=False,
               ifPlotEulerwinkel=False,
               ifPlotAngle_alfa=True)


#seq 05 i2400 beta vibrated to +-150 : the car stopped bevor a cross
# & seq07 i680 : the car stopped bevor a cross
# & seq08 i4000 : the car stopped bevor a cross, shortly
# but they won't affect the result cause car is not moving, rot is independently calculated from 2 poses